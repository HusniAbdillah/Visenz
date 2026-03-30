"""
Supervision-based tracker manager for edge-vision-counter V2.
Pure AI and annotation logic using supervision library.
Strictly decoupled from video capture - takes ThreadedVideoReader and StateManager instances.
"""

import logging
import threading
import time
from typing import Dict, List, Optional, Tuple, Any

import cv2
import numpy as np
import supervision as sv
from supervision.draw.color import Color, ColorPalette

import config
from core.detector import VisionModel
from core.video_reader import ThreadedVideoReader
from core.state_manager import StateManager

logger = logging.getLogger(__name__)


class SupervisionTracker:
    """
    Supervision-based tracking and line crossing detection for a single camera.
    
    Features:
    - Uses sv.ByteTrack for robust tracking with extended track buffer
    - Uses sv.LineZone for line crossing detection
    - Uses sv.BoxAnnotator, sv.LineZoneAnnotator, sv.TraceAnnotator for visualization
    - Syncs crossing counts to StateManager
    """

    def __init__(
        self,
        camera_config: Dict[str, Any],
        video_reader: ThreadedVideoReader,
        state_manager: StateManager,
        model: VisionModel,
    ) -> None:
        """
        Initialize the supervision tracker for a single camera.

        Args:
            camera_config: Camera configuration dictionary from config.py.
            video_reader: ThreadedVideoReader instance for this camera.
            state_manager: StateManager instance for persisting counts.
            model: VisionModel instance for detection.
        """
        self.camera_id: str = camera_config.get("id", "unknown")
        self.camera_config: Dict[str, Any] = camera_config
        self.video_reader: ThreadedVideoReader = video_reader
        self.state_manager: StateManager = state_manager
        self.model: VisionModel = model

        self.orientation: str = camera_config.get("orientation", "vertical")
        self.line_ratio: float = camera_config.get("line_ratio", 0.5)
        self.in_direction: str = camera_config.get("in_direction", "left_to_right")
        self.frame_skip: int = camera_config.get("frame_skip", 3)

        self._running: bool = False
        self._processing_thread: Optional[threading.Thread] = None
        self._frame_count: int = 0
        self._last_annotated_frame: Optional[np.ndarray] = None
        self._frame_lock: threading.Lock = threading.Lock()

        self._line_zone: Optional[sv.LineZone] = None
        self._byte_tracker: Optional[sv.ByteTrack] = None
        self._box_annotator: Optional[sv.BoxAnnotator] = None
        self._label_annotator: Optional[sv.LabelAnnotator] = None
        self._trace_annotator: Optional[sv.TraceAnnotator] = None
        self._line_zone_annotator: Optional[sv.LineZoneAnnotator] = None

        self._local_in_count: int = 0
        self._local_out_count: int = 0
        self._last_line_in: int = 0
        self._last_line_out: int = 0

        self._initialized: bool = False

    def _initialize_supervision_components(self, frame_shape: Tuple[int, int]) -> None:
        """
        Initialize all supervision components based on frame dimensions.

        Args:
            frame_shape: Tuple of (height, width) of the frame.
        """
        frame_h, frame_w = frame_shape
        logger.info(
            "Camera '%s': Initializing supervision components for frame %dx%d",
            self.camera_id,
            frame_w,
            frame_h
        )

        line_start, line_end = self._calculate_line_coordinates(frame_w, frame_h)
        logger.info(
            "Camera '%s': Line coordinates: start=%s, end=%s",
            self.camera_id,
            line_start,
            line_end
        )

        self._line_zone = sv.LineZone(
            start=line_start,
            end=line_end,
            triggering_anchors=(sv.Position.CENTER,),
        )

        self._byte_tracker = sv.ByteTrack(
            track_activation_threshold=config.CONFIDENCE_THRESHOLD,
            lost_track_buffer=90,
            minimum_matching_threshold=config.IOU_THRESHOLD,
            frame_rate=config.CAMERA_FPS,
        )

        self._box_annotator = sv.BoxAnnotator(
            thickness=2,
            color=ColorPalette.from_hex(['#00FF00', '#FF0000', '#0000FF', '#FFFF00']),
        )

        self._label_annotator = sv.LabelAnnotator(
            text_thickness=1,
            text_scale=0.5,
            color=ColorPalette.from_hex(['#00FF00', '#FF0000', '#0000FF', '#FFFF00']),
        )

        self._trace_annotator = sv.TraceAnnotator(
            thickness=2,
            trace_length=20,
            position=sv.Position.CENTER,
            color=ColorPalette.from_hex(['#00FF00', '#FF0000', '#0000FF', '#FFFF00']),
        )

        self._line_zone_annotator = sv.LineZoneAnnotator(
            thickness=3,
            text_thickness=2,
            text_scale=0.8,
            color=Color.from_hex('#FFFFFF'),
        )

        self._initialized = True
        logger.info(
            "Camera '%s': Supervision components initialized successfully",
            self.camera_id
        )

    def _calculate_line_coordinates(
        self,
        frame_w: int,
        frame_h: int
    ) -> Tuple[sv.Point, sv.Point]:
        """
        Calculate line start and end coordinates based on orientation and direction.

        Args:
            frame_w: Frame width in pixels.
            frame_h: Frame height in pixels.

        Returns:
            Tuple of (start_point, end_point) as sv.Point objects.
        """
        if self.orientation == "vertical":
            x_pos = int(self.line_ratio * frame_w)
            if self.in_direction in ["left_to_right", "right_to_left"]:
                start = sv.Point(x=x_pos, y=0)
                end = sv.Point(x=x_pos, y=frame_h)
            else:
                start = sv.Point(x=x_pos, y=0)
                end = sv.Point(x=x_pos, y=frame_h)
        else:
            y_pos = int(self.line_ratio * frame_h)
            if self.in_direction in ["top_to_bottom", "bottom_to_top"]:
                start = sv.Point(x=0, y=y_pos)
                end = sv.Point(x=frame_w, y=y_pos)
            else:
                start = sv.Point(x=0, y=y_pos)
                end = sv.Point(x=frame_w, y=y_pos)

        return start, end

    def _determine_in_out_direction(self) -> Tuple[str, str]:
        """
        Determine which LineZone count corresponds to IN and OUT.

        Returns:
            Tuple of (in_attr, out_attr) indicating which line_zone attribute is IN/OUT.
        """
        if self.in_direction == "left_to_right":
            return ("in_count", "out_count")
        elif self.in_direction == "right_to_left":
            return ("out_count", "in_count")
        elif self.in_direction == "top_to_bottom":
            return ("in_count", "out_count")
        elif self.in_direction == "bottom_to_top":
            return ("out_count", "in_count")
        else:
            return ("in_count", "out_count")

    def start(self) -> None:
        """Start the processing thread."""
        if self._running:
            logger.warning("Camera '%s': Tracker already running", self.camera_id)
            return

        self._running = True
        self._processing_thread = threading.Thread(
            target=self._processing_loop,
            name=f"Tracker-{self.camera_id}",
            daemon=True
        )
        self._processing_thread.start()
        logger.info("Camera '%s': Supervision tracker started", self.camera_id)

    def _processing_loop(self) -> None:
        """Main processing loop for detection, tracking, and annotation."""
        logger.info("Camera '%s': Processing loop started", self.camera_id)

        while self._running:
            try:
                frame = self.video_reader.get_frame()
                
                if frame is None:
                    time.sleep(0.01)
                    continue

                self._frame_count += 1

                if self._frame_count % self.frame_skip != 0:
                    continue

                if not self._initialized:
                    frame_shape = (frame.shape[0], frame.shape[1])
                    self._initialize_supervision_components(frame_shape)

                annotated_frame = self._process_frame(frame)

                with self._frame_lock:
                    self._last_annotated_frame = annotated_frame

            except Exception as e:
                logger.error(
                    "Camera '%s': Error in processing loop: %s",
                    self.camera_id,
                    str(e),
                    exc_info=True
                )
                time.sleep(0.1)

        logger.info("Camera '%s': Processing loop ended", self.camera_id)

    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Process a single frame: detect, track, count crossings, annotate.

        Args:
            frame: Input BGR frame.

        Returns:
            Annotated frame.
        """
        if not self._initialized or self._line_zone is None or self._byte_tracker is None:
            logger.error("Camera '%s': Supervision components not initialized, skipping frame", self.camera_id)
            return frame

        detection_result = self.model.predict_with_tracking(
            frame,
            conf=config.CONFIDENCE_THRESHOLD,
            iou=config.IOU_THRESHOLD,
        )

        boxes = detection_result.get('boxes', [])
        confs = detection_result.get('confs', [])
        ids = detection_result.get('ids', [])

        if len(boxes) > 0:
            boxes_array = np.array(boxes) if not isinstance(boxes, np.ndarray) else boxes
            confs_array = np.array(confs) if not isinstance(confs, np.ndarray) else confs

            detections = sv.Detections(
                xyxy=boxes_array,
                confidence=confs_array,
                class_id=np.zeros(len(boxes_array), dtype=int),
            )

            if len(ids) > 0:
                ids_array = np.array(ids) if not isinstance(ids, np.ndarray) else ids
                detections.tracker_id = ids_array.astype(int)
            else:
                detections = self._byte_tracker.update_with_detections(detections)
        else:
            detections = sv.Detections.empty()

        crossed_in, crossed_out = self._line_zone.trigger(detections=detections)

        current_line_in = self._line_zone.in_count
        current_line_out = self._line_zone.out_count

        in_attr, out_attr = self._determine_in_out_direction()
        
        if in_attr == "in_count":
            actual_in = current_line_in
            actual_out = current_line_out
        else:
            actual_in = current_line_out
            actual_out = current_line_in

        new_in = actual_in - self._local_in_count
        new_out = actual_out - self._local_out_count

        if new_in > 0 or new_out > 0:
            self.state_manager.bulk_update(new_in, new_out)
            self._local_in_count = actual_in
            self._local_out_count = actual_out
            logger.info(
                "Camera '%s': Line crossing - IN: +%d (total: %d), OUT: +%d (total: %d)",
                self.camera_id,
                new_in,
                actual_in,
                new_out,
                actual_out
            )

        annotated_frame = frame.copy()

        if len(detections) > 0 and self._trace_annotator is not None:
            annotated_frame = self._trace_annotator.annotate(
                scene=annotated_frame,
                detections=detections
            )

        if len(detections) > 0 and self._box_annotator is not None:
            annotated_frame = self._box_annotator.annotate(
                scene=annotated_frame,
                detections=detections
            )

        if len(detections) > 0 and self._label_annotator is not None:
            if detections.tracker_id is not None:
                labels = [f"ID:{tid}" for tid in detections.tracker_id]
            else:
                labels = [f"P{i}" for i in range(len(detections))]

            annotated_frame = self._label_annotator.annotate(
                scene=annotated_frame,
                detections=detections,
                labels=labels
            )

        if self._line_zone_annotator is not None:
            annotated_frame = self._line_zone_annotator.annotate(
                frame=annotated_frame,
                line_counter=self._line_zone
            )

        global_stats = self.state_manager.get_stats()
        self._draw_stats_overlay(
            annotated_frame,
            global_stats['total_in'],
            global_stats['total_out'],
            global_stats['current_inside']
        )

        return annotated_frame

    def _draw_stats_overlay(
        self,
        frame: np.ndarray,
        total_in: int,
        total_out: int,
        current_inside: int
    ) -> None:
        """
        Draw statistics overlay on the frame.

        Args:
            frame: Frame to draw on (modified in place).
            total_in: Total IN count.
            total_out: Total OUT count.
            current_inside: Current inside count.
        """
        overlay_height = 100
        overlay = frame[:overlay_height, :].copy()
        cv2.rectangle(frame, (0, 0), (frame.shape[1], overlay_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.3, frame[:overlay_height, :], 0.7, 0, frame[:overlay_height, :])

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2

        cv2.putText(
            frame,
            f"Camera: {self.camera_id}",
            (10, 25),
            font,
            font_scale,
            (255, 255, 255),
            thickness
        )

        cv2.putText(
            frame,
            f"IN: {total_in}",
            (10, 55),
            font,
            font_scale,
            (0, 255, 0),
            thickness
        )

        cv2.putText(
            frame,
            f"OUT: {total_out}",
            (150, 55),
            font,
            font_scale,
            (0, 0, 255),
            thickness
        )

        display_inside = max(0, current_inside)
        cv2.putText(
            frame,
            f"INSIDE: {display_inside}",
            (10, 85),
            font,
            font_scale,
            (255, 255, 0),
            thickness
        )

        connection_status = "CONNECTED" if self.video_reader.is_connected() else "DISCONNECTED"
        color = (0, 255, 0) if self.video_reader.is_connected() else (0, 0, 255)
        cv2.putText(
            frame,
            connection_status,
            (frame.shape[1] - 180, 25),
            font,
            font_scale,
            color,
            thickness
        )

    def get_annotated_frame(self) -> Optional[np.ndarray]:
        """
        Get the latest annotated frame.

        Returns:
            Copy of latest annotated frame or None.
        """
        with self._frame_lock:
            if self._last_annotated_frame is not None:
                return self._last_annotated_frame.copy()
            return None

    def get_local_stats(self) -> Dict[str, Any]:
        """
        Get local statistics for this camera.

        Returns:
            Dictionary with local IN/OUT counts.
        """
        return {
            'camera_id': self.camera_id,
            'local_in': self._local_in_count,
            'local_out': self._local_out_count,
            'connected': self.video_reader.is_connected(),
            'reconnect_count': self.video_reader.get_reconnect_count(),
        }

    def stop(self) -> None:
        """Stop the processing thread."""
        logger.info("Camera '%s': Stopping supervision tracker", self.camera_id)
        self._running = False

        if self._processing_thread is not None:
            self._processing_thread.join(timeout=5.0)
            if self._processing_thread.is_alive():
                logger.warning(
                    "Camera '%s': Processing thread did not terminate within timeout",
                    self.camera_id
                )

        logger.info("Camera '%s': Supervision tracker stopped", self.camera_id)


class CameraManager:
    """
    Manager for multiple cameras with supervision trackers.
    Orchestrates all video readers and trackers, provides aggregated statistics.
    """

    def __init__(
        self,
        model: VisionModel,
        state_manager: StateManager
    ) -> None:
        """
        Initialize the camera manager.

        Args:
            model: Shared VisionModel instance for all cameras.
            state_manager: Shared StateManager instance for persisting counts.
        """
        self.model: VisionModel = model
        self.state_manager: StateManager = state_manager
        self._video_readers: Dict[str, ThreadedVideoReader] = {}
        self._trackers: Dict[str, SupervisionTracker] = {}
        self._running: bool = False
        self._display_thread: Optional[threading.Thread] = None
        self._lock: threading.Lock = threading.Lock()

        logger.info("CameraManager initialized")

    def add_camera(self, camera_config: Dict[str, Any]) -> bool:
        """
        Add a camera to the manager.

        Args:
            camera_config: Camera configuration dictionary.

        Returns:
            True if camera added successfully, False otherwise.
        """
        camera_id = camera_config.get("id", f"camera_{len(self._video_readers)}")
        camera_url = camera_config.get("url", "")

        if not camera_url:
            logger.error("Camera '%s': No URL provided", camera_id)
            return False

        if camera_id in self._video_readers:
            logger.warning("Camera '%s': Already exists, skipping", camera_id)
            return False

        with self._lock:
            video_reader = ThreadedVideoReader(
                camera_id=camera_id,
                camera_url=camera_url,
                buffer_size=config.CAMERA_BUFFER_SIZE
            )
            self._video_readers[camera_id] = video_reader

            tracker = SupervisionTracker(
                camera_config=camera_config,
                video_reader=video_reader,
                state_manager=self.state_manager,
                model=self.model,
            )
            self._trackers[camera_id] = tracker

        logger.info("Camera '%s' added to manager", camera_id)
        return True

    def start(self) -> None:
        """Start all video readers and trackers."""
        if self._running:
            logger.warning("CameraManager already running")
            return

        self._running = True

        with self._lock:
            for camera_id, video_reader in self._video_readers.items():
                logger.info("Starting video reader for camera '%s'", camera_id)
                video_reader.start()

            for camera_id, tracker in self._trackers.items():
                logger.info("Starting tracker for camera '%s'", camera_id)
                tracker.start()

        if config.ENABLE_DEBUG_DISPLAY:
            self._display_thread = threading.Thread(
                target=self._display_loop,
                name="DisplayLoop",
                daemon=True
            )
            self._display_thread.start()
            logger.info("Debug display loop started")

        logger.info("CameraManager started with %d cameras", len(self._video_readers))

    def _display_loop(self) -> None:
        """Display loop for showing annotated frames in debug mode."""
        logger.info("Display loop started")

        while self._running:
            try:
                with self._lock:
                    for camera_id, tracker in self._trackers.items():
                        frame = tracker.get_annotated_frame()
                        if frame is not None:
                            window_name = f"Camera: {camera_id}"
                            cv2.imshow(window_name, frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    logger.info("Display loop: 'q' pressed, stopping")
                    break

                time.sleep(0.03)

            except Exception as e:
                logger.error("Error in display loop: %s", str(e))
                time.sleep(0.1)

        cv2.destroyAllWindows()
        logger.info("Display loop ended")

    def get_aggregated_stats(self) -> Dict[str, Any]:
        """
        Get aggregated statistics from all cameras.

        Returns:
            Dictionary with global and per-camera statistics.
        """
        global_stats = self.state_manager.get_stats()
        
        camera_stats = []
        connected_count = 0

        with self._lock:
            for camera_id, tracker in self._trackers.items():
                local_stats = tracker.get_local_stats()
                camera_stats.append(local_stats)
                if local_stats.get('connected', False):
                    connected_count += 1

        return {
            'total_in': global_stats['total_in'],
            'total_out': global_stats['total_out'],
            'current_inside': global_stats['current_inside'],
            'camera_count': len(self._trackers),
            'connected_count': connected_count,
            'cameras': camera_stats,
        }

    def get_camera_stats(self, camera_id: str) -> Optional[Dict[str, Any]]:
        """
        Get statistics for a specific camera.

        Args:
            camera_id: Camera identifier.

        Returns:
            Dictionary with camera statistics or None if not found.
        """
        with self._lock:
            tracker = self._trackers.get(camera_id)
            if tracker:
                return tracker.get_local_stats()
            return None

    def stop(self) -> None:
        """Stop all video readers and trackers."""
        logger.info("Stopping CameraManager")
        self._running = False

        with self._lock:
            for camera_id, tracker in self._trackers.items():
                logger.info("Stopping tracker for camera '%s'", camera_id)
                tracker.stop()

            for camera_id, video_reader in self._video_readers.items():
                logger.info("Stopping video reader for camera '%s'", camera_id)
                video_reader.stop()

        if self._display_thread is not None:
            self._display_thread.join(timeout=3.0)

        cv2.destroyAllWindows()
        logger.info("CameraManager stopped")
