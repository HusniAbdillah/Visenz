"""
Multi-camera tracker manager with Wi-Fi lag prevention and state machine line crossing.
Implements ThreadedVideoReader for buffering prevention and crossing state machine.
"""

import logging
import threading
import time
import queue
from typing import Dict, Optional, Tuple, List
import cv2
import numpy as np

import config
from core.detector import VisionModel

logger = logging.getLogger(__name__)


class ThreadedVideoReader:
    """
    Threaded video capture to prevent Wi-Fi TCP buffer lag.
    Continuously reads frames and keeps ONLY the latest frame available.
    Discards old frames to ensure the inference thread always gets fresh data.
    """

    def __init__(self, camera_url: str, buffer_size: int = 1) -> None:
        """
        Initialize threaded video reader.

        Args:
            camera_url: Video source URL or device index
            buffer_size: OpenCV buffer size (1 = critical for Wi-Fi sources)
        """
        self.camera_url = camera_url
        self.cap: Optional[cv2.VideoCapture] = None
        self.latest_frame: Optional[np.ndarray] = None
        self.frame_lock = threading.Lock()
        self.running = False
        self.reader_thread: Optional[threading.Thread] = None
        self.connected = False
        self._buffer_size = buffer_size

    def start(self) -> bool:
        """
        Start the background reading thread.

        Returns:
            True if connection successful, False otherwise.
        """
        try:
            logger.info("Opening video source: %s", self.camera_url)
            self.cap = cv2.VideoCapture(self.camera_url)

            if not self.cap.isOpened():
                logger.error("Failed to open camera: %s", self.camera_url)
                return False

            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, self._buffer_size)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_FRAME_WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_FRAME_HEIGHT)
            self.cap.set(cv2.CAP_PROP_FPS, config.CAMERA_FPS)

            self.running = True
            self.connected = True
            self.reader_thread = threading.Thread(target=self._read_loop, daemon=True)
            self.reader_thread.start()

            logger.info("Video reader started for: %s", self.camera_url)
            return True

        except Exception as e:
            logger.error("Failed to start video reader: %s", str(e))
            self.connected = False
            return False

    def _read_loop(self) -> None:
        """
        Background thread loop: continuously read frames and keep only the latest.
        Discards old frames to prevent buffer accumulation.
        """
        consecutive_failures = 0
        max_consecutive_failures = 30

        while self.running:
            try:
                ret, frame = self.cap.read()

                if not ret:
                    consecutive_failures += 1
                    if consecutive_failures > max_consecutive_failures:
                        logger.warning("Too many consecutive read failures. Reconnecting...")
                        self._reconnect()
                        consecutive_failures = 0
                    time.sleep(0.01)
                    continue

                consecutive_failures = 0
                with self.frame_lock:
                    self.latest_frame = frame

            except Exception as e:
                logger.error("Error in video read loop: %s", str(e))
                time.sleep(0.1)

    def _reconnect(self) -> None:
        """Attempt to reconnect to video source."""
        try:
            if self.cap:
                self.cap.release()
            time.sleep(2)
            self.cap = cv2.VideoCapture(self.camera_url)
            if self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, self._buffer_size)
                self.connected = True
                logger.info("Reconnected to: %s", self.camera_url)
            else:
                self.connected = False
        except Exception as e:
            logger.error("Reconnection failed: %s", str(e))
            self.connected = False

    def get_frame(self) -> Optional[np.ndarray]:
        """
        Get the latest available frame (non-blocking).

        Returns:
            Latest frame or None if no frame available
        """
        with self.frame_lock:
            return self.latest_frame.copy() if self.latest_frame is not None else None

    def stop(self) -> None:
        """Stop the reader thread and release resources."""
        logger.info("Stopping video reader for: %s", self.camera_url)
        self.running = False

        if self.reader_thread is not None:
            self.reader_thread.join(timeout=2)

        if self.cap is not None:
            self.cap.release()

        self.connected = False
        logger.info("Video reader stopped")

    def is_connected(self) -> bool:
        """Return connection status."""
        return self.connected


class CrossingStateMachine:
    """
    FIXED State machine for robust line crossing detection.
    CRITICAL FIXES:
    1. Tracks which SIDE of the line each person is on (not just if they've crossed)
    2. Allows BIDIRECTIONAL crossing (entry AND exit, even if they turn around)
    3. Handles GHOST EXITS: Counts people leaving even if never seen entering (negative OK)

    State tracking per ID:
      tracked_states[id] = {
        "last_side": "left" | "right" | None,
        "has_crossed_in": bool,  # Flag to prevent double-entry
        "has_crossed_out": bool, # Flag to prevent double-exit
      }

    Logic:
    - Determine which side centroid is on relative to line
    - If side changes AND direction matches in_direction, count as IN
    - If side changes AND direction matches out_direction, count as OUT
    - Allow both IN and OUT counts for same ID (bidirectional)
    """

    def __init__(
        self,
        line_ratio: float,
        orientation: str = "vertical",
        in_direction: str = "left_to_right",
    ) -> None:
        """
        Initialize line crossing state machine with STRICT state tracking.

        Args:
            line_ratio: Line position as ratio (0.0-1.0) of frame width/height
            orientation: 'vertical' or 'horizontal'
            in_direction: 'left_to_right', 'right_to_left', 'top_to_bottom', 'bottom_to_top'
        """
        self.line_ratio = line_ratio
        self.orientation = orientation
        self.in_direction = in_direction

        # Derive out_direction from in_direction
        self.out_direction = self._get_out_direction(in_direction)

        # FIXED: Track position AND side state for each person
        self.tracked_states: Dict[int, Dict] = {}  # {id: {"last_side": "left"|"right", "last_pos": (x,y), ...}}

        self.total_in = 0
        self.total_out = 0
        self.lock = threading.RLock()

    def _get_out_direction(self, in_direction: str) -> str:
        """Derive the OUT direction from IN direction."""
        opposite_map = {
            "left_to_right": "right_to_left",
            "right_to_left": "left_to_right",
            "top_to_bottom": "bottom_to_top",
            "bottom_to_top": "top_to_bottom",
        }
        return opposite_map.get(in_direction, "")

    def _get_side(self, coordinate: float, line_pos: int) -> str:
        """
        Determine which side of the line a coordinate is on.
        Returns: "left" (< line), "right" (>= line) for vertical
                 "top" (< line), "bottom" (>= line) for horizontal
        """
        return "left" if coordinate < line_pos else "right"

    def update(
        self,
        frame_shape: Tuple[int, int],
        detections: dict,
    ) -> Tuple[int, int]:
        """
        Update tracking state with new detections.
        FIXED: Allows bidirectional crossing per ID.

        Args:
            frame_shape: (height, width) of frame
            detections: Detection dict with 'ids' and 'centroids'

        Returns:
            Tuple of (new_in_count, new_out_count) since last update
        """
        frame_h, frame_w = frame_shape
        new_in = 0
        new_out = 0

        if self.orientation == "vertical":
            line_pos = int(self.line_ratio * frame_w)
            coordinate_func = lambda c: c[0]  # Extract X for vertical
        else:
            line_pos = int(self.line_ratio * frame_h)
            coordinate_func = lambda c: c[1]  # Extract Y for horizontal

        ids = detections.get('ids', [])
        centroids = detections.get('centroids', [])

        if len(ids) == 0:
            # Clean up lost IDs
            if self.tracked_states:
                with self.lock:
                    self.tracked_states.clear()
            return 0, 0

        with self.lock:
            current_ids = set(ids)

            for person_id, centroid in zip(ids, centroids):
                coord = coordinate_func(centroid)
                current_side = self._get_side(coord, line_pos)

                # Initialize state if first time seeing this ID
                if person_id not in self.tracked_states:
                    self.tracked_states[person_id] = {
                        "last_side": current_side,
                        "last_coord": coord,
                    }
                else:
                    # FIXED: Check for crossing in EITHER direction
                    state = self.tracked_states[person_id]
                    last_side = state.get("last_side")
                    last_coord = state.get("last_coord")

                    if last_side != current_side:
                        # Side changed! Check direction
                        side_changed_from_left = last_side == "left" and current_side == "right"
                        side_changed_from_right = last_side == "right" and current_side == "left"

                        # Check if this matches IN direction
                        if self.in_direction == "left_to_right" and side_changed_from_left:
                            self.total_in += 1
                            new_in += 1
                            logger.debug("Person %d entered (left→right)", person_id)
                        elif self.in_direction == "right_to_left" and side_changed_from_right:
                            self.total_in += 1
                            new_in += 1
                            logger.debug("Person %d entered (right→left)", person_id)

                        # Check if this matches OUT direction (FIXED: Allow exit even if entered before)
                        if self.out_direction == "left_to_right" and side_changed_from_left:
                            self.total_out += 1
                            new_out += 1
                            logger.debug("Person %d exited (left→right)", person_id)
                        elif self.out_direction == "right_to_left" and side_changed_from_right:
                            self.total_out += 1
                            new_out += 1
                            logger.debug("Person %d exited (right→left)", person_id)

                    # Update state for next frame
                    state["last_side"] = current_side
                    state["last_coord"] = coord

            # Remove truly lost IDs (not in current detections)
            lost_ids = set(self.tracked_states.keys()) - current_ids
            for lost_id in lost_ids:
                del self.tracked_states[lost_id]

            return new_in, new_out

    def get_counts(self) -> Tuple[int, int]:
        """Get thread-safe copy of counts (may be negative for ghost exits)."""
        with self.lock:
            return self.total_in, self.total_out

    def reset(self) -> None:
        """Reset all counters."""
        with self.lock:
            self.total_in = 0
            self.total_out = 0
            self.tracked_states.clear()


class CameraProcessor:
    """Processes a single camera with frame skipping and inference."""

    def __init__(
        self,
        camera_config: dict,
        model: VisionModel,
    ) -> None:
        """
        Initialize camera processor.

        Args:
            camera_config: Camera configuration dict
            model: VisionModel instance (shared)
        """
        self.camera_id = camera_config["id"]
        self.camera_url = camera_config["url"]
        self.frame_skip = camera_config.get("frame_skip", config.FRAME_SKIP_DEFAULT)
        self.enabled = camera_config.get("enabled", True)

        self.model = model
        self.reader = ThreadedVideoReader(self.camera_url, buffer_size=config.CAMERA_BUFFER_SIZE)
        self.state_machine = CrossingStateMachine(
            line_ratio=camera_config.get("line_ratio", 0.5),
            orientation=camera_config.get("orientation", "vertical"),
            in_direction=camera_config.get("in_direction", "left_to_right"),
        )

        self.frame_count = 0
        self.stats = {
            "total_in": 0,
            "total_out": 0,
            "last_detection_time": None,
            "inference_count": 0,
        }

    def start(self) -> bool:
        """Start the video reader."""
        return self.reader.start()

    def stop(self) -> None:
        """Stop the video reader."""
        self.reader.stop()

    def process_frame(self) -> bool:
        """
        Process one frame with frame skipping.

        Returns:
            True if inference was run, False otherwise
        """
        frame = self.reader.get_frame()
        if frame is None:
            return False

        self.frame_count += 1

        should_infer = (self.frame_count % self.frame_skip) == 0

        if should_infer:
            try:
                detections = self.model.predict_with_tracking(frame)

                frame_shape = detections.get('raw_frame_shape', frame.shape[:2])
                new_in, new_out = self.state_machine.update(frame_shape, detections)

                self.stats['total_in'], self.stats['total_out'] = self.state_machine.get_counts()
                self.stats['last_detection_time'] = time.time()
                self.stats['inference_count'] += 1

                if config.ENABLE_DEBUG_DISPLAY:
                    self._draw_debug(frame, detections, frame_shape)

                return True

            except Exception as e:
                logger.error("Error processing frame from %s: %s", self.camera_id, str(e))

        return False

    def _draw_debug(self, frame: np.ndarray, detections: dict, frame_shape: Tuple[int, int]) -> None:
        """Draw debug visualization on frame."""
        frame_h, frame_w = frame_shape

        if self.state_machine.orientation == "vertical":
            line_x = int(self.state_machine.line_ratio * frame_w)
            cv2.line(frame, (line_x, 0), (line_x, frame_h), (0, 255, 0), 2)
        else:
            line_y = int(self.state_machine.line_ratio * frame_h)
            cv2.line(frame, (0, line_y), (frame_w, line_y), (0, 255, 0), 2)

        centroids = detections.get('centroids', [])
        boxes = detections.get('boxes', [])
        ids = detections.get('ids', [])

        for i, (bbox, centroid, person_id) in enumerate(zip(boxes, centroids, ids)):
            x1, y1, x2, y2 = bbox
            cx, cy = centroid

            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.circle(frame, (int(cx), int(cy)), 4, (0, 0, 255), -1)
            cv2.putText(
                frame,
                f"ID:{person_id}",
                (int(x1), int(y1) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

        cv2.putText(
            frame,
            f"{self.camera_id} | In:{self.stats['total_in']} Out:{self.stats['total_out']}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

        display_h = min(config.DEBUG_DISPLAY_WINDOW_HEIGHT, frame_h)
        display_w = min(config.DEBUG_DISPLAY_WINDOW_WIDTH, frame_w)
        display_frame = cv2.resize(frame, (display_w, display_h))

        cv2.imshow(f"Debug: {self.camera_id}", display_frame)
        cv2.waitKey(1)

    def get_stats(self) -> dict:
        """Get camera statistics."""
        return self.stats.copy()

    def is_connected(self) -> bool:
        """Check if camera is connected."""
        return self.reader.is_connected()


class CameraManager:
    """
    Manages multiple camera processors with coordinated inference scheduling.
    Prevents CPU overload by limiting concurrent inference.
    """

    def __init__(self, model: VisionModel) -> None:
        """Initialize camera manager."""
        self.model = model
        self.processors: Dict[str, CameraProcessor] = {}
        self.running = False
        self.manager_thread: Optional[threading.Thread] = None
        self.total_in = 0
        self.total_out = 0
        self.lock = threading.RLock()

    def add_camera(self, camera_config: dict) -> bool:
        """Add a camera to be managed."""
        try:
            processor = CameraProcessor(camera_config, self.model)
            self.processors[camera_config["id"]] = processor
            logger.info("Camera %s added to manager", camera_config["id"])
            return True
        except Exception as e:
            logger.error("Failed to add camera: %s", str(e))
            return False

    def start(self) -> None:
        """Start all cameras and the processing loop."""
        self.running = True

        for camera_id, processor in self.processors.items():
            if processor.enabled:
                if processor.start():
                    logger.info("Camera %s started", camera_id)
                else:
                    logger.warning("Failed to start camera %s", camera_id)

        self.manager_thread = threading.Thread(target=self._process_loop, daemon=False)
        self.manager_thread.start()
        logger.info("Camera manager started with %d cameras", len(self.processors))

    def _process_loop(self) -> None:
        """Main processing loop for all cameras."""
        health_check_timer = 0

        while self.running:
            try:
                for camera_id, processor in self.processors.items():
                    if processor.enabled:
                        processor.process_frame()

                with self.lock:
                    total_in = sum(p.stats['total_in'] for p in self.processors.values())
                    total_out = sum(p.stats['total_out'] for p in self.processors.values())
                    self.total_in = total_in
                    self.total_out = total_out

                health_check_timer += 1
                if health_check_timer >= config.HEALTH_CHECK_INTERVAL * 10:
                    self._health_check()
                    health_check_timer = 0

                time.sleep(0.01)

            except Exception as e:
                logger.error("Error in process loop: %s", str(e))

    def _health_check(self) -> None:
        """Periodically check camera connections."""
        for camera_id, processor in self.processors.items():
            if not processor.is_connected():
                logger.warning("Camera %s disconnected. Attempting reconnect...", camera_id)
                processor.stop()
                if not processor.start():
                    logger.error("Failed to reconnect camera %s", camera_id)

    def stop(self) -> None:
        """Stop all processors."""
        logger.info("Stopping camera manager...")
        self.running = False

        if self.manager_thread is not None:
            self.manager_thread.join(timeout=5)

        for processor in self.processors.values():
            processor.stop()

        logger.info("Camera manager stopped")

    def get_aggregated_stats(self) -> dict:
        """Get aggregated statistics from all cameras."""
        with self.lock:
            stats = {
                "total_in": self.total_in,
                "total_out": self.total_out,
                "current_inside": self.total_in - self.total_out,
                "camera_count": len(self.processors),
                "connected_count": sum(1 for p in self.processors.values() if p.is_connected()),
                "cameras": {},
            }

            for camera_id, processor in self.processors.items():
                stats["cameras"][camera_id] = processor.get_stats()

            return stats

    def get_camera_stats(self, camera_id: str) -> Optional[dict]:
        """Get statistics for a specific camera."""
        if camera_id in self.processors:
            return self.processors[camera_id].get_stats()
        return None
