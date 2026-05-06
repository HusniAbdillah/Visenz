"""
Supervision-based tracker manager for edge-vision-counter V2.
Pure AI and annotation logic using supervision library.
Strictly decoupled from video capture - takes ThreadedVideoReader and StateManager instances.
"""

import logging
import threading
import time
from contextlib import nullcontext
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
        self._line_start: Optional[sv.Point] = None
        self._line_end: Optional[sv.Point] = None
        self._count_lock: threading.Lock = threading.Lock()

        self._initialized: bool = False

    def _normalize_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Normalize frame size based on configuration.

        Modes:
        - none: return original frame
        - letterbox: keep aspect ratio + pad (no crop, no distortion)
        - stretch: direct resize to target (no crop, possible distortion)
        """
        mode = str(getattr(config, "FRAME_NORMALIZATION_MODE", "none")).lower()
        target_w = int(getattr(config, "FRAME_NORMALIZATION_WIDTH", frame.shape[1]))
        target_h = int(getattr(config, "FRAME_NORMALIZATION_HEIGHT", frame.shape[0]))

        if mode == "none":
            return frame

        if target_w <= 0 or target_h <= 0:
            return frame

        if mode == "stretch":
            return cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

        if mode == "letterbox":
            src_h, src_w = frame.shape[:2]
            if src_w <= 0 or src_h <= 0:
                return frame

            scale = min(target_w / src_w, target_h / src_h)
            new_w = max(1, int(round(src_w * scale)))
            new_h = max(1, int(round(src_h * scale)))

            resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            canvas = np.zeros((target_h, target_w, 3), dtype=frame.dtype)

            x0 = (target_w - new_w) // 2
            y0 = (target_h - new_h) // 2
            canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
            return canvas

        return frame

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
        self._line_start = line_start
        self._line_end = line_end
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

        # ByteTrack operates on the inference cadence, not the raw camera FPS.
        # This preserves IDs through brief Wi-Fi jitter and lateral occlusion
        # without keeping lost tracks alive so long that ghosts dominate.
        effective_tracker_fps = max(1, int(round(float(config.CAMERA_FPS) / max(1, self.frame_skip))))

        self._byte_tracker = sv.ByteTrack(
            track_activation_threshold=float(getattr(config, "BYTE_TRACK_TRACK_THRESH", 0.25)),
            lost_track_buffer=int(getattr(config, "TRACK_PERSISTENCE", 45)),
            minimum_matching_threshold=float(getattr(config, "BYTE_TRACK_MATCH_THRESH", 0.70)),
            frame_rate=effective_tracker_fps,
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
            # vertical line runs top->bottom by default; swap endpoints
            # so the LineZone orientation matches the configured in_direction
            start = sv.Point(x=x_pos, y=0)
            end = sv.Point(x=x_pos, y=frame_h)
            if self.in_direction == "right_to_left":
                start, end = end, start
        else:
            y_pos = int(self.line_ratio * frame_h)
            # horizontal line runs left->right by default; swap endpoints
            # when configured for bottom_to_top so annotations/direction
            # align with the configured in_direction
            start = sv.Point(x=0, y=y_pos)
            end = sv.Point(x=frame_w, y=y_pos)
            if self.in_direction == "bottom_to_top":
                start, end = end, start

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

    def reset_counts(self) -> None:
        """Reset local camera counters and line-zone state for a new session."""
        with self._count_lock:
            self._local_in_count = 0
            self._local_out_count = 0

            if self._line_start is not None and self._line_end is not None:
                self._line_zone = sv.LineZone(
                    start=self._line_start,
                    end=self._line_end,
                    triggering_anchors=(sv.Position.CENTER,),
                )

        logger.info("Camera '%s': Local counters reset", self.camera_id)

    def _boost_edge_confidence(
        self,
        detections: sv.Detections,
        frame_w: int,
        frame_h: int,
    ) -> sv.Detections:
        """
        PHASE 1: Edge-Aware Confidence Boost
        
        Boost confidence for detections at frame edges (partial visibility).
        YOLOv8n produces lower confidence for partial objects; this recovers
        confidence for valid edge detections based on spatial position.
        
        Args:
            detections: Input detections from YOLOv8n
            frame_w: Frame width in pixels
            frame_h: Frame height in pixels
            
        Returns:
            Detections with boosted confidence for edge cases
        """
        if len(detections) == 0:
            return detections

        edge_margin = int(getattr(config, "EDGE_ZONE_MARGIN_PX", 80))
        confidence_floor = float(getattr(config, "EDGE_OCCLUSION_CONFIDENCE_FLOOR", 0.20))
        confidence_boost = float(getattr(config, "EDGE_OCCLUSION_CONFIDENCE_BOOST", 0.10))

        # Be defensive: detections.confidence may be None or not a numpy array
        xyxy = np.array(detections.xyxy, copy=True)
        conf_attr = getattr(detections, 'confidence', None)
        if conf_attr is None:
            # Nothing to boost
            return detections
        confidence = np.array(conf_attr, copy=True)
        
        for i, (x1, y1, x2, y2) in enumerate(xyxy):
            # Check if bbox is at frame edge
            at_left_edge = x1 <= edge_margin
            at_right_edge = x2 >= (frame_w - edge_margin)
            at_top_edge = y1 <= edge_margin
            at_bottom_edge = y2 >= (frame_h - edge_margin)
            
            is_at_edge = at_left_edge or at_right_edge or at_top_edge or at_bottom_edge
            
            # Only boost if confidence is above floor and at edge
            if is_at_edge and confidence[i] > confidence_floor:
                # Boost based on visibility ratio (more partial = more boost)
                bbox_area = (x2 - x1) * (y2 - y1)
                frame_area = frame_w * frame_h
                visibility_ratio = bbox_area / frame_area if frame_area > 0 else 1.0
                
                # More boost for more partial (lower visibility)
                boost_factor = confidence_boost * (1.5 - visibility_ratio)
                new_conf = min(0.95, confidence[i] + boost_factor)
                confidence[i] = new_conf

                try:
                    logger.debug(
                        "Camera '%s': Boosted edge detection conf: %.2f → %.2f (visibility: %.1f%%)",
                        self.camera_id,
                        float(confidence[i] - boost_factor),
                        float(new_conf),
                        float(visibility_ratio * 100),
                    )
                except Exception:
                    # Ignore logging formatting errors
                    pass
        
        # Assign back the boosted confidences
        try:
            detections.confidence = confidence
        except Exception:
            # If assignment fails, construct a new Detections object preserving tracker_id/class_id
            new_det = sv.Detections(xyxy=xyxy, confidence=confidence, class_id=getattr(detections, 'class_id', None))
            if getattr(detections, 'tracker_id', None) is not None:
                new_det.tracker_id = np.array(getattr(detections, 'tracker_id'))
            return new_det

        return detections

    def _validate_crossing_with_motion(
        self,
        line_zone: sv.LineZone,
        detections: sv.Detections,
        frame_w: int,
        frame_h: int,
    ) -> Tuple[int, int]:
        """
        PHASE 2: Motion-Aware Crossing Validation
        
        Use ByteTrack's tracking history to detect crossings even with
        brief detection gaps (occlusion). When a track is re-acquired
        after being lost, check if the motion trajectory crosses the line.
        
        Args:
            line_zone: LineZone instance for crossing detection
            detections: Current frame detections
            frame_w: Frame width
            frame_h: Frame height
            
        Returns:
            Tuple of (in_count, out_count) from primary + motion-based crossings
        """
        if line_zone is None or self._byte_tracker is None:
            return 0, 0
        
        # Primary crossing detection
        crossed_in, crossed_out = line_zone.trigger(detections)
        in_count = int(np.count_nonzero(crossed_in))
        out_count = int(np.count_nonzero(crossed_out))
        
        # Secondary: Check momentum-based crossings for re-emerging tracks
        motion_extrapolation = bool(getattr(config, "MOTION_EXTRAPOLATION_ENABLED", True))

        if motion_extrapolation:
            # ByteTrack internal container name may vary; try common attributes
            track_container = getattr(self._byte_tracker, 'tracked_tracks', None) or getattr(self._byte_tracker, 'tracks', None) or []
            reacq_distance = int(getattr(config, "LOST_TRACK_REACQ_DISTANCE_PX", 150))

            for track in track_container:
                try:
                    time_since_update = getattr(track, 'time_since_update', None)
                    # Consider recently re-acquired tracks (time_since_update == 0 or small)
                    if time_since_update is None or time_since_update > 3:
                        continue

                    # Try to get previous and current positions from common attributes
                    # Many STrack-like objects expose .mean or .tlwh or a history list
                    if hasattr(track, 'mean') and track.mean is not None:
                        mean = getattr(track, 'mean')
                        # mean may be a length-4 vector [cx, cy, w, h] or similar
                        if len(mean) >= 2:
                            curr_x, curr_y = float(mean[0]), float(mean[1])
                        else:
                            continue
                    elif hasattr(track, 'tlbr') and track.tlbr is not None:
                        tlbr = getattr(track, 'tlbr')
                        curr_x = float((tlbr[0] + tlbr[2]) / 2.0)
                        curr_y = float((tlbr[1] + tlbr[3]) / 2.0)
                    else:
                        continue

                    # Previous position: try multiple fallbacks
                    prev_pos = None
                    if hasattr(track, '_last_pos'):
                        prev_pos = getattr(track, '_last_pos')
                    elif hasattr(track, 'last_tlbr'):
                        lt = getattr(track, 'last_tlbr')
                        prev_pos = ((lt[0] + lt[2]) / 2.0, (lt[1] + lt[3]) / 2.0)
                    elif hasattr(track, 'history') and len(getattr(track, 'history')) >= 1:
                        hist = getattr(track, 'history')
                        prev = hist[-1]
                        if len(prev) >= 2:
                            prev_pos = (float(prev[0]), float(prev[1]))

                    if prev_pos is None:
                        continue

                    prev_x, prev_y = float(prev_pos[0]), float(prev_pos[1])

                    # Line endpoints may be attributes or properties
                    ls = getattr(line_zone, 'start', None) or getattr(line_zone, 'start_point', None)
                    le = getattr(line_zone, 'end', None) or getattr(line_zone, 'end_point', None)
                    if ls is None or le is None:
                        continue

                    # Extract numeric coords
                    ls_x, ls_y = float(getattr(ls, 'x', ls[0] if isinstance(ls, (list, tuple)) else 0)), float(getattr(ls, 'y', ls[1] if isinstance(ls, (list, tuple)) else 0))
                    le_x, le_y = float(getattr(le, 'x', le[0] if isinstance(le, (list, tuple)) else 0)), float(getattr(le, 'y', le[1] if isinstance(le, (list, tuple)) else 0))

                    crosses = self._segment_intersects_line(
                        (prev_x, prev_y),
                        (curr_x, curr_y),
                        (ls_x, ls_y),
                        (le_x, le_y),
                    )

                    if crosses:
                        direction = self._determine_motion_direction((prev_x, prev_y), (curr_x, curr_y))
                        if direction == "in":
                            in_count += 1
                        else:
                            out_count += 1

                        track_id = getattr(track, 'track_id', getattr(track, 'id', None))
                        try:
                            logger.info(
                                "Camera '%s': Motion extrapolation crossing detected - Track %s: %s (pos: %.0f,%.0f)",
                                self.camera_id,
                                str(track_id),
                                direction,
                                curr_x,
                                curr_y,
                            )
                        except Exception:
                            pass

                except Exception:
                    # Be robust: don't let track attribute differences stop processing
                    continue
        
        return in_count, out_count

    def _segment_intersects_line(
        self,
        p1: Tuple[float, float],
        p2: Tuple[float, float],
        line_start: Tuple[float, float],
        line_end: Tuple[float, float],
    ) -> bool:
        """
        Check if a segment (p1->p2) intersects with a line.
        
        Args:
            p1: Start point (x, y)
            p2: End point (x, y)
            line_start: Line start point (x, y)
            line_end: Line end point (x, y)
            
        Returns:
            True if segment crosses the line
        """
        def ccw(A, B, C):
            """Counter-clockwise check for line intersection."""
            return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])
        
        # Check if segment p1-p2 intersects with line_start-line_end
        return ccw(p1, line_start, line_end) != ccw(p2, line_start, line_end) and \
               ccw(p1, p2, line_start) != ccw(p1, p2, line_end)

    def _determine_motion_direction(
        self,
        p1: Tuple[float, float],
        p2: Tuple[float, float],
    ) -> str:
        """
        Determine crossing direction based on motion vector and configured in_direction.
        
        Args:
            p1: Previous position
            p2: Current position
            
        Returns:
            "in" or "out" based on motion direction
        """
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        
        if self.orientation == "vertical":
            # Vertical line: check horizontal motion
            if self.in_direction == "left_to_right":
                return "in" if dx > 0 else "out"
            else:  # right_to_left
                return "in" if dx < 0 else "out"
        else:  # horizontal orientation
            # Horizontal line: check vertical motion
            if self.in_direction == "top_to_bottom":
                return "in" if dy > 0 else "out"
            else:  # bottom_to_top
                return "in" if dy < 0 else "out"

    def _expand_detection_roi(
        self,
        detections: sv.Detections,
        frame_w: int,
        frame_h: int,
    ) -> sv.Detections:
        """
        PHASE 3: Edge-Aware ROI Expansion
        
        Allow detections slightly outside frame bounds for people who are
        partially off-screen. This enables tracking and line crossing detection
        for people whose center is visible but edges are clipped.
        
        Args:
            detections: Input detections
            frame_w: Frame width
            frame_h: Frame height
            
        Returns:
            Detections with centers inside frame (edges may be outside)
        """
        if len(detections) == 0:
            return detections
        
        expand_px = int(getattr(config, "VIRTUAL_EDGE_EXPANSION_PX", 100))
        xyxy = detections.xyxy.copy()
        
        # Calculate centers
        centers_x = (xyxy[:, 0] + xyxy[:, 2]) / 2
        centers_y = (xyxy[:, 1] + xyxy[:, 3]) / 2
        
        # Keep only bboxes with center inside frame
        valid_mask = (
            (centers_x >= 0) & (centers_x < frame_w) &
            (centers_y >= 0) & (centers_y < frame_h)
        )
        
        # Clamp bbox edges (allow outside but with limit)
        xyxy[:, 0] = np.maximum(xyxy[:, 0], -expand_px)
        xyxy[:, 2] = np.minimum(xyxy[:, 2], frame_w + expand_px)
        xyxy[:, 1] = np.maximum(xyxy[:, 1], -expand_px)
        xyxy[:, 3] = np.minimum(xyxy[:, 3], frame_h + expand_px)
        
        # Update detections with expanded ROI and valid mask filter
        detections_xyxy = xyxy
        confidences = getattr(detections, 'confidence', None)

        # Build a new Detections object filtered by valid_mask to avoid typing issues
        if confidences is None:
            new_conf = None
        else:
            new_conf = np.array(confidences, copy=True)

        # Apply mask
        mask = np.array(valid_mask, dtype=bool)
        filtered_xyxy = detections_xyxy[mask]
        if new_conf is not None:
            filtered_conf = new_conf[mask]
        else:
            filtered_conf = None

        new_det = sv.Detections(xyxy=filtered_xyxy, confidence=filtered_conf, class_id=getattr(detections, 'class_id', None))
        if getattr(detections, 'tracker_id', None) is not None:
            new_det.tracker_id = np.array(getattr(detections, 'tracker_id'))[mask]

        return new_det

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

                frame = self._normalize_frame(frame)

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
        Includes occlusion-robust detection handling (Phases 1-3).

        Args:
            frame: Input BGR frame.

        Returns:
            Annotated frame.
        """
        if not self._initialized or self._line_zone is None or self._byte_tracker is None:
            logger.error("Camera '%s': Supervision components not initialized, skipping frame", self.camera_id)
            return frame

        frame_h, frame_w = frame.shape[:2]

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

            # PHASE 3: Expand detection ROI for partial objects
            detections = self._expand_detection_roi(detections, frame_w, frame_h)
            
            # PHASE 1: Boost confidence for edge detections
            detections = self._boost_edge_confidence(detections, frame_w, frame_h)

            if len(ids) > 0:
                ids_array = np.array(ids) if not isinstance(ids, np.ndarray) else ids
                detections.tracker_id = ids_array.astype(int)
            else:
                detections = self._byte_tracker.update_with_detections(detections)
        else:
            detections = sv.Detections.empty()

        session_guard = self.state_manager.session_mutation() if self.state_manager is not None else nullcontext()
        with session_guard:
            with self._count_lock:
                if self._line_zone is not None:
                    # PHASE 2: Motion-aware crossing validation
                    # Combines primary LineZone detection with track momentum analysis
                    line_in_count, line_out_count = self._validate_crossing_with_motion(
                        self._line_zone, detections, frame_w, frame_h
                    )

                    in_attr, out_attr = self._determine_in_out_direction()
                    if in_attr == "in_count" and out_attr == "out_count":
                        new_in, new_out = line_in_count, line_out_count
                    else:
                        new_in, new_out = line_out_count, line_in_count
                else:
                    new_in, new_out = 0, 0

                if new_in > 0 or new_out > 0:
                    if self.state_manager is not None:
                        if new_in > 0:
                            self.state_manager.record_crossing(
                                camera_id=self.camera_id,
                                direction="IN",
                                count=new_in,
                            )
                        if new_out > 0:
                            self.state_manager.record_crossing(
                                camera_id=self.camera_id,
                                direction="OUT",
                                count=new_out,
                            )
                    self._local_in_count += new_in
                    self._local_out_count += new_out
                    logger.info(
                        "Camera '%s': Vector crossing - IN: +%d (total: %d), OUT: +%d (total: %d)",
                        self.camera_id,
                        new_in,
                        self._local_in_count,
                        new_out,
                        self._local_out_count,
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

        if self.state_manager is not None:
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
        if self.state_manager is None:
            logger.warning("StateManager not available for aggregated stats")
            global_stats = {
                'total_in': 0,
                'total_out': 0,
                'current_inside': 0
            }
            camera_summary = []
        else:
            global_stats = self.state_manager.get_stats()
            analytics = self.state_manager.get_analytics()
            camera_summary = analytics.get('camera_summary', [])
        
        camera_stats = []
        connected_count = 0
        camera_summary_map = {item['camera_id']: item for item in camera_summary}

        with self._lock:
            for camera_id, tracker in self._trackers.items():
                summary = camera_summary_map.get(camera_id, {})
                local_stats = {
                    'camera_id': camera_id,
                    'local_in': int(summary.get('total_in', 0)),
                    'local_out': int(summary.get('total_out', 0)),
                    'connected': tracker.video_reader.is_connected(),
                    'reconnect_count': tracker.video_reader.get_reconnect_count(),
                }
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
                if self.state_manager is None:
                    return tracker.get_local_stats()

                analytics = self.state_manager.get_analytics()
                summary_map = {item['camera_id']: item for item in analytics.get('camera_summary', [])}
                summary = summary_map.get(camera_id, {})
                return {
                    'camera_id': camera_id,
                    'local_in': int(summary.get('total_in', 0)),
                    'local_out': int(summary.get('total_out', 0)),
                    'connected': tracker.video_reader.is_connected(),
                    'reconnect_count': tracker.video_reader.get_reconnect_count(),
                }
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

    def reset_all_camera_counts(self) -> None:
        """Reset local counters for every active camera tracker."""
        with self._lock:
            for camera_id, tracker in self._trackers.items():
                try:
                    logger.info("Resetting local counters for camera '%s'", camera_id)
                    tracker.reset_counts()
                except Exception as exc:
                    logger.error(
                        "Failed to reset local counters for camera '%s': %s",
                        camera_id,
                        str(exc),
                        exc_info=True,
                    )
