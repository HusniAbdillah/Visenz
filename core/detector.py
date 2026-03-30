"""
Vision detection module using YOLOv8 with OpenVINO acceleration.
Handles model loading with robust hardware fallback and person detection.
Optimized for multi-camera inference with frame resizing.
"""

import logging
from typing import Optional, Tuple
import cv2
import numpy as np
from ultralytics import YOLO

import config

logger = logging.getLogger(__name__)


class VisionModel:
    """
    Wrapper for YOLOv8 model with OpenVINO acceleration and robust CPU fallback.
    Detects and tracks persons with multi-camera optimization.
    Automatically resizes frames to optimal inference size.
    """

    def __init__(self) -> None:
        """Initialize the vision model with hardware acceleration if available."""
        self.model: Optional[YOLO] = None
        self.device: str = "cpu"
        self.inference_size = config.CAMERA_INFERENCE_SIZE
        self._load_model()

    def _load_model(self) -> None:
        """
        Load YOLOv8 model with OpenVINO acceleration attempt.
        Falls back to CPU if OpenVINO export fails.
        """
        try:
            logger.info("Loading YOLOv8 model: %s", config.MODEL_NAME)
            self.model = YOLO(f"{config.MODEL_NAME}.pt")

            if config.USE_OPENVINO:
                self._try_openvino_export()
            else:
                logger.info("OpenVINO disabled. Using CPU inference.")
                self.device = "cpu"

        except Exception as e:
            logger.error("Failed to load model: %s", str(e))
            raise

    def _try_openvino_export(self) -> None:
        """
        Attempt to export model to OpenVINO format for Intel Iris Xe acceleration.
        Falls back to CPU on failure.
        """
        try:
            logger.info("Attempting OpenVINO export for Intel Iris Xe acceleration...")

            export_kwargs = {
                "format": "openvino",
                "device": config.OPENVINO_DEVICE,
            }

            ov_model_path = self.model.export(**export_kwargs)
            logger.info("OpenVINO export successful: %s", ov_model_path)

            self.model = YOLO(str(ov_model_path))
            self.device = f"openvino_{config.OPENVINO_DEVICE}"
            logger.info("Model loaded with device: %s", self.device)

        except ImportError as e:
            logger.warning("OpenVINO dependencies not installed: %s", str(e))
            self.device = "cpu"
            logger.info("Falling back to CPU inference")
        except Exception as e:
            logger.warning("OpenVINO export failed, falling back to CPU: %s", str(e))
            self.device = "cpu"
            logger.info("Model using CPU inference")

    def _resize_frame_for_inference(self, frame: np.ndarray) -> Tuple[np.ndarray, Tuple[float, float]]:
        """
        Resize frame to optimal inference size and return scale factors.

        Args:
            frame: Input frame (BGR)

        Returns:
            Tuple of (resized_frame, (scale_x, scale_y))
        """
        h, w = frame.shape[:2]

        if w > self.inference_size or h > self.inference_size:
            scale = min(self.inference_size / w, self.inference_size / h)
            new_w, new_h = int(w * scale), int(h * scale)
            resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            return resized, (1.0 / scale, 1.0 / scale)
        else:
            return frame, (1.0, 1.0)

    def predict_with_tracking(
        self,
        frame: np.ndarray,
        conf: float = config.CONFIDENCE_THRESHOLD,
        iou: float = config.IOU_THRESHOLD,
    ) -> dict:
        """
        Run inference with tracking on a frame and return person detections only.
        FIXED: Enhanced tracker config for ID persistence through Wi-Fi lag and occlusions.

        Args:
            frame: Input image frame (BGR format)
            conf: Confidence threshold (0.0-1.0)
            iou: IOU threshold for NMS (0.0-1.0)

        Returns:
            Dictionary containing detection results for persons only.
            Format: {
                'boxes': [[x1,y1,x2,y2], ...],  # Bounding boxes in original frame coords
                'confs': [0.95, ...],            # Confidence scores
                'ids': [1, 2, ...],              # ByteTrack IDs (stable across frames)
                'centroids': [[x, y], ...],     # Centroid coordinates
                'raw_frame_shape': (h, w),      # Original frame dimensions
            }

        Raises:
            RuntimeError: If model is not initialized or inference fails.
        """
        if self.model is None:
            raise RuntimeError("Model not initialized")

        try:
            frame_h, frame_w = frame.shape[:2]
            resized_frame, (scale_x, scale_y) = self._resize_frame_for_inference(frame)

            # FIXED: Enhanced tracker configuration via tracker config file
            # Note: max_age, track_buffer, min_hits must be configured via tracker YAML file,
            # not via kwargs to model.track()
            results = self.model.track(
                resized_frame,
                conf=conf,
                iou=iou,
                persist=True,
                verbose=False,
                tracker="bytetrack.yaml",
            )

            detection_dict = {
                'boxes': [],
                'confs': [],
                'ids': [],
                'centroids': [],
                'raw_frame_shape': (frame_h, frame_w),
            }

            if results and len(results) > 0:
                result = results[0]

                if result.boxes is not None and len(result.boxes) > 0:
                    boxes = result.boxes

                    try:
                        class_ids = boxes.cls.cpu().numpy().astype(int)
                    except:
                        class_ids = np.zeros(len(boxes), dtype=int)

                    # Filter to person class only (COCO class 0)
                    person_mask = class_ids == config.MODEL_TRACK_CLASS

                    if np.any(person_mask):
                        # Get detections in resized space
                        xyxy_resized = boxes.xyxy[person_mask].cpu().numpy()

                        # Scale back to original frame coordinates
                        xyxy_original = xyxy_resized * np.array([scale_x, scale_y, scale_x, scale_y])

                        detection_dict['boxes'] = xyxy_original
                        detection_dict['confs'] = boxes.conf[person_mask].cpu().numpy()

                        if boxes.id is not None:
                            detection_dict['ids'] = boxes.id[person_mask].cpu().numpy().astype(int)
                        else:
                            detection_dict['ids'] = np.arange(len(xyxy_original), dtype=int)

                        # Calculate centroids for line crossing logic
                        centroids = (xyxy_original[:, :2] + xyxy_original[:, 2:]) / 2
                        detection_dict['centroids'] = centroids

            return detection_dict

        except Exception as e:
            logger.error("Inference failed: %s", str(e))
            raise RuntimeError(f"Inference failed: {str(e)}")

    def get_device(self) -> str:
        """Return the device being used for inference."""
        return self.device
