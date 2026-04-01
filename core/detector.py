"""
Vision detection module using YOLOv8 with hardware-agnostic acceleration.
Implements automatic hardware detection with fallback chain: CUDA > MPS > OpenVINO > CPU.
Handles model loading and person detection optimized for multi-camera inference.
"""

import logging
import os
from pathlib import Path
from typing import Optional, Tuple, Any
import cv2
import numpy as np
import torch
from ultralytics import YOLO

import config

# Clear invalid CUDA_VISIBLE_DEVICES before torch/ultralytics initialization
_cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', '').strip()
if _cuda_visible and not all(c.isdigit() or c in ',-' for c in _cuda_visible):
    logger_early = logging.getLogger(__name__)
    logger_early.info(f"Clearing invalid CUDA_VISIBLE_DEVICES='{_cuda_visible}'")
    os.environ.pop('CUDA_VISIBLE_DEVICES', None)

logger = logging.getLogger(__name__)


class VisionModel:
    """
    Hardware-agnostic YOLOv8 model wrapper with automatic acceleration detection.
    Implements priority fallback: NVIDIA GPU (CUDA) > Apple Silicon (MPS) >
    Intel iGPU/CPU (OpenVINO) > Standard CPU.
    Detects and tracks persons with multi-camera optimization.
    Automatically resizes frames to optimal inference size.
    """

    def __init__(self) -> None:
        """Initialize the vision model with hardware acceleration detection."""
        self.model: Optional[YOLO] = None
        self.device: str = "cpu"
        self.openvino_runtime_device: str = "CPU"
        self.inference_size = config.CAMERA_INFERENCE_SIZE
        self._initialize_model()

    def _initialize_model(self) -> None:
        """
        Load YOLOv8 model with hardware detection hierarchy.
        Priority chain: CUDA > MPS > OpenVINO > CPU

        Raises:
            RuntimeError: If model cannot be loaded after all fallbacks.
        """
        model_path = f"{config.MODEL_NAME}.pt"

        try:
            logger.info("Starting hardware detection for YOLOv8n model initialization")

            # PRIORITY 1: NVIDIA GPU (CUDA)
            if self._try_cuda():
                return

            # PRIORITY 2: Apple Silicon (MPS)
            if self._try_mps():
                return

            # PRIORITY 3: Intel iGPU / CPU Optimization (OpenVINO)
            if config.USE_OPENVINO and self._try_openvino():
                return

            # PRIORITY 4: Standard CPU Fallback
            logger.info("All hardware accelerators unavailable. Falling back to standard CPU inference")
            self.model = YOLO(model_path)
            self.device = "cpu"
            logger.info("Model initialized successfully on device: cpu")

        except Exception as e:
            logger.error("Failed to initialize model after all fallback attempts: %s", str(e))
            raise RuntimeError(f"Model initialization failed: {str(e)}")

    def _try_cuda(self) -> bool:
        """
        Attempt to initialize model on NVIDIA GPU (CUDA).

        Returns:
            True if successful, False otherwise.
        """
        try:
            if not torch.cuda.is_available():
                logger.debug("CUDA not available on this system")
                return False

            logger.info("CUDA detected. Attempting YOLOv8n model initialization on GPU")
            model_path = f"{config.MODEL_NAME}.pt"
            self.model = YOLO(model_path)
            self.device = "cuda:0"
            logger.info("Model successfully loaded on NVIDIA GPU (device: cuda:0)")
            return True

        except Exception as e:
            logger.warning("CUDA initialization failed: %s", str(e))
            return False

    def _try_mps(self) -> bool:
        """
        Attempt to initialize model on Apple Silicon (Metal Performance Shaders).

        Returns:
            True if successful, False otherwise.
        """
        try:
            if not torch.backends.mps.is_available():
                logger.debug("MPS (Apple Metal) not available on this system")
                return False

            logger.info("Apple Metal Performance Shaders (MPS) detected. Attempting YOLOv8n initialization")
            model_path = f"{config.MODEL_NAME}.pt"
            self.model = YOLO(model_path)
            self.device = "mps"
            logger.info("Model successfully loaded on Apple Silicon (device: mps)")
            return True

        except Exception as e:
            logger.warning("MPS initialization failed: %s", str(e))
            return False

    def _try_openvino(self) -> bool:
        """
        Attempt to initialize model with OpenVINO backend for Intel iGPU/CPU optimization.
        Checks for existing OpenVINO export, exports if necessary.

        Returns:
            True if successful, False otherwise.
        """
        try:
            logger.info("Attempting OpenVINO backend initialization for Intel iGPU optimization")

            # Check for existing OpenVINO model
            openvino_dir = Path(config.PROJECT_ROOT) / f"{config.MODEL_NAME}_openvino_model"

            if not openvino_dir.exists():
                logger.info("OpenVINO model directory not found. Exporting YOLOv8n to OpenVINO format")
                self._export_to_openvino()
            else:
                logger.info("Found existing OpenVINO model at: %s", openvino_dir)

            # Load OpenVINO model
            logger.info("Loading OpenVINO model for inference")
            ov_model_path = str(openvino_dir)
            self.model = YOLO(ov_model_path)
            self.device = "openvino"
            self.openvino_runtime_device = str(getattr(config, "OPENVINO_DEVICE", "GPU")).upper()
            logger.info(
                "Model successfully loaded with OpenVINO backend (runtime device: %s)",
                self.openvino_runtime_device,
            )
            return True

        except ImportError as e:
            logger.warning("OpenVINO dependencies not installed: %s", str(e))
            return False
        except Exception as e:
            logger.warning("OpenVINO initialization failed: %s", str(e))
            return False

    def _export_to_openvino(self) -> None:
        """
        Export the YOLOv8n model to OpenVINO format.
        Export is always done on CPU/default path to avoid CUDA device parsing.

        Raises:
            RuntimeError: If export fails.
        """
        try:
            logger.info("Loading base YOLOv8n model for OpenVINO export")
            model_path = f"{config.MODEL_NAME}.pt"
            base_model = YOLO(model_path)

            logger.info("Exporting YOLOv8n to OpenVINO format using default export device")
            export_path = base_model.export(format="openvino")
            logger.info("OpenVINO export completed successfully: %s", export_path)

        except Exception as e:
            logger.error("OpenVINO export failed: %s", str(e))
            raise RuntimeError(f"OpenVINO export failed: {str(e)}")

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
            # For OpenVINO models, pass 'cpu' device (OpenVINO runtime device handled separately)
            track_device = "cpu" if self.device == "openvino" else self.device
            results = self.model.track(
                resized_frame,
                conf=conf,
                iou=iou,
                persist=True,
                verbose=False,
                tracker="bytetrack.yaml",
                device=track_device,
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
                        class_ids = _as_numpy(boxes.cls).astype(int, copy=False)
                    except Exception:
                        class_ids = np.zeros(len(boxes), dtype=int)

                    # Filter to person class only (COCO class 0)
                    person_mask = class_ids == config.MODEL_TRACK_CLASS

                    if np.any(person_mask):
                        # Get detections in resized space
                        xyxy_resized = _to_numpy(boxes.xyxy[person_mask])

                        # Scale back to original frame coordinates
                        xyxy_original = xyxy_resized * np.array([scale_x, scale_y, scale_x, scale_y], dtype=float)

                        detection_dict['boxes'] = xyxy_original
                        detection_dict['confs'] = _to_numpy(boxes.conf[person_mask])

                        if boxes.id is not None:
                            detection_dict['ids'] = _to_numpy(boxes.id[person_mask]).astype(int, copy=False)
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


def _to_numpy(x: Any) -> np.ndarray:
    """Convert torch tensor / ultralytics tensor-like / ndarray to ndarray."""
    if x is None:
        return np.array([])
    if hasattr(x, "cpu"):   # torch-like
        x = x.cpu()
    if hasattr(x, "numpy"): # torch tensor on CPU
        return x.numpy()
    return np.asarray(x)


def _as_numpy(x: Any) -> np.ndarray:
    if x is None:
        return np.array([])
    if hasattr(x, "cpu"):   # torch tensor
        x = x.cpu()
    if hasattr(x, "numpy"): # torch tensor on CPU
        return x.numpy()
    return np.asarray(x)
