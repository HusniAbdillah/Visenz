"""
Vision detection module using YOLOv8 with hardware-agnostic acceleration.
Implements automatic hardware detection with fallback chain: CUDA > MPS > OpenVINO > CPU.
Handles model loading and person detection optimized for multi-camera inference.
"""

import logging
import os
import queue
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
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


@dataclass
class _InferenceRequest:
    frame: np.ndarray
    conf: float
    iou: float
    event: threading.Event = field(default_factory=threading.Event)
    result: Optional[Dict[str, Any]] = None
    error: Optional[BaseException] = None


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
        self._predict_device: Optional[str] = "cpu"
        self.inference_size = config.CAMERA_INFERENCE_SIZE
        self._batch_queue: "queue.Queue[_InferenceRequest]" = queue.Queue()
        self._batch_worker: Optional[threading.Thread] = None
        self._running: bool = False
        self._batch_size: int = max(1, int(getattr(config, "INFERENCE_BATCH_SIZE", 3)))
        self._batch_wait_seconds: float = float(getattr(config, "INFERENCE_BATCH_WAIT_SECONDS", 0.02))
        self._sync_inference_lock: threading.Lock = threading.Lock()
        self._initialize_model()

    def _initialize_model(self) -> None:
        """
        Load YOLOv8 model using configured backend preference.

        Priority for "auto" backend:
        1) CUDA
        2) OpenVINO (Intel)
        3) CPU fallback

        Raises:
            RuntimeError: If all selected backend options fail.
        """
        model_path = f"{config.MODEL_NAME}.pt"
        backend = str(getattr(config, "INFERENCE_BACKEND", "auto")).strip().lower()

        logger.info("Initializing YOLOv8 backend='%s' model='%s'", backend, model_path)

        errors = []
        backend_order = [backend]
        if backend == "auto":
            backend_order = ["cuda", "openvino", "cpu"]

        for candidate in backend_order:
            try:
                if candidate == "cuda":
                    self._initialize_cuda(model_path)
                    return
                if candidate == "openvino":
                    self._initialize_openvino(model_path)
                    return
                if candidate == "cpu":
                    self._initialize_cpu(model_path)
                    return
                raise RuntimeError(f"Unsupported INFERENCE_BACKEND='{candidate}'")
            except Exception as e:
                errors.append(f"{candidate}: {e}")
                logger.warning("Backend '%s' initialization failed: %s", candidate, str(e))

        raise RuntimeError("Model initialization failed for all candidates: " + " | ".join(errors))

    def _initialize_cuda(self, model_path: str) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("torch.cuda.is_available() returned False")

        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = True

        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

        self.model = YOLO(model_path, task="detect")
        self.device = "cuda:0"
        self._predict_device = "cuda:0"
        self._running = True
        self._batch_worker = threading.Thread(
            target=self._batch_inference_loop,
            name="VisionModelBatchWorker",
            daemon=True,
        )
        self._batch_worker.start()
        logger.info("Model initialized on CUDA (device: cuda:0)")

    def _initialize_openvino(self, model_path: str) -> None:
        model_stem = config.MODEL_NAME
        configured_xml = getattr(config, "OPENVINO_XML_PATH", None)
        if configured_xml:
            ov_xml_path = Path(configured_xml)
        else:
            ov_xml_path = config.PROJECT_ROOT / f"{model_stem}_openvino_model" / f"{model_stem}.xml"

        if not ov_xml_path.exists():
            auto_export = bool(getattr(config, "OPENVINO_AUTO_EXPORT", True))
            if not auto_export:
                raise RuntimeError(f"OpenVINO model not found: {ov_xml_path}")

            logger.info("OpenVINO IR not found. Exporting from PT model on CPU...")
            base_model = YOLO(model_path, task="detect")
            self._export_to_openvino(base_model, model_stem)

        if not ov_xml_path.exists():
            raise RuntimeError(f"OpenVINO export did not produce XML: {ov_xml_path}")

        # Ultralytics OpenVINO backend expects the exported directory path
        # (e.g. yolov8n_openvino_model), not only the .xml file path.
        ov_model_dir = ov_xml_path.parent if ov_xml_path.suffix.lower() == ".xml" else ov_xml_path
        if not ov_model_dir.exists() or not ov_model_dir.is_dir():
            raise RuntimeError(f"OpenVINO model directory not found: {ov_model_dir}")

        self.model = YOLO(str(ov_model_dir), task="detect")
        target = str(getattr(config, "OPENVINO_DEVICE", "GPU")).strip().upper()
        self.device = f"openvino:{target.lower()}"
        # Keep runtime predictable across Ultralytics/OpenVINO versions.
        self._predict_device = "cpu"
        self._running = False
        logger.info("Model initialized on OpenVINO (target=%s, dir=%s)", target, ov_model_dir)

    def _initialize_cpu(self, model_path: str) -> None:
        self.model = YOLO(model_path, task="detect")
        self.device = "cpu"
        self._predict_device = "cpu"
        self._running = False
        logger.info("Model initialized on CPU")

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
            model = self.model
            if model is None:
                raise RuntimeError("Model not initialized")

            if self._running and self._batch_worker is not None:
                request = _InferenceRequest(frame=frame, conf=conf, iou=iou)
                self._batch_queue.put(request)
                timeout_seconds = float(getattr(config, "INFERENCE_RESULT_TIMEOUT_SECONDS", 4.0))
                if not request.event.wait(timeout=timeout_seconds):
                    raise RuntimeError("Timed out waiting for batched GPU inference")
                if request.error is not None:
                    raise RuntimeError(f"Inference failed: {request.error}") from request.error
                if request.result is None:
                    raise RuntimeError("Inference returned no result")
                return request.result

            with self._sync_inference_lock:
                return self._run_single_inference(model, frame, conf=conf, iou=iou)

        except Exception as e:
            logger.error("Inference failed: %s", str(e))
            raise RuntimeError(f"Inference failed: {str(e)}")

    def _batch_inference_loop(self) -> None:
        """Collect pending requests and execute them as a small batch on CUDA."""
        while self._running:
            try:
                first = self._batch_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            batch = [first]
            deadline = time.monotonic() + self._batch_wait_seconds
            while len(batch) < self._batch_size:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    batch.append(self._batch_queue.get(timeout=remaining))
                except queue.Empty:
                    break

            try:
                frames = [request.frame for request in batch]
                conf = batch[0].conf
                iou = batch[0].iou
                model = self.model
                if model is None:
                    raise RuntimeError("Model not initialized")

                results = model.predict(
                    frames,
                    conf=conf,
                    iou=iou,
                    imgsz=self.inference_size,
                    device=self._predict_device,
                    half=bool(getattr(config, "CUDA_HALF", True) and self.device.startswith("cuda")),
                    verbose=False,
                )

                if not isinstance(results, list):
                    results = [results]

                for request, result, frame in zip(batch, results, frames):
                    request.result = self._result_to_detection_dict(result, frame)
                    request.event.set()

            except Exception as exc:
                logger.error("Batched inference failed: %s", str(exc))
                for request in batch:
                    request.error = exc
                    request.event.set()

    def _run_single_inference(self, model: YOLO, frame: np.ndarray, conf: float, iou: float) -> Dict[str, Any]:
        """Fallback synchronous inference path used when the batch worker is unavailable."""
        results = model.predict(
            frame,
            conf=conf,
            iou=iou,
            imgsz=self.inference_size,
            device=self._predict_device,
            half=bool(getattr(config, "CUDA_HALF", True) and self.device.startswith("cuda")),
            verbose=False,
        )

        result = results[0] if isinstance(results, list) and results else results
        return self._result_to_detection_dict(result, frame)

    def _result_to_detection_dict(self, result: Any, frame: np.ndarray) -> Dict[str, Any]:
        frame_h, frame_w = frame.shape[:2]
        detection_dict: Dict[str, Any] = {
            "boxes": [],
            "confs": [],
            "ids": [],
            "centroids": [],
            "raw_frame_shape": (frame_h, frame_w),
        }

        if result is None or result.boxes is None or len(result.boxes) == 0:
            return detection_dict

        boxes = result.boxes

        try:
            class_ids = _as_numpy(boxes.cls).astype(int, copy=False)
        except Exception:
            class_ids = np.zeros(len(boxes), dtype=int)

        person_mask = class_ids == config.MODEL_TRACK_CLASS
        if not np.any(person_mask):
            return detection_dict

        xyxy = _to_numpy(boxes.xyxy[person_mask])
        detection_dict["boxes"] = xyxy
        detection_dict["confs"] = _to_numpy(boxes.conf[person_mask])

        if getattr(boxes, "id", None) is not None:
            detection_dict["ids"] = _to_numpy(boxes.id[person_mask]).astype(int, copy=False)

        detection_dict["centroids"] = (xyxy[:, :2] + xyxy[:, 2:]) / 2
        return detection_dict

    def get_device(self) -> str:
        """Return the device being used for inference."""
        return self.device

    def get_gpu_stats(self) -> Dict[str, Any]:
        """Return current GPU memory and utilization metrics if CUDA is available."""
        stats: Dict[str, Any] = {
            "device": self.device,
            "cuda_available": bool(torch.cuda.is_available()),
            "batch_queue_depth": self._batch_queue.qsize(),
            "batch_size": self._batch_size,
            "utilization_percent": None,
            "memory_used_mb": None,
            "memory_total_mb": None,
            "memory_allocated_mb": None,
            "memory_reserved_mb": None,
            "device_name": None,
        }

        if not torch.cuda.is_available():
            return stats

        try:
            stats["device_name"] = torch.cuda.get_device_name(0)
            stats["memory_allocated_mb"] = round(torch.cuda.memory_allocated(0) / (1024 ** 2), 2)
            stats["memory_reserved_mb"] = round(torch.cuda.memory_reserved(0) / (1024 ** 2), 2)
            props = torch.cuda.get_device_properties(0)
            stats["memory_total_mb"] = round(props.total_memory / (1024 ** 2), 2)
        except Exception:
            pass

        try:
            command = [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ]
            completed = subprocess.run(command, capture_output=True, text=True, timeout=2)
            if completed.returncode == 0 and completed.stdout.strip():
                first_line = completed.stdout.strip().splitlines()[0]
                utilization_str, memory_used_str, memory_total_str = [part.strip() for part in first_line.split(",")]
                stats["utilization_percent"] = float(utilization_str)
                stats["memory_used_mb"] = float(memory_used_str)
                stats["memory_total_mb"] = float(memory_total_str)
        except Exception:
            pass

        if stats["memory_used_mb"] is None and stats["memory_allocated_mb"] is not None:
            stats["memory_used_mb"] = stats["memory_allocated_mb"]

        return stats

    def _export_to_openvino(self, model: YOLO, model_stem: str) -> None:
        """
        Export PT model to OpenVINO IR while forcing CPU export device.

        For some CUDA environments, Ultralytics export raises
        `Invalid CUDA device` unless export is pinned to CPU.
        """
        export_kwargs = {
            "format": "openvino",
            "device": "cpu",
        }
        logger.info("Exporting model '%s' to OpenVINO with args=%s", model_stem, export_kwargs)
        model.export(**export_kwargs)


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
