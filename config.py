"""
Configuration module for edge-vision-counter.
Multi-camera setup with IP camera support via Android phones or other IP camera sources.
Contains all hardcoded values, camera configurations, and constants.
"""

import os
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).parent

# Multi-Camera Configuration (Support up to 10 cameras)
CAMERAS = [
    {
        "id": "Cam_Pintu_Utama",
        "url": "0",
        "orientation": "vertical",  # 'vertical' or 'horizontal'
        "line_ratio": 0.5,  # 0.0 to 1.0 (50% of width for vertical, 50% of height for horizontal)
        "in_direction": "left_to_right",  # Options: 'left_to_right', 'right_to_left', 'top_to_bottom', 'bottom_to_top'
        "frame_skip": 3,  # Process every Nth frame for inference (1=every frame, 3=every 3rd)
        "enabled": True,
    },
    # {
    #     "id": "Cam_Pintu_Belakang",
    #     "url": "6",
    #     "orientation": "vertical",  # 'vertical' or 'horizontal'
    #     "line_ratio": 0.5,  # 0.0 to 1.0 (50% of width for vertical, 50% of height for horizontal)
    #     "in_direction": "right_to_left",  # Options: 'left_to_right', 'right_to_left', 'top_to_bottom', 'bottom_to_top'
    #     "frame_skip": 3,  # Process every Nth frame for inference (1=every frame, 3=every 3rd)
    #     "enabled": False,
    # },
    # {
    #     "id": "Cam_Pintu_Samping",
    #     "url": "0",
    #     "orientation": "vertical",  # 'vertical' or 'horizontal'
    #     "line_ratio": 0.5,  # 0.0 to 1.0 (50% of width for vertical, 50% of height for horizontal)
    #     "in_direction": "right_to_left",  # Options: 'left_to_right', 'right_to_left', 'top_to_bottom', 'bottom_to_top'
    #     "frame_skip": 3,  # Process every Nth frame for inference (1=every frame, 3=every 3rd)
    #     "enabled": True,
    # },
    # Add more cameras here with the same structure:
    # {
    #     "id": "Cam_Ruang_Belakang",
    #     "url": "http://192.168.68.101:8080/video",
    #     "orientation": "vertical",
    #     "line_ratio": 0.4,
    #     "in_direction": "left_to_right",
    #     "frame_skip": 3,
    #     "enabled": True,
    # },
]

# Video Capture Configuration
CAMERA_FPS = 15
CAMERA_FRAME_WIDTH = 640
CAMERA_FRAME_HEIGHT = 480
CAMERA_INFERENCE_SIZE = 640  # Max resolution for inference (YOLOv8n optimal)
CAMERA_BUFFER_SIZE = 1  # OpenCV buffer size (1 = no buffering, prevents Wi-Fi lag)
CAMERA_CONNECT_TIMEOUT = 10  # Seconds to wait for camera connection
CAMERA_READ_TIMEOUT = 5.0  # Seconds to wait per frame read
RTSP_FFMPEG_CAPTURE_OPTIONS = (
    "rtsp_transport;tcp|"
    "fflags;nobuffer+discardcorrupt|"
    "flags;low_delay|"
    "max_delay;0|"
    "analyzeduration;0|"
    "probesize;32"
)

# Uniform capture profile for all cameras.
# Keep enabled so every camera uses the same requested profile.
CAMERA_ENFORCE_UNIFORM_CAPTURE = True
CAMERA_PREFERRED_FOURCC = "MJPG"  # Common webcam format to reduce backend mismatch
CAMERA_DISABLE_AUTO_ZOOM = True

# Frame normalization before tracking/inference.
# - "none": use raw camera frame as-is (native/default behavior)
# - "letterbox": resize while keeping aspect ratio, then pad (no crop, no distortion)
# - "stretch": force resize to target size (no crop, but may distort)
FRAME_NORMALIZATION_MODE = "letterbox"
FRAME_NORMALIZATION_WIDTH = 640
FRAME_NORMALIZATION_HEIGHT = 480

# Model Configuration
MODEL_NAME = "yolov8n"  # Nano model for edge devices
MODEL_TRACK_CLASS = 0  # 0 = person class in COCO dataset
CONFIDENCE_THRESHOLD = 0.35
IOU_THRESHOLD = 0.45

# Inference backend selection:
# - "auto"     : CUDA -> OpenVINO -> CPU fallback
# - "cuda"     : NVIDIA GPU only
# - "openvino" : Intel OpenVINO only
# - "cpu"      : CPU only
INFERENCE_BACKEND = os.environ.get("INFERENCE_BACKEND", "auto").lower()
CUDA_HALF = True
INFERENCE_BATCH_SIZE = 3
INFERENCE_BATCH_WAIT_SECONDS = 0.01
INFERENCE_RESULT_TIMEOUT_SECONDS = 4.0

# Tracking & State Machine
TRACK_PERSISTENCE = 60  # Keep lost IDs alive longer through lateral occlusion and Wi-Fi gaps.
BYTE_TRACK_TRACK_THRESH = 0.25
BYTE_TRACK_MATCH_THRESH = 0.80  # Tighten matching to reduce ID switches in dense crowds.
LINE_CROSSING_THRESHOLD = 10  # Minimum pixels to cross the line to count
DETECTION_MIN_CONSECUTIVE = 2  # Min frames to track before counting
LINE_BUFFER_ZONE_PX = 20  # Deadzone around counting line to avoid jitter double-count
LINE_CROSSING_COOLDOWN_FRAMES = 18  # Anti-loitering cooldown per ID after a valid crossing
LINE_MIN_MOTION_PX = 3.0  # Minimum directional motion to classify IN/OUT
TRACK_STATE_STALE_FRAMES = 180  # Cleanup window for inactive tracker IDs

# Threading & Performance Optimization
MAX_CONCURRENT_INFERENCE = 2  # Max parallel inference threads
VIDEO_CAPTURE_TIMEOUT = 5.0  # Seconds
FRAME_SKIP_DEFAULT = 3  # Default frame skip (5-10 FPS at 30 FPS source)
TARGET_FPS_PER_CAMERA = 8  # Target inference FPS per camera

# Web Server Configuration
FLASK_HOST = "0.0.0.0"
FLASK_PORT = 5000
FLASK_DEBUG = False
SSE_UPDATE_INTERVAL = 0.5  # Seconds
ADMIN_RESET_PASSWORD = os.environ.get("ADMIN_RESET_PASSWORD", "admin")

# OpenVINO Configuration (Intel Iris Xe Acceleration)
# Updated to OpenVINO 2024.6.0 - export fixed, GPU acceleration enabled
USE_OPENVINO = True
OPENVINO_DEVICE = "GPU"  # 'GPU' for iGPU (Intel Iris Xe), 'CPU' as fallback
OPENVINO_COMPILE_OPTIMIZATION = "LATENCY"  # 'LATENCY' or 'THROUGHPUT'
OPENVINO_AUTO_EXPORT = True
OPENVINO_XML_PATH = os.environ.get(
    "OPENVINO_XML_PATH",
    str(PROJECT_ROOT / f"{MODEL_NAME}_openvino_model" / f"{MODEL_NAME}.xml"),
)

# Visualization & Debug
# ENABLE_DEBUG_DISPLAY = False  # Disable cv2.imshow() for production to avoid UI blocking.
ENABLE_DEBUG_DISPLAY = True  # Show cv2.imshow() with bboxes, lines, IDs
DEBUG_DISPLAY_WINDOW_WIDTH = 960
DEBUG_DISPLAY_WINDOW_HEIGHT = 540

# Logging Configuration
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)
DATABASE_FILE = PROJECT_ROOT / "edge_vision_counter.db"

# Application Behavior
ENABLE_GRACEFUL_DEGRADATION = True  # Continue with available cameras if one fails
HEALTH_CHECK_INTERVAL = 30  # Seconds between camera health checks

# mDNS / Discovery
MDNS_ENABLED = True
MDNS_SERVICE_NAME = os.environ.get("COUNTER_MDNS_SERVICE_NAME", "counter-gww")
MDNS_HOSTNAME = os.environ.get("COUNTER_MDNS_HOSTNAME", "counter-gww")

# ============================================================================
# PHASE 1-3: LATERAL OCCLUSION HANDLING
# ============================================================================
# Enable all occlusion-robust detection and crossing validation strategies

# Phase 1: Edge-Aware Confidence Boost
# Boosts confidence for detections at frame edges (partial visibility)
EDGE_OCCLUSION_CONFIDENCE_FLOOR = 0.20      # Minimum confidence at edge (normally 0.35)
EDGE_OCCLUSION_CONFIDENCE_BOOST = 0.10      # Confidence boost amount for edge detections
EDGE_ZONE_MARGIN_PX = 80                    # Pixels from frame edge considered "edge zone"

# Phase 2: Motion-Aware Crossing Validation
# Uses ByteTrack history to detect crossings even with brief detection gaps
MOTION_EXTRAPOLATION_ENABLED = True         # Enable track momentum-based crossing detection
LOST_TRACK_REACQ_DISTANCE_PX = 150          # Max distance for track re-identification at re-acquisition

# Phase 3: Edge-Aware ROI Expansion
# Allow detections slightly outside frame bounds for partial people
VIRTUAL_EDGE_EXPANSION_PX = 100             # Allow detections this far outside frame edges

