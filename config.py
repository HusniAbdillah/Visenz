"""
Default runtime configuration for Edge Vision Counter.

This file ships with sample camera entries for a distant WiFi / RTSP setup.
Replace the URLs, credentials, and enable flags with your own deployment values.
"""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent

# Sample camera entries for distant WiFi / RTSP sources.
# Update the URL, credentials, and enabled flag before production use.
CAMERAS = [
    {
        "id": "Cam_Pintu_Keluar_Samping_Kiri",
        "url": "rtsp://ijfkiri:IJFCDAKiri26@10.2.241.184:554/stream2",
        "orientation": "vertical",  # Vertical line crossing on the x-axis.
        "line_ratio": 0.5,  # 0.0 to 1.0; 0.5 places the line in the middle.
        "in_direction": "left_to_right",  # IN/OUT direction used by the counter.
        "frame_skip": 3,  # Inference cadence: process every 3rd frame.
        "enabled": False,
    },
    {
        "id": "Cam_Pintu_Masuk_Depan",
        "url": "rtsp://ijfcda:IPBJobFair26@10.2.241.4:554/stream2",
        "orientation": "vertical",  # Vertical line crossing on the x-axis.
        "line_ratio": 0.5,  # 0.0 to 1.0; 0.5 places the line in the middle.
        "in_direction": "left_to_right",  # IN/OUT direction used by the counter.
        "frame_skip": 3,  # Inference cadence: process every 3rd frame.
        "enabled": False,
    },
    {
        "id": "Cam_Pintu_Keluar_Samping_Kanan",
        "url": "rtsp://ijfkanan:IJFCDA2026@10.2.240.198:554/stream2",
        "orientation": "vertical",  # Vertical line crossing on the x-axis.
        "line_ratio": 0.5,  # 0.0 to 1.0; 0.5 places the line in the middle.
        "in_direction": "left_to_right",  # IN/OUT direction used by the counter.
        "frame_skip": 3,  # Inference cadence: process every 3rd frame.
        "enabled": False,
    },
    # Example local HTTP stream; replace with your own source before enabling.
    {
        "id": "Cam_Ruang_Belakang",
        "url": "http://192.168.68.128:8080/video",
        "orientation": "vertical",
        "line_ratio": 0.4,
        "in_direction": "left_to_right",
        "frame_skip": 3,
        "enabled": False,
    },
]

# Video capture defaults for distant WiFi / RTSP sources
CAMERA_FPS = 10
CAMERA_FRAME_WIDTH = 640
CAMERA_FRAME_HEIGHT = 480
CAMERA_INFERENCE_SIZE = 640
CAMERA_BUFFER_SIZE = 1  # Keep only the most recent frame.
CAMERA_CONNECT_TIMEOUT = 25  # Seconds to wait for initial connection.
CAMERA_READ_TIMEOUT = 12.0  # Seconds to wait for frame reads.

RTSP_FFMPEG_CAPTURE_OPTIONS = (
    "rtsp_transport;tcp|"
    "fflags;nobuffer+discardcorrupt+flush_packets|"
    "flags;low_delay|"
    "max_delay;0|"
    "analyzeduration;0|"
    "probesize;32|"
    "rtsp_transport;tcp"
)

CAMERA_ENFORCE_UNIFORM_CAPTURE = True  # Normalize camera capture settings when possible.
CAMERA_PREFERRED_FOURCC = "MJPG"
CAMERA_DISABLE_AUTO_ZOOM = True

# Frame normalization
FRAME_NORMALIZATION_MODE = "letterbox"
FRAME_NORMALIZATION_WIDTH = 640
FRAME_NORMALIZATION_HEIGHT = 480

# Model configuration
MODEL_NAME = "yolov8n"
MODEL_TRACK_CLASS = 0  # COCO class index for person.
CONFIDENCE_THRESHOLD = 0.30
IOU_THRESHOLD = 0.40

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

# Tracking tuned for WiFi dropout resilience.
TRACK_PERSISTENCE = 120
BYTE_TRACK_TRACK_THRESH = 0.35
BYTE_TRACK_MATCH_THRESH = 0.75

# Line crossing detection
LINE_CROSSING_THRESHOLD = 15
DETECTION_MIN_CONSECUTIVE = 3
LINE_BUFFER_ZONE_PX = 25
LINE_CROSSING_COOLDOWN_FRAMES = 20
LINE_MIN_MOTION_PX = 5.0
TRACK_STATE_STALE_FRAMES = 300

# Threading & performance
MAX_CONCURRENT_INFERENCE = 1
VIDEO_CAPTURE_TIMEOUT = 12.0
FRAME_SKIP_DEFAULT = 3  # Baseline inference cadence for cameras without an explicit value.
TARGET_FPS_PER_CAMERA = 10

# Web server defaults
FLASK_HOST = "0.0.0.0"
FLASK_PORT = 5000
SSE_UPDATE_INTERVAL = 0.25  # Real-time updates at 250 ms intervals.
FLASK_DEBUG = False
ADMIN_RESET_PASSWORD = os.environ.get("ADMIN_RESET_PASSWORD", "admin")  # Default password for POST /api/reset.

# OpenVINO backend options
USE_OPENVINO = False
OPENVINO_DEVICE = "CPU"
OPENVINO_COMPILE_OPTIMIZATION = "LATENCY"
OPENVINO_AUTO_EXPORT = False  # Keep disabled because the shipped IR model already exists.
OPENVINO_XML_PATH = os.environ.get(
    "OPENVINO_XML_PATH",
    str(PROJECT_ROOT / f"{MODEL_NAME}_openvino_model" / f"{MODEL_NAME}.xml"),
)

# Visualization & debug
ENABLE_DEBUG_DISPLAY = False
DEBUG_DISPLAY_WINDOW_WIDTH = 960
DEBUG_DISPLAY_WINDOW_HEIGHT = 540

# Logging
LOG_LEVEL = "WARNING"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)
DATABASE_FILE = PROJECT_ROOT / "edge_vision_counter.db"

# Application behavior
ENABLE_GRACEFUL_DEGRADATION = True
HEALTH_CHECK_INTERVAL = 30

# mDNS service discovery
MDNS_ENABLED = True
MDNS_SERVICE_NAME = os.environ.get("COUNTER_MDNS_SERVICE_NAME", "counter-3wifi")
MDNS_HOSTNAME = os.environ.get("COUNTER_MDNS_HOSTNAME", "counter-3wifi")

# Occlusion handling
EDGE_OCCLUSION_CONFIDENCE_FLOOR = 0.15
EDGE_OCCLUSION_CONFIDENCE_BOOST = 0.15
EDGE_ZONE_MARGIN_PX = 100
MOTION_EXTRAPOLATION_ENABLED = True
LOST_TRACK_REACQ_DISTANCE_PX = 200
VIRTUAL_EDGE_EXPANSION_PX = 150

