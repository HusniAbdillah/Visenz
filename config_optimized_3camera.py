"""
Configuration module for edge-vision-counter - OPTIMIZED untuk 3-Kamera Setup.
Multi-camera setup dengan TP-Link WiFi (x2) + Logitech USB (x1).

PERUBAHAN DARI DEFAULT:
- frame_skip: 3 → 2 (RTSP), 1 (USB) untuk latency lebih rendah
- CONFIDENCE_THRESHOLD: 0.35 → 0.30 untuk occlusion handling lebih baik
- TRACK_PERSISTENCE: 60 → 90 untuk WiFi dropout resilience
- SSE_UPDATE_INTERVAL: 0.5 → 0.25 untuk UI lebih responsive
- INFERENCE_BACKEND: auto → cpu untuk Windows stability
"""

import os
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).parent

# ============================================================================
# MULTI-CAMERA CONFIGURATION - 3 Kamera Setup
# ============================================================================
# Kamera 1 & 2: TP-Link IP Cameras (RTSP over WiFi)
# Kamera 3: Logitech USB Camera (direct device)

CAMERAS = [
    {
        "id": "Cam_TPLink_Entrance_A",
        "url": "rtsp://admin:password@192.168.1.100:554/stream1",  # TP-Link Camera #1 - ganti IP
        "orientation": "vertical",
        "line_ratio": 0.50,  # Garis di tengah (50% dari lebar)
        "in_direction": "left_to_right",  # Orang masuk dari kiri ke kanan
        "frame_skip": 2,  # CHANGED: 3→2 untuk latency lebih rendah di WiFi
        "enabled": True,
    },
    {
        "id": "Cam_TPLink_Entrance_B",
        "url": "rtsp://admin:password@192.168.1.101:554/stream1",  # TP-Link Camera #2 - ganti IP
        "orientation": "vertical",
        "line_ratio": 0.45,  # Garis sedikit ke kiri untuk view yang lebih baik
        "in_direction": "left_to_right",
        "frame_skip": 2,  # CHANGED: 3→2
        "enabled": True,
    },
    {
        "id": "Cam_Logitech_USB_Direct",
        "url": "0",  # Local USB camera (index 0) - atau gunakan path jika ada multiple USB
        "orientation": "horizontal",  # Logitech biasanya horizontal
        "line_ratio": 0.55,  # Garis 55% dari tinggi
        "in_direction": "top_to_bottom",  # Orang masuk dari atas ke bawah
        "frame_skip": 1,  # CHANGED: 3→1 karena USB sudah low-latency, bisa process every frame!
        "enabled": True,
    },
]

# ============================================================================
# VIDEO CAPTURE CONFIGURATION
# ============================================================================

# Target frame rate dari kamera
CAMERA_FPS = 15  # Keep at 15 (lebih stabil untuk WiFi dari pada 30)

# Resolusi capture yang diminta
CAMERA_FRAME_WIDTH = 640
CAMERA_FRAME_HEIGHT = 480

# Ukuran optimal untuk inference (YOLOv8n optimal di 640px)
CAMERA_INFERENCE_SIZE = 640

# CRITICAL: Buffer size untuk mencegah lag accumulation
# JANGAN UBAH INI! Ini yang mencegah video "patah-patah"
CAMERA_BUFFER_SIZE = 1  # 1 = hanya simpan frame terakhir, discard frame lama

# RTSP connection timeouts
CAMERA_CONNECT_TIMEOUT = 15  # CHANGED: 10→15 (WiFi lebih variable timeout-nya)
CAMERA_READ_TIMEOUT = 8.0    # CHANGED: 5.0→8.0 (give WiFi more time per frame)

# FFmpeg RTSP options untuk low-latency streaming
RTSP_FFMPEG_CAPTURE_OPTIONS = (
    "rtsp_transport;tcp|"
    "fflags;nobuffer+discardcorrupt+flush_packets|"
    "flags;low_delay|"
    "max_delay;0|"
    "analyzeduration;0|"
    "probesize;32|"
    "rtsp_transport;tcp"
)

# Uniform capture profile untuk semua kamera
CAMERA_ENFORCE_UNIFORM_CAPTURE = True
CAMERA_PREFERRED_FOURCC = "MJPG"  # Motion JPEG - lebih compatible dari H.264
CAMERA_DISABLE_AUTO_ZOOM = True

# ============================================================================
# FRAME NORMALIZATION (Letterbox = no distortion, preserve aspect ratio)
# ============================================================================
# Mode: "none" = raw, "letterbox" = pad dengan black borders, "stretch" = distort
FRAME_NORMALIZATION_MODE = "letterbox"  # Keep - avoid distortion
FRAME_NORMALIZATION_WIDTH = 640
FRAME_NORMALIZATION_HEIGHT = 480

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

MODEL_NAME = "yolov8n"  # Nano model - balance speed (30-50ms) & accuracy
MODEL_TRACK_CLASS = 0   # Person class dalam COCO dataset

# CHANGED: Lebih lenient untuk menangkap orang yang sebagian tersembunyi
CONFIDENCE_THRESHOLD = 0.30  # CHANGED: 0.35→0.30 (catch partial visibility)
IOU_THRESHOLD = 0.40         # CHANGED: 0.45→0.40 (lebih aggressive NMS)

# ============================================================================
# INFERENCE BACKEND SELECTION
# ============================================================================
# "auto" = CUDA → OpenVINO → CPU fallback
# "cuda" = NVIDIA GPU only
# "openvino" = Intel OpenVINO only
# "cpu" = CPU only

# CHANGED: auto→cpu karena Windows CPU lebih stabil (OpenVINO bisa flaky)
INFERENCE_BACKEND = "cpu"
# Jika ingin try OpenVINO: INFERENCE_BACKEND = os.environ.get("INFERENCE_BACKEND", "auto")

CUDA_HALF = False  # CHANGED: True→False (CPU tidak perlu fp16)

# Inference batch settings
INFERENCE_BATCH_SIZE = 1    # CHANGED: 3→1 (CPU tidak bisa batch efficient)
INFERENCE_BATCH_WAIT_SECONDS = 0.001  # CHANGED: 0.01→0.001 (faster response)
INFERENCE_RESULT_TIMEOUT_SECONDS = 2.0  # CHANGED: 4.0→2.0 (stricter timeout)

# ============================================================================
# TRACKING & STATE MACHINE (Occlusion Handling - AGGRESSIVE)
# ============================================================================

# CHANGED: Extended persistence untuk WiFi dropout recovery
TRACK_PERSISTENCE = 90  # CHANGED: 60→90 (survive 3 second WiFi dropout)

# ByteTrack configuration
BYTE_TRACK_TRACK_THRESH = 0.35  # CHANGED: 0.25→0.35 (stricter new tracks)
BYTE_TRACK_MATCH_THRESH = 0.75  # CHANGED: 0.80→0.75 (lebih mudah re-match)

# Line crossing detection parameters
LINE_CROSSING_THRESHOLD = 15     # CHANGED: 10→15 (stricter crossing validation)
DETECTION_MIN_CONSECUTIVE = 3    # CHANGED: 2→3 (require 3 frame consistency)
LINE_BUFFER_ZONE_PX = 25         # CHANGED: 20→25 (wider deadzone untuk stability)
LINE_CROSSING_COOLDOWN_FRAMES = 20  # CHANGED: 18→20 (longer cooldown for jitter)
LINE_MIN_MOTION_PX = 5.0         # CHANGED: 3.0→5.0 (require more motion)

# Track cleanup (remove very old ghost tracks)
TRACK_STATE_STALE_FRAMES = 300   # CHANGED: 180→300 (longer history retention)

# ============================================================================
# THREADING & PERFORMANCE OPTIMIZATION
# ============================================================================

MAX_CONCURRENT_INFERENCE = 1  # CHANGED: 2→1 (CPU better with serialization)
VIDEO_CAPTURE_TIMEOUT = 8.0   # CHANGED: 5.0→8.0 (WiFi needs longer timeout)

# Frame skipping strategy (balance between latency & accuracy)
FRAME_SKIP_DEFAULT = 2  # CHANGED: 3→2 (lower initial latency)
TARGET_FPS_PER_CAMERA = 10  # Target: ~10 FPS inference per camera

# ============================================================================
# WEB SERVER CONFIGURATION
# ============================================================================

FLASK_HOST = "0.0.0.0"  # Listen on all interfaces
FLASK_PORT = 5000

# SSE (Server-Sent Events) update frequency
# CHANGED: 0.5→0.25 (2 Hz → 4 Hz) untuk UI lebih responsive
SSE_UPDATE_INTERVAL = 0.25  # Dalam detik (setiap 0.25s = 4 updates/second)

FLASK_DEBUG = False  # Keep False untuk production
ADMIN_RESET_PASSWORD = os.environ.get("ADMIN_RESET_PASSWORD", "admin")

# ============================================================================
# OPENVINO CONFIGURATION (Intel Iris Xe Acceleration)
# ============================================================================
# CHANGED: True→False karena Windows CPU lebih reliable

USE_OPENVINO = False  # Set to True jika ingin GPU acceleration (optional)
OPENVINO_DEVICE = "CPU"  # Fallback device
OPENVINO_COMPILE_OPTIMIZATION = "LATENCY"  # LATENCY atau THROUGHPUT
OPENVINO_AUTO_EXPORT = False

OPENVINO_XML_PATH = os.environ.get(
    "OPENVINO_XML_PATH",
    str(PROJECT_ROOT / f"{MODEL_NAME}_openvino_model" / f"{MODEL_NAME}.xml"),
)

# ============================================================================
# VISUALIZATION & DEBUG
# ============================================================================

# CHANGED: True→False untuk production (cv2.imshow blocks)
ENABLE_DEBUG_DISPLAY = False  # Set to True hanya untuk local testing

DEBUG_DISPLAY_WINDOW_WIDTH = 960
DEBUG_DISPLAY_WINDOW_HEIGHT = 540

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

# CHANGED: INFO→WARNING (reduce disk I/O)
LOG_LEVEL = "WARNING"  # DEBUG, INFO, WARNING, ERROR

LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)
DATABASE_FILE = PROJECT_ROOT / "edge_vision_counter.db"

# ============================================================================
# APPLICATION BEHAVIOR
# ============================================================================

ENABLE_GRACEFUL_DEGRADATION = True  # Continue jika 1 kamera down
HEALTH_CHECK_INTERVAL = 30  # Cek health setiap 30 detik

# ============================================================================
# mDNS / SERVICE DISCOVERY (Optional)
# ============================================================================

MDNS_ENABLED = True
MDNS_SERVICE_NAME = os.environ.get("COUNTER_MDNS_SERVICE_NAME", "counter-gww")
MDNS_HOSTNAME = os.environ.get("COUNTER_MDNS_HOSTNAME", "counter-gww")

# ============================================================================
# PHASE 1-3: LATERAL OCCLUSION HANDLING (AGGRESSIVE MODE)
# ============================================================================

# PHASE 1: Edge-Aware Confidence Boost
# Tangani case dimana orang partially visible di frame edges
EDGE_OCCLUSION_CONFIDENCE_FLOOR = 0.15      # CHANGED: 0.20→0.15 (catch more edges)
EDGE_OCCLUSION_CONFIDENCE_BOOST = 0.15      # CHANGED: 0.10→0.15 (boost more!)
EDGE_ZONE_MARGIN_PX = 100                   # CHANGED: 80→100 (wider edge zone)

# PHASE 2: Motion-Aware Crossing Validation
# Gunakan ByteTrack history untuk detect crossing bahkan saat detection gap
MOTION_EXTRAPOLATION_ENABLED = True
LOST_TRACK_REACQ_DISTANCE_PX = 200          # CHANGED: 150→200 (wider tolerance)

# PHASE 3: Edge-Aware ROI Expansion
# Allow bounding boxes extend outside frame bounds
VIRTUAL_EDGE_EXPANSION_PX = 150             # CHANGED: 100→150 (allow further out)

# ============================================================================
# CUSTOM IMPROVEMENTS (TODO - implementasi di phase 2-3)
# ============================================================================

# Uncomment untuk enable adaptive frame skipping (beta)
# ENABLE_ADAPTIVE_FRAME_SKIP = True
# ADAPTIVE_TARGET_CPU_PERCENT = 70

# Uncomment untuk enable momentum extrapolation (untuk occlusion heavy scenes)
# ENABLE_MOMENTUM_EXTRAPOLATION = True
# MOMENTUM_EXTRAPOLATION_FRAMES = 30

# ============================================================================
# SETUP INSTRUCTIONS
# ============================================================================
"""
KONFIGURASI SEBELUM PRODUCTION:

1. Update IP addresses:
   - Line 24: "rtsp://admin:password@192.168.1.100:554/stream1"
   - Line 33: "rtsp://admin:password@192.168.1.101:554/stream1"
   
   Ganti dengan IP actual TP-Link cameras Anda
   Ganti "admin" dan "password" dengan credentials Anda
   Ganti "/stream1" dengan path stream yang benar (bisa "/stream2" etc)

2. Verify Logitech USB:
   - Jalankan: python -c "import cv2; cap = cv2.VideoCapture(0); print(cap.isOpened())"
   - Jika False, coba index 1, 2, etc sampai find USB camera

3. Test per-camera:
   - Uncomment satu camera, test latency & accuracy
   - Lalu uncomment yang lain one-by-one

4. Monitor initial run:
   - Check logs/edge_vision_counter.log untuk errors
   - Validate latency di browser dashboard
   - Expected: 80-120ms total latency
   
5. Adjust jika perlu:
   - Jika CPU >80%: increase FRAME_SKIP_DEFAULT to 3
   - Jika video choppy: check CAMERA_BUFFER_SIZE = 1 applied
   - Jika occlusion bad: decrease CONFIDENCE_THRESHOLD to 0.25
"""
