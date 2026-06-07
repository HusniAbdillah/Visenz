"""
Configuration module untuk 3 WiFi Cameras jarak jauh (distant setup).

Optimized untuk:
- 3 WiFi IP cameras (tidak ada USB)
- Distance jauh (signal lebih lemah, latency tinggi, dropout sering)
- Need aggressive timeout & persistence handling
"""

import os
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).parent

# ============================================================================
# MULTI-CAMERA CONFIGURATION - 3 WiFi Cameras Jarak Jauh
# ============================================================================

CAMERAS = [
    {
        "id": "Cam_WiFi_1_Distant",
        "url": "rtsp://admin:password@192.168.1.100:554/stream1",  # Ganti IP & password!
        "orientation": "vertical",
        "line_ratio": 0.50,
        "in_direction": "left_to_right",
        "frame_skip": 3,  # Jarak jauh → frame_skip lebih tinggi
        "enabled": True,
    },
    {
        "id": "Cam_WiFi_2_Distant",
        "url": "rtsp://admin:password@192.168.1.101:554/stream1",  # Ganti IP & password!
        "orientation": "vertical",
        "line_ratio": 0.45,
        "in_direction": "left_to_right",
        "frame_skip": 3,
        "enabled": True,
    },
    {
        "id": "Cam_WiFi_3_Distant",
        "url": "rtsp://admin:password@192.168.1.102:554/stream1",  # Ganti IP & password!
        "orientation": "vertical",
        "line_ratio": 0.55,
        "in_direction": "left_to_right",
        "frame_skip": 3,
        "enabled": True,
    },
]

# ============================================================================
# VIDEO CAPTURE CONFIGURATION - WiFi Distant Optimized
# ============================================================================

CAMERA_FPS = 10  # Reduced dari 15 untuk jarak jauh (lebih stabil)

CAMERA_FRAME_WIDTH = 640
CAMERA_FRAME_HEIGHT = 480

CAMERA_INFERENCE_SIZE = 640

# CRITICAL: Buffer = 1 untuk prevent accumulation
CAMERA_BUFFER_SIZE = 1

# INCREASED timeouts untuk jarak jauh WiFi (signal unstable)
CAMERA_CONNECT_TIMEOUT = 25  # Increased: 10→25 (jarak jauh butuh lebih lama)
CAMERA_READ_TIMEOUT = 12.0   # Increased: 5→12 (setiap frame butuh tunggu lebih lama)

# FFmpeg RTSP options untuk low-latency (tapi robust untuk distant)
RTSP_FFMPEG_CAPTURE_OPTIONS = (
    "rtsp_transport;tcp|"
    "fflags;nobuffer+discardcorrupt+flush_packets|"
    "flags;low_delay|"
    "max_delay;0|"
    "analyzeduration;0|"
    "probesize;32|"
    "rtsp_transport;tcp"
)

CAMERA_ENFORCE_UNIFORM_CAPTURE = True
CAMERA_PREFERRED_FOURCC = "MJPG"
CAMERA_DISABLE_AUTO_ZOOM = True

# ============================================================================
# FRAME NORMALIZATION
# ============================================================================
FRAME_NORMALIZATION_MODE = "letterbox"
FRAME_NORMALIZATION_WIDTH = 640
FRAME_NORMALIZATION_HEIGHT = 480

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

MODEL_NAME = "yolov8n"  # Nano model
MODEL_TRACK_CLASS = 0   # Person class

# Lenient untuk occlusion
CONFIDENCE_THRESHOLD = 0.30
IOU_THRESHOLD = 0.40

# ============================================================================
# INFERENCE BACKEND
# ============================================================================
INFERENCE_BACKEND = "cpu"  # Stable untuk CPU
CUDA_HALF = False

INFERENCE_BATCH_SIZE = 1
INFERENCE_BATCH_WAIT_SECONDS = 0.001
INFERENCE_RESULT_TIMEOUT_SECONDS = 2.0

# ============================================================================
# TRACKING & STATE MACHINE - AGGRESSIVE untuk Distant WiFi
# ============================================================================

# INCREASED persistence untuk handle WiFi dropout jarak jauh
TRACK_PERSISTENCE = 120  # Increased: 60→120 (jarak jauh sering dropout)

BYTE_TRACK_TRACK_THRESH = 0.35
BYTE_TRACK_MATCH_THRESH = 0.75

# Line crossing (lebih strict untuk distant)
LINE_CROSSING_THRESHOLD = 15
DETECTION_MIN_CONSECUTIVE = 3
LINE_BUFFER_ZONE_PX = 25
LINE_CROSSING_COOLDOWN_FRAMES = 20
LINE_MIN_MOTION_PX = 5.0

# Stale cleanup
TRACK_STATE_STALE_FRAMES = 300

# ============================================================================
# THREADING & PERFORMANCE
# ============================================================================

MAX_CONCURRENT_INFERENCE = 1
VIDEO_CAPTURE_TIMEOUT = 12.0  # Increased: 5→12

FRAME_SKIP_DEFAULT = 3  # Jarak jauh butuh lebih aggressive
TARGET_FPS_PER_CAMERA = 10

# ============================================================================
# WEB SERVER
# ============================================================================

FLASK_HOST = "0.0.0.0"
FLASK_PORT = 5000

# Update interval untuk UI
SSE_UPDATE_INTERVAL = 0.25  # Still 0.25 untuk responsive

FLASK_DEBUG = False
ADMIN_RESET_PASSWORD = os.environ.get("ADMIN_RESET_PASSWORD", "admin")

# ============================================================================
# OPENVINO (Optional GPU)
# ============================================================================

USE_OPENVINO = False  # CPU mode
OPENVINO_DEVICE = "CPU"
OPENVINO_COMPILE_OPTIMIZATION = "LATENCY"
OPENVINO_AUTO_EXPORT = False

OPENVINO_XML_PATH = os.environ.get(
    "OPENVINO_XML_PATH",
    str(PROJECT_ROOT / f"{MODEL_NAME}_openvino_model" / f"{MODEL_NAME}.xml"),
)

# ============================================================================
# VISUALIZATION & DEBUG
# ============================================================================

ENABLE_DEBUG_DISPLAY = False
DEBUG_DISPLAY_WINDOW_WIDTH = 960
DEBUG_DISPLAY_WINDOW_HEIGHT = 540

# ============================================================================
# LOGGING
# ============================================================================

LOG_LEVEL = "WARNING"  # Reduce disk I/O
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)
DATABASE_FILE = PROJECT_ROOT / "edge_vision_counter.db"

# ============================================================================
# APPLICATION BEHAVIOR
# ============================================================================

ENABLE_GRACEFUL_DEGRADATION = True  # Continue jika 1 kamera down
HEALTH_CHECK_INTERVAL = 30

# ============================================================================
# mDNS / SERVICE DISCOVERY
# ============================================================================

MDNS_ENABLED = True
MDNS_SERVICE_NAME = os.environ.get("COUNTER_MDNS_SERVICE_NAME", "counter-3wifi")
MDNS_HOSTNAME = os.environ.get("COUNTER_MDNS_HOSTNAME", "counter-3wifi")

# ============================================================================
# OCCLUSION HANDLING (Aggressive untuk distant WiFi)
# ============================================================================

# Jarak jauh → signals bisa lemah → butuh lebih aggressive occlusion handling
EDGE_OCCLUSION_CONFIDENCE_FLOOR = 0.15
EDGE_OCCLUSION_CONFIDENCE_BOOST = 0.15  # Boost high untuk jarak jauh
EDGE_ZONE_MARGIN_PX = 100

MOTION_EXTRAPOLATION_ENABLED = True
LOST_TRACK_REACQ_DISTANCE_PX = 200  # Wider tolerance untuk jarak jauh

VIRTUAL_EDGE_EXPANSION_PX = 150

# ============================================================================
# SETUP INSTRUCTIONS
# ============================================================================
"""
KONFIGURASI UNTUK 3 WIFI KAMERA JARAK JAUH:

1. Update IP addresses di CAMERAS list:
   - Camera 1: 192.168.1.100 → Ganti dengan IP WiFi camera #1
   - Camera 2: 192.168.1.101 → Ganti dengan IP WiFi camera #2
   - Camera 3: 192.168.1.102 → Ganti dengan IP WiFi camera #3

2. Update credentials:
   - Ganti "admin" dengan username (default biasanya admin)
   - Ganti "password" dengan password camera

3. Update stream path (jika perlu):
   - Default: /stream1
   - Bisa juga: /stream, /main, /sub, etc (check camera manual)

4. Test connectivity dulu:
   bash> ffprobe rtsp://admin:password@192.168.1.100:554/stream1
   - Harus show: codec info, resolution, fps

5. Startup dan monitor:
   bash> python main.py
   - Harusnya: "3 cameras connected" dalam log
   - Jika ada yang timeout: increase CAMERA_CONNECT_TIMEOUT ke 30-40s

6. Adjust settings untuk performance:

   Jika latency tinggi (>500ms):
   - Increase frame_skip: 3 → 4 (process every 4th frame)
   - Or reduce CAMERA_READ_TIMEOUT: 12 → 10

   Jika video sering drop/freeze:
   - Increase TRACK_PERSISTENCE: 120 → 150
   - Check WiFi signal strength (move router closer?)
   - Or increase CAMERA_CONNECT_TIMEOUT: 25 → 35

   Jika accuracy kurang bagus:
   - Adjust confidence threshold: 0.30 → 0.25
   - Or increase line_crossing patience: adjust LINE_BUFFER_ZONE_PX

EXPECTED PERFORMANCE (3 WiFi Distant):
- Latency: 200-500ms (tergantung distance & signal)
- CPU: 70-80% (3 cameras, frame_skip=3)
- Accuracy: ~90-95% (dengan occlusion handling)
- Uptime: ~95%+ (WiFi bisa drop 5-10% of time)

TROUBLESHOOTING:

Problem: "Connection timeout"
→ Increase CAMERA_CONNECT_TIMEOUT ke 30-40s
→ Check IP address (ping 192.168.1.100)
→ Check WiFi signal strength
→ Check camera firewall/port 554

Problem: "Frequent disconnects"
→ Increase TRACK_PERSISTENCE: 120 → 150
→ Check WiFi 2.4GHz vs 5GHz (try switch)
→ Check for interference (WiFi analyzer app)
→ Consider WiFi repeater/extender

Problem: "Low frame rate"
→ Increase frame_skip: 3 → 4-5
→ Reduce CAMERA_FRAME_WIDTH/HEIGHT: 640 → 480
→ Check WiFi bandwidth (speedtest)

Problem: "CPU high >85%"
→ Increase frame_skip: 3 → 4-5
→ Reduce inference resolution
→ Or reduce INFERENCE_BATCH_WAIT_SECONDS
"""
