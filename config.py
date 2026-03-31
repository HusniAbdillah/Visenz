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
        "url": "http://192.168.1.8:8080/video",
        "orientation": "vertical",  # 'vertical' or 'horizontal'
        "line_ratio": 0.5,  # 0.0 to 1.0 (50% of width for vertical, 50% of height for horizontal)
        "in_direction": "right_to_left",  # Options: 'left_to_right', 'right_to_left', 'top_to_bottom', 'bottom_to_top'
        "frame_skip": 3,  # Process every Nth frame for inference (1=every frame, 3=every 3rd)
        "enabled": True,
    },
    {
        "id": "Cam_Pintu_Belakang",
        "url": "http://192.168.1.38:4747/video",
        "orientation": "vertical",  # 'vertical' or 'horizontal'
        "line_ratio": 0.5,  # 0.0 to 1.0 (50% of width for vertical, 50% of height for horizontal)
        "in_direction": "left_to_right",  # Options: 'left_to_right', 'right_to_left', 'top_to_bottom', 'bottom_to_top'
        "frame_skip": 3,  # Process every Nth frame for inference (1=every frame, 3=every 3rd)
        "enabled": True,
    },
    {
        "id": "Cam_Pintu_Samping",
        "url": "0",
        "orientation": "vertical",  # 'vertical' or 'horizontal'
        "line_ratio": 0.5,  # 0.0 to 1.0 (50% of width for vertical, 50% of height for horizontal)
        "in_direction": "right_to_left",  # Options: 'left_to_right', 'right_to_left', 'top_to_bottom', 'bottom_to_top'
        "frame_skip": 3,  # Process every Nth frame for inference (1=every frame, 3=every 3rd)
        "enabled": True,
    },
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

# Model Configuration
MODEL_NAME = "yolov8n"  # Nano model for edge devices
MODEL_TRACK_CLASS = 0  # 0 = person class in COCO dataset
CONFIDENCE_THRESHOLD = 0.35
IOU_THRESHOLD = 0.45

# Tracking & State Machine
TRACK_PERSISTENCE = 30  # Frames to keep track after detection lost
LINE_CROSSING_THRESHOLD = 10  # Minimum pixels to cross the line to count
DETECTION_MIN_CONSECUTIVE = 2  # Min frames to track before counting

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

# OpenVINO Configuration (Intel Iris Xe Acceleration)
USE_OPENVINO = True
OPENVINO_DEVICE = "GPU"  # 'GPU' for iGPU, 'CPU' as fallback
OPENVINO_COMPILE_OPTIMIZATION = "LATENCY"  # 'LATENCY' or 'THROUGHPUT'

# Visualization & Debug
ENABLE_DEBUG_DISPLAY = True  # Show cv2.imshow() with bboxes, lines, IDs
DEBUG_DISPLAY_WINDOW_WIDTH = 960
DEBUG_DISPLAY_WINDOW_HEIGHT = 540

# Logging Configuration
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

# Application Behavior
ENABLE_GRACEFUL_DEGRADATION = True  # Continue with available cameras if one fails
HEALTH_CHECK_INTERVAL = 30  # Seconds between camera health checks
