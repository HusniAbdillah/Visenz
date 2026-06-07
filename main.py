"""
Main entry point for edge-vision-counter V2 (Supervision Edition).
Orchestrates StateManager, VisionModel, CameraManager, and Flask web server.
Implements graceful shutdown on SIGINT (Ctrl+C) with resource cleanup.
"""

import sys
import logging
import logging.handlers
import signal
import threading
import atexit
import time
from typing import Optional

import cv2

import config
from core.detector import VisionModel
from core.state_manager import StateManager
from core.tracker_manager import CameraManager
from web.app import run_app, set_camera_manager, set_state_manager, stop_mdns_service

logger = logging.getLogger(__name__)

camera_manager: Optional[CameraManager] = None
state_manager: Optional[StateManager] = None
shutdown_in_progress: bool = False


def _log_runtime_metrics(model: VisionModel, state: StateManager) -> None:
    """Periodically log GPU and counting snapshots for long-running sessions."""
    while not shutdown_in_progress:
        try:
            stats = state.get_stats()
            gpu_stats = model.get_gpu_stats()
            logger.info(
                "Runtime snapshot | IN=%d OUT=%d INSIDE=%d | GPU util=%s%% mem=%s/%s MB queue=%s",
                stats['total_in'],
                stats['total_out'],
                stats['current_inside'],
                gpu_stats.get('utilization_percent'),
                gpu_stats.get('memory_used_mb'),
                gpu_stats.get('memory_total_mb'),
                gpu_stats.get('batch_queue_depth'),
            )
        except Exception as exc:
            logger.warning("Failed to log runtime metrics: %s", str(exc))

        for _ in range(60):
            if shutdown_in_progress:
                return
            time.sleep(1)


def setup_logging() -> None:
    """Configure Python logging to file and console."""
    log_format = logging.Formatter(config.LOG_FORMAT)
    log_level = getattr(logging, config.LOG_LEVEL.upper(), logging.INFO)

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    root_logger.handlers.clear()

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(log_format)
    root_logger.addHandler(console_handler)

    log_file = config.LOG_DIR / "edge_vision_counter.log"
    try:
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,
            backupCount=5,
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(log_format)
        root_logger.addHandler(file_handler)
    except Exception as e:
        logger.warning("Could not set up file logging: %s", str(e))

    logger.info("Logging system initialized at level: %s", config.LOG_LEVEL)


def graceful_shutdown() -> None:
    """
    Perform graceful shutdown of all components.
    Stops camera manager, releases cv2 windows, and cleans up resources.
    """
    global camera_manager, shutdown_in_progress

    if shutdown_in_progress:
        return

    shutdown_in_progress = True

    logger.info("=" * 70)
    logger.info("Initiating graceful shutdown...")

    try:
        if camera_manager is not None:
            logger.info("Stopping camera manager...")
            camera_manager.stop()
            logger.info("Camera manager stopped")
    except Exception as e:
        logger.error("Error stopping camera manager: %s", str(e))

    try:
        cv2.destroyAllWindows()
        logger.info("OpenCV windows destroyed")
    except Exception as e:
        logger.error("Error destroying OpenCV windows: %s", str(e))

    try:
        stop_mdns_service()
    except Exception as e:
        logger.debug("Error stopping mDNS service: %s", str(e))

    logger.info("Edge Vision Counter V2 - Shutdown complete")
    logger.info("=" * 70)


def signal_handler(sig, frame) -> None:
    """
    Handle interrupt signals (SIGINT, SIGTERM) gracefully.

    Args:
        sig: Signal number.
        frame: Current stack frame.
    """
    signal_name = signal.Signals(sig).name
    logger.info("Received signal: %s", signal_name)
    graceful_shutdown()
    sys.exit(0)


def main() -> None:
    """
    Main application entry point for Edge Vision Counter V2.

    Workflow:
    1. Setup logging
    2. Initialize StateManager (loads persisted state)
    3. Load VisionModel with hardware acceleration
    4. Initialize CameraManager with StateManager
    5. Add and start all configured cameras
    6. Inject managers into Flask app
    7. Run Flask web server (blocking)
    8. Handle graceful shutdown on exit
    """
    global camera_manager, state_manager

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    atexit.register(graceful_shutdown)

    try:
        setup_logging()

        logger.info("=" * 70)
        logger.info("Edge Vision Counter V2 (Supervision Edition)")
        logger.info("Enterprise People Counting System")
        logger.info("=" * 70)

        logger.info("Initializing StateManager...")
        state_manager = StateManager()
        current_stats = state_manager.get_stats()
        logger.info(
            "StateManager initialized. Resumed state: IN=%d, OUT=%d, INSIDE=%d",
            current_stats['total_in'],
            current_stats['total_out'],
            current_stats['current_inside']
        )

        logger.info("Loading VisionModel...")
        model = VisionModel()
        logger.info("VisionModel loaded. Inference device: %s", model.get_device())

        metrics_thread = threading.Thread(
            target=_log_runtime_metrics,
            args=(model, state_manager),
            name="RuntimeMetricsLogger",
            daemon=True,
        )
        metrics_thread.start()

        logger.info("Initializing CameraManager...")
        camera_manager = CameraManager(model=model, state_manager=state_manager)

        enabled_cameras = [c for c in config.CAMERAS if c.get("enabled", True)]
        logger.info("Found %d enabled camera(s) in configuration", len(enabled_cameras))

        if not enabled_cameras:
            logger.error("No cameras configured. Please update config.py with at least one enabled camera.")
            sys.exit(1)

        for cam_config in enabled_cameras:
            camera_id = cam_config.get("id", "unknown")
            camera_url = cam_config.get("url", "")

            if camera_manager.add_camera(cam_config):
                logger.info(
                    "Camera '%s' added successfully. URL: %s",
                    camera_id,
                    camera_url
                )
            else:
                logger.warning("Failed to add camera '%s'", camera_id)

        logger.info("Starting CameraManager...")
        camera_manager.start()
        logger.info(
            "CameraManager started. Processing %d camera(s)",
            len(enabled_cameras)
        )

        set_state_manager(state_manager)
        set_camera_manager(camera_manager)
        logger.info("StateManager and CameraManager injected into Flask app")

        logger.info("=" * 70)
        logger.info("Starting Flask web server...")
        logger.info("Dashboard URL: http://localhost:%d", config.FLASK_PORT)
        logger.info("Dashboard mDNS URL: http://%s.local:%d", config.MDNS_HOSTNAME, config.FLASK_PORT)
        logger.info("Admin APIs:")
        logger.info("  - POST /api/set_count : Manual calibration")
        logger.info("  - POST /api/reset     : Reset all counts")
        logger.info("  - GET  /stream        : SSE real-time updates")
        logger.info("  - GET  /health        : System health check")
        logger.info("=" * 70)

        run_app()

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received in main")
    except Exception as e:
        logger.critical("Critical application error: %s", str(e), exc_info=True)
        sys.exit(1)
    finally:
        graceful_shutdown()


if __name__ == "__main__":
    main()

