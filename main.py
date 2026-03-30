"""
Main entry point for edge-vision-counter multi-camera system.
Orchestrates model loading, camera manager initialization, and Flask web server startup.
Implements graceful shutdown on SIGINT (Ctrl+C) with resource cleanup.
"""

import sys
import logging
import logging.handlers
import signal
import threading
from pathlib import Path

import config
from core.detector import VisionModel
from core.tracker_manager import CameraManager
from web.app import run_app, set_camera_manager, app

logger = logging.getLogger(__name__)


def setup_logging() -> None:
    """Configure Python logging to file and console."""
    log_format = logging.Formatter(config.LOG_FORMAT)
    log_level = getattr(logging, config.LOG_LEVEL.upper(), logging.INFO)

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Clear existing handlers
    root_logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(log_format)
    root_logger.addHandler(console_handler)

    # File handler (rotating)
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
        print(f"Warning: Could not set up file logging: {e}")

    logger.info("Logging system initialized at level: %s", config.LOG_LEVEL)


def main() -> None:
    """
    Main application entry point.
    - Sets up logging
    - Loads vision model with OpenVINO acceleration attempt
    - Initializes camera manager with all configured cameras
    - Injects manager into Flask app
    - Starts Flask web server (blocking call)
    - Handles graceful shutdown on SIGINT
    """
    camera_manager = None

    def signal_handler(sig, frame):
        """Handle interrupt signal gracefully."""
        logger.info("Interrupt signal received. Shutting down...")
        if camera_manager:
            camera_manager.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    try:
        setup_logging()
        logger.info("=" * 70)
        logger.info("Edge Vision Counter - Multi-Camera System")
        logger.info("Starting application...")
        logger.info("=" * 70)

        try:
            logger.info("Loading vision model...")
            model = VisionModel()
            logger.info("Vision model loaded successfully. Device: %s", model.get_device())

            logger.info("Initializing camera manager...")
            camera_manager = CameraManager(model)

            enabled_cameras = [c for c in config.CAMERAS if c.get("enabled", True)]
            logger.info("Found %d enabled camera(s) in configuration", len(enabled_cameras))

            for camera_config in enabled_cameras:
                if camera_manager.add_camera(camera_config):
                    logger.info("Camera '%s' added: %s", camera_config["id"], camera_config["url"])
                else:
                    logger.warning("Failed to add camera '%s'", camera_config["id"])

            if not enabled_cameras:
                logger.error("No cameras configured. Please update config.py")
                sys.exit(1)

            logger.info("Starting camera manager...")
            camera_manager.start()
            logger.info("Camera manager started. Processing %d camera(s)", len(enabled_cameras))

            set_camera_manager(camera_manager)
            logger.info("Camera manager injected into Flask app")

            logger.info("=" * 70)
            logger.info("Starting Flask web server...")
            logger.info("Open your browser to http://localhost:%d", config.FLASK_PORT)
            logger.info("=" * 70)

            run_app()

        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")
        except Exception as e:
            logger.error("Application error: %s", str(e), exc_info=True)
            raise

    except Exception as e:
        logger.critical("Critical application error: %s", str(e), exc_info=True)
        sys.exit(1)
    finally:
        logger.info("=" * 70)
        logger.info("Shutting down...")
        try:
            if camera_manager:
                camera_manager.stop()
                logger.info("Camera manager stopped")
        except Exception as e:
            logger.error("Error during shutdown: %s", str(e))

        logger.info("Edge Vision Counter - Multi-Camera System stopped")
        logger.info("=" * 70)


if __name__ == "__main__":
    main()

