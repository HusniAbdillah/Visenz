"""
Auto-healing threaded video reader for edge-vision-counter V2.
Pure video stream handling with watchdog logic for Wi-Fi/IP cameras.
Strictly handles pulling frames via cv2.VideoCapture with auto-reconnect.
"""

import logging
import threading
import time
from typing import Optional

import cv2
import numpy as np

import config

logger = logging.getLogger(__name__)


def parse_source(url: str):
    """
    Parse camera source URL to handle both webcam indices and IP camera URLs.
    
    Args:
        url: Camera source URL (string digit for webcam like "0", "1" or URL for IP camera)
    
    Returns:
        Integer for webcam index, or string URL for IP/RTSP cameras
    """
    if isinstance(url, str):
        if url.isdigit():
            return int(url)
        if url.startswith(("http", "rtsp", "https")):
            return url
    return url


class ThreadedVideoReader:
    """
    Auto-healing threaded video capture for Wi-Fi/IP cameras.
    
    Features:
    - Keeps cv2.CAP_PROP_BUFFERSIZE at 1 to prevent lag
    - Only keeps the most recent frame in a thread-safe variable
    - Watchdog logic: if cap.read() fails, automatically reconnect
    - Thread never dies on stream drop, sleeps and retries
    """

    RECONNECT_DELAY_SECONDS: float = 3.0
    MAX_CONSECUTIVE_FAILURES: int = 30

    def __init__(
        self,
        camera_id: str,
        camera_url: str,
        buffer_size: int = 1
    ) -> None:
        """
        Initialize threaded video reader.

        Args:
            camera_id: Unique identifier for this camera.
            camera_url: Video source URL or device index.
            buffer_size: OpenCV buffer size (1 = minimal buffering for Wi-Fi).
        """
        self.camera_id: str = camera_id
        self.camera_url: str = camera_url
        self._buffer_size: int = buffer_size

        self._cap: Optional[cv2.VideoCapture] = None
        self._latest_frame: Optional[np.ndarray] = None
        self._frame_lock: threading.Lock = threading.Lock()
        self._running: bool = False
        self._reader_thread: Optional[threading.Thread] = None
        self._connected: bool = False
        self._frame_width: int = 0
        self._frame_height: int = 0
        self._reconnect_count: int = 0

    def start(self) -> bool:
        """
        Start the background reading thread.

        Returns:
            True if initial connection successful, False otherwise.
        """
        success = self._initialize_capture()
        
        self._running = True
        self._reader_thread = threading.Thread(
            target=self._read_loop,
            name=f"VideoReader-{self.camera_id}",
            daemon=True
        )
        self._reader_thread.start()
        
        logger.info(
            "Video reader thread started for camera '%s' (%s). Initial connection: %s",
            self.camera_id,
            self.camera_url,
            "success" if success else "pending"
        )
        return success

    def _initialize_capture(self) -> bool:
        """
        Initialize or reinitialize the video capture.

        Returns:
            True if successful, False otherwise.
        """
        try:
            if self._cap is not None:
                try:
                    self._cap.release()
                except Exception:
                    pass
                self._cap = None

            logger.info(
                "Opening video source for camera '%s': %s",
                self.camera_id,
                self.camera_url
            )

            source = parse_source(self.camera_url)
            
            if isinstance(source, int):
                self._cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
            else:
                self._cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)

            if not self._cap.isOpened():
                logger.warning(
                    "Failed to open camera '%s' at %s",
                    self.camera_id,
                    self.camera_url
                )
                self._connected = False
                return False

            # Optimization settings
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, self._buffer_size)
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_FRAME_WIDTH)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_FRAME_HEIGHT)
            self._cap.set(cv2.CAP_PROP_FPS, config.CAMERA_FPS)

            self._frame_width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self._frame_height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            self._connected = True
            logger.info(
                "Camera '%s' connected via %s. Resolution: %dx%d",
                self.camera_id,
                "DirectShow" if isinstance(source, int) else "FFMPEG",
                self._frame_width,
                self._frame_height
            )
            return True

        except Exception as e:
            logger.error(
                "Exception initializing capture for camera '%s': %s",
                self.camera_id,
                str(e)
            )
            self._connected = False
            return False

    def _read_loop(self) -> None:
        """
        Background thread loop: continuously read frames and keep only the latest.
        Implements watchdog logic: on failure, sleep and retry connection.
        Thread never dies on stream drop.
        """
        consecutive_failures: int = 0

        while self._running:
            try:
                if self._cap is None or not self._cap.isOpened():
                    self._handle_reconnect()
                    continue

                ret, frame = self._cap.read()

                if not ret or frame is None:
                    consecutive_failures += 1
                    
                    if consecutive_failures > self.MAX_CONSECUTIVE_FAILURES:
                        logger.warning(
                            "Camera '%s': %d consecutive read failures. Triggering reconnect.",
                            self.camera_id,
                            consecutive_failures
                        )
                        self._handle_reconnect()
                        consecutive_failures = 0
                    else:
                        time.sleep(0.01)
                    continue

                consecutive_failures = 0
                self._connected = True

                with self._frame_lock:
                    self._latest_frame = frame

            except cv2.error as cv_err:
                logger.warning(
                    "Camera '%s' OpenCV error: %s. Triggering reconnect.",
                    self.camera_id,
                    str(cv_err)
                )
                self._handle_reconnect()
                consecutive_failures = 0

            except Exception as e:
                logger.error(
                    "Camera '%s' unexpected error in read loop: %s. Triggering reconnect.",
                    self.camera_id,
                    str(e)
                )
                self._handle_reconnect()
                consecutive_failures = 0

    def _handle_reconnect(self) -> None:
        """
        Handle reconnection with watchdog sleep.
        Sleep for configured delay then attempt to reinitialize capture.
        """
        self._connected = False
        self._reconnect_count += 1

        logger.info(
            "Camera '%s': Sleeping %.1f seconds before reconnect attempt #%d",
            self.camera_id,
            self.RECONNECT_DELAY_SECONDS,
            self._reconnect_count
        )

        time.sleep(self.RECONNECT_DELAY_SECONDS)

        if self._running:
            success = self._initialize_capture()
            if success:
                logger.info(
                    "Camera '%s': Reconnect attempt #%d successful.",
                    self.camera_id,
                    self._reconnect_count
                )
            else:
                logger.warning(
                    "Camera '%s': Reconnect attempt #%d failed. Will retry.",
                    self.camera_id,
                    self._reconnect_count
                )

    def get_frame(self) -> Optional[np.ndarray]:
        """
        Get the latest available frame (non-blocking).

        Returns:
            Copy of latest frame or None if no frame available.
        """
        with self._frame_lock:
            if self._latest_frame is not None:
                return self._latest_frame.copy()
            return None

    def get_frame_dimensions(self) -> tuple:
        """
        Get the frame dimensions.

        Returns:
            Tuple of (width, height).
        """
        return (self._frame_width, self._frame_height)

    def is_connected(self) -> bool:
        """
        Check if the camera is currently connected.

        Returns:
            True if connected and reading frames, False otherwise.
        """
        return self._connected

    def get_reconnect_count(self) -> int:
        """
        Get the number of reconnection attempts made.

        Returns:
            Number of reconnect attempts.
        """
        return self._reconnect_count

    def stop(self) -> None:
        """
        Stop the reader thread and release resources.
        """
        logger.info("Stopping video reader for camera '%s'", self.camera_id)
        
        self._running = False

        if self._reader_thread is not None:
            self._reader_thread.join(timeout=5.0)
            if self._reader_thread.is_alive():
                logger.warning(
                    "Camera '%s': Reader thread did not terminate within timeout.",
                    self.camera_id
                )

        if self._cap is not None:
            try:
                self._cap.release()
            except Exception as e:
                logger.error(
                    "Camera '%s': Error releasing capture: %s",
                    self.camera_id,
                    str(e)
                )
            self._cap = None

        self._connected = False
        logger.info("Video reader stopped for camera '%s'", self.camera_id)

    def __del__(self) -> None:
        """Destructor to ensure resources are released."""
        if self._running:
            self.stop()
