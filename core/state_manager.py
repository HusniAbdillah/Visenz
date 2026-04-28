"""
Thread-safe state persistence manager for edge-vision-counter V2.

Stores the current counts in SQLite and keeps a JSON snapshot for backward
compatibility. Crossing events are journaled in SQLite for time-series analytics.
"""

from __future__ import annotations

import json
import logging
import threading
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
from uuid import uuid4

import pytz

import config
from core.analytics_db import AnalyticsDB

logger = logging.getLogger(__name__)
WIB_TZ = pytz.timezone("Asia/Jakarta")


class StateManager:
    """Thread-safe state persistence and analytics coordinator."""

    def __init__(self, state_file: str = "state.json", db_file: str = "edge_vision_counter.db") -> None:
        self._state_path: Path = config.PROJECT_ROOT / state_file
        self._db_path: Path = config.PROJECT_ROOT / db_file
        self._analytics_db = AnalyticsDB(self._db_path)
        self._lock: threading.Lock = threading.Lock()
        self._session_lock: threading.RLock = threading.RLock()
        self._total_in: int = 0
        self._total_out: int = 0
        self._session_id: str = self._new_session_id()
        self._last_reset_at: str = self._wib_now()
        self._load_state()
        self._ensure_daily_session()

    @staticmethod
    def _wib_now() -> str:
        return datetime.now(WIB_TZ).strftime("%Y-%m-%d %H:%M:%S")

    @staticmethod
    def _new_session_id() -> str:
        return datetime.now(WIB_TZ).strftime("%Y%m%d%H%M%S") + "-" + uuid4().hex[:8]

    @staticmethod
    def _wib_today() -> str:
        return datetime.now(WIB_TZ).strftime("%Y-%m-%d")

    def _load_state(self) -> None:
        """
        Load state from JSON file if it exists.
        If file does not exist or is corrupted, start with zeros.
        """
        with self._lock:
            try:
                db_state = self._analytics_db.load_state()
                self._total_in = int(db_state.get("total_in", 0))
                self._total_out = int(db_state.get("total_out", 0))
                self._session_id = str(db_state.get("session_id", self._new_session_id()))
                self._last_reset_at = str(db_state.get("last_reset_at", self._wib_now()))
                logger.info(
                    "State loaded from SQLite %s: total_in=%d, total_out=%d, session_id=%s",
                    self._db_path,
                    self._total_in,
                    self._total_out,
                    self._session_id,
                )
                self._save_state()
                return
            except Exception as db_error:
                logger.warning("Failed to load state from SQLite %s: %s", self._db_path, str(db_error))

            try:
                if self._state_path.exists():
                    with open(self._state_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        self._total_in = int(data.get("total_in", 0))
                        self._total_out = int(data.get("total_out", 0))
                        self._session_id = str(data.get("session_id", self._new_session_id()))
                        self._last_reset_at = str(data.get("last_reset_at", self._wib_now()))
                        logger.info(
                            "State loaded from %s: total_in=%d, total_out=%d",
                            self._state_path,
                            self._total_in,
                            self._total_out,
                        )
                else:
                    logger.info("State file not found at %s. Starting with zeros.", self._state_path)
                    self._total_in = 0
                    self._total_out = 0
                    self._session_id = self._new_session_id()
                    self._last_reset_at = self._wib_now()
                self._analytics_db.save_state(self._total_in, self._total_out, self._session_id, self._last_reset_at)
            except (json.JSONDecodeError, ValueError, TypeError) as e:
                logger.warning("Failed to parse state file %s: %s. Starting with zeros.", self._state_path, str(e))
                self._total_in = 0
                self._total_out = 0
                self._session_id = self._new_session_id()
                self._last_reset_at = self._wib_now()
            except Exception as e:
                logger.error("Unexpected error loading state from %s: %s. Starting with zeros.", self._state_path, str(e))
                self._total_in = 0
                self._total_out = 0
                self._session_id = self._new_session_id()
                self._last_reset_at = self._wib_now()

    def _ensure_daily_session(self) -> None:
        """Roll the live session forward when the stored session belongs to a previous day."""
        stored_day = str(self._last_reset_at)[:10]
        today = self._wib_today()
        if stored_day == today:
            return

        logger.info(
            "Stored session day %s differs from today %s. Starting a fresh daily session.",
            stored_day,
            today,
        )
        self.reset()

    def _save_state(self) -> None:
        """
        Persist current state to JSON file.
        Must be called while holding the lock.
        """
        try:
            data = {
                "total_in": self._total_in,
                "total_out": self._total_out,
                "session_id": self._session_id,
                "last_reset_at": self._last_reset_at,
            }
            with open(self._state_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            self._analytics_db.save_state(self._total_in, self._total_out, self._session_id, self._last_reset_at)
            logger.debug(
                "State saved to %s: total_in=%d, total_out=%d, session_id=%s",
                self._state_path,
                self._total_in,
                self._total_out,
                self._session_id,
            )
        except Exception as e:
            logger.error("Failed to save state to %s: %s", self._state_path, str(e))

    def add_in(self, count: int = 1) -> None:
        """
        Increment total_in count and persist state.

        Args:
            count: Number to add (default 1).
        """
        self.record_crossing(camera_id="legacy", direction="IN", count=count)

    def add_out(self, count: int = 1) -> None:
        """
        Increment total_out count and persist state.

        Args:
            count: Number to add (default 1).
        """
        self.record_crossing(camera_id="legacy", direction="OUT", count=count)

    def record_crossing(
        self,
        camera_id: str,
        direction: str,
        count: int = 1,
        timestamp: Optional[str] = None,
    ) -> None:
        """Record one or more crossing events for a camera."""
        direction = direction.upper()
        if direction not in {"IN", "OUT"}:
            raise ValueError(f"Invalid direction: {direction}")

        if count <= 0:
            return

        timestamp_wib = timestamp or self._wib_now()

        with self._session_lock:
            with self._lock:
                current_session_id = self._session_id
                events = [(timestamp_wib, camera_id, direction, current_session_id) for _ in range(int(count))]

                if direction == "IN":
                    self._total_in += int(count)
                else:
                    self._total_out += int(count)

                self._analytics_db.record_events(events)
                self._save_state()

                logger.info(
                    "Recorded %d %s crossing(s) for camera '%s' in session '%s'. Totals: in=%d, out=%d",
                    count,
                    direction,
                    camera_id,
                    current_session_id,
                    self._total_in,
                    self._total_out,
                )

    def get_total_in(self) -> int:
        """
        Get current total_in count.

        Returns:
            Current total_in value.
        """
        with self._lock:
            return self._total_in

    def get_total_out(self) -> int:
        """
        Get current total_out count.

        Returns:
            Current total_out value.
        """
        with self._lock:
            return self._total_out

    def get_current_inside(self) -> int:
        """
        Get current number of people inside (total_in - total_out).

        Returns:
            Current inside count (can be negative if more exits than entries).
        """
        with self._lock:
            return self._total_in - self._total_out

    def get_stats(self) -> Dict[str, Any]:
        """
        Get all statistics as a dictionary.

        Returns:
            Dictionary with total_in, total_out, and current_inside.
        """
        with self._lock:
            return {
                "total_in": self._total_in,
                "total_out": self._total_out,
                "current_inside": self._total_in - self._total_out,
                "session_id": self._session_id,
                "last_reset_at": self._last_reset_at,
                "timezone": "Asia/Jakarta",
            }

    def get_analytics(self, interval_minutes: int = 15, trend_granularity: str = "day") -> Dict[str, Any]:
        """Return time-series and per-camera aggregates for the current session."""
        with self._lock:
            start_at = self._last_reset_at
            session_id = self._session_id
            granularity = str(trend_granularity).lower()
            effective_interval_minutes = 1440 if granularity == "day" else int(interval_minutes)
            # For day-granularity, align day buckets to the timestamp of the last recorded event
            align_start = False
            if granularity == "day":
                last_event = self._analytics_db.fetch_last_event_time(session_id)
                if last_event:
                    # start_at remains last_reset_at for filtering, but buckets will align to last_event
                    align_start = True
                    bucket_anchor = last_event
                else:
                    bucket_anchor = start_at
            else:
                bucket_anchor = start_at

        return {
            "interval_minutes": effective_interval_minutes,
            "trend_granularity": granularity,
            "start_at": start_at,
            "session_id": session_id,
            "timezone": "Asia/Jakarta",
            "time_series": self._analytics_db.fetch_time_series(
                start_at,
                session_id=session_id,
                interval_minutes=interval_minutes,
                granularity=granularity,
                align_start=align_start if granularity == "day" else False,
            ),
            "camera_summary": self._analytics_db.fetch_camera_summary(start_at, session_id=session_id),
        }

    def get_event_logs(self, only_current_session: bool = True) -> Dict[str, Any]:
        with self._lock:
            session_id = self._session_id
        rows = self._analytics_db.fetch_logs(session_id=session_id if only_current_session else None)
        return {
            "session_id": session_id,
            "timezone": "Asia/Jakarta",
            "rows": rows,
        }

    def set_current_inside(self, target: int) -> None:
        """
        Adjust total_out so that (total_in - total_out) equals the target value.
        This allows manual calibration of the current inside count.

        Args:
            target: Desired current_inside value.
        """
        with self._session_lock:
            with self._lock:
                new_total_out = self._total_in - target
                old_total_out = self._total_out
                self._total_out = new_total_out
                self._save_state()
                logger.info(
                    "Manual calibration: target_inside=%d, adjusted total_out from %d to %d",
                    target,
                    old_total_out,
                    self._total_out
                )

    def reset(self) -> None:
        """
        Start a new active session and reset live counters only.
        """
        with self._session_lock:
            with self._lock:
                self._total_in = 0
                self._total_out = 0
                self._session_id = self._new_session_id()
                self._last_reset_at = self._wib_now()
                self._save_state()
                logger.info("State reset for new session '%s'. total_in=0, total_out=0", self._session_id)

    @contextmanager
    def session_mutation(self):
        """Serialize session-changing operations against live counting."""
        with self._session_lock:
            yield

    def bulk_update(self, new_in: int, new_out: int, camera_id: str = "legacy", timestamp: Optional[str] = None) -> None:
        """
        Add multiple counts at once and log them as individual events.

        Args:
            new_in: Number of new entries to add.
            new_out: Number of new exits to add.
            camera_id: Camera identifier for event logs.
            timestamp: Optional timestamp shared by the batched events.
        """
        if new_in == 0 and new_out == 0:
            return

        if new_in > 0:
            self.record_crossing(camera_id=camera_id, direction="IN", count=new_in, timestamp=timestamp)
        if new_out > 0:
            self.record_crossing(camera_id=camera_id, direction="OUT", count=new_out, timestamp=timestamp)
