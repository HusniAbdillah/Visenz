"""
Thread-safe state persistence manager for edge-vision-counter V2.
Handles saving and loading of counting state to local JSON file.
Provides atomic operations for count manipulation with threading.Lock().
"""

import json
import logging
import threading
from pathlib import Path
from typing import Dict, Any

import config

logger = logging.getLogger(__name__)


class StateManager:
    """
    Thread-safe state persistence manager.
    Saves total_in and total_out to a local state.json file.
    Resumes from existing state on initialization if file exists.
    All read/write operations are protected by threading.Lock().
    """

    def __init__(self, state_file: str = "state.json") -> None:
        """
        Initialize the state manager.

        Args:
            state_file: Name of the JSON file to persist state.
        """
        self._state_path: Path = config.PROJECT_ROOT / state_file
        self._lock: threading.Lock = threading.Lock()
        self._total_in: int = 0
        self._total_out: int = 0
        self._load_state()

    def _load_state(self) -> None:
        """
        Load state from JSON file if it exists.
        If file does not exist or is corrupted, start with zeros.
        """
        with self._lock:
            try:
                if self._state_path.exists():
                    with open(self._state_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        self._total_in = int(data.get('total_in', 0))
                        self._total_out = int(data.get('total_out', 0))
                        logger.info(
                            "State loaded from %s: total_in=%d, total_out=%d",
                            self._state_path,
                            self._total_in,
                            self._total_out
                        )
                else:
                    logger.info(
                        "State file not found at %s. Starting with zeros.",
                        self._state_path
                    )
                    self._total_in = 0
                    self._total_out = 0
            except (json.JSONDecodeError, ValueError, TypeError) as e:
                logger.warning(
                    "Failed to parse state file %s: %s. Starting with zeros.",
                    self._state_path,
                    str(e)
                )
                self._total_in = 0
                self._total_out = 0
            except Exception as e:
                logger.error(
                    "Unexpected error loading state from %s: %s. Starting with zeros.",
                    self._state_path,
                    str(e)
                )
                self._total_in = 0
                self._total_out = 0

    def _save_state(self) -> None:
        """
        Persist current state to JSON file.
        Must be called while holding the lock.
        """
        try:
            data = {
                'total_in': self._total_in,
                'total_out': self._total_out,
            }
            with open(self._state_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            logger.debug(
                "State saved to %s: total_in=%d, total_out=%d",
                self._state_path,
                self._total_in,
                self._total_out
            )
        except Exception as e:
            logger.error("Failed to save state to %s: %s", self._state_path, str(e))

    def add_in(self, count: int = 1) -> None:
        """
        Increment total_in count and persist state.

        Args:
            count: Number to add (default 1).
        """
        with self._lock:
            self._total_in += count
            self._save_state()
            logger.info("Added %d to IN count. New total_in=%d", count, self._total_in)

    def add_out(self, count: int = 1) -> None:
        """
        Increment total_out count and persist state.

        Args:
            count: Number to add (default 1).
        """
        with self._lock:
            self._total_out += count
            self._save_state()
            logger.info("Added %d to OUT count. New total_out=%d", count, self._total_out)

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
                'total_in': self._total_in,
                'total_out': self._total_out,
                'current_inside': self._total_in - self._total_out,
            }

    def set_current_inside(self, target: int) -> None:
        """
        Adjust total_out so that (total_in - total_out) equals the target value.
        This allows manual calibration of the current inside count.

        Args:
            target: Desired current_inside value.
        """
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
        Reset all counts to zero and persist state.
        """
        with self._lock:
            self._total_in = 0
            self._total_out = 0
            self._save_state()
            logger.info("State reset to zero. total_in=0, total_out=0")

    def bulk_update(self, new_in: int, new_out: int) -> None:
        """
        Add multiple counts at once (for efficiency).

        Args:
            new_in: Number of new entries to add.
            new_out: Number of new exits to add.
        """
        if new_in == 0 and new_out == 0:
            return

        with self._lock:
            self._total_in += new_in
            self._total_out += new_out
            self._save_state()
            if new_in > 0 or new_out > 0:
                logger.info(
                    "Bulk update: added in=%d, out=%d. Totals: in=%d, out=%d",
                    new_in,
                    new_out,
                    self._total_in,
                    self._total_out
                )
