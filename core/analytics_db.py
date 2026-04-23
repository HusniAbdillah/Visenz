"""
SQLite-backed analytics storage for edge-vision-counter.

Stores crossing events as time-series rows and keeps a small app_state table
for the current running totals and session reset timestamp.
"""

from __future__ import annotations

import logging
import sqlite3
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)


class AnalyticsDB:
    """Small helper around SQLite for event journaling and aggregate queries."""

    def __init__(self, db_path: Path) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_schema()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self._db_path, timeout=10, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _initialize_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    camera_id TEXT NOT NULL,
                    direction TEXT NOT NULL CHECK(direction IN ('IN', 'OUT'))
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_logs_timestamp ON logs(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_logs_camera_timestamp ON logs(camera_id, timestamp)")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS app_state (
                    id INTEGER PRIMARY KEY CHECK(id = 1),
                    total_in INTEGER NOT NULL DEFAULT 0,
                    total_out INTEGER NOT NULL DEFAULT 0,
                    last_reset_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            row = conn.execute("SELECT id FROM app_state WHERE id = 1").fetchone()
            if row is None:
                now = self._utc_now()
                conn.execute(
                    """
                    INSERT OR REPLACE INTO app_state (id, total_in, total_out, last_reset_at, updated_at)
                    VALUES (1, 0, 0, ?, ?)
                    """,
                    (now, now),
                )

    @staticmethod
    def _utc_now() -> str:
        return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

    def load_state(self) -> Dict[str, Any]:
        with self._connect() as conn:
            row = conn.execute("SELECT total_in, total_out, last_reset_at FROM app_state WHERE id = 1").fetchone()
            if row is None:
                return {
                    "total_in": 0,
                    "total_out": 0,
                    "current_inside": 0,
                    "last_reset_at": self._utc_now(),
                }

            total_in = int(row["total_in"])
            total_out = int(row["total_out"])
            return {
                "total_in": total_in,
                "total_out": total_out,
                "current_inside": total_in - total_out,
                "last_reset_at": row["last_reset_at"],
            }

    def save_state(self, total_in: int, total_out: int, last_reset_at: str) -> None:
        updated_at = self._utc_now()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO app_state (id, total_in, total_out, last_reset_at, updated_at)
                VALUES (1, ?, ?, ?, ?)
                """,
                (int(total_in), int(total_out), last_reset_at, updated_at),
            )

    def record_event(self, timestamp: str, camera_id: str, direction: str) -> None:
        direction = direction.upper()
        if direction not in {"IN", "OUT"}:
            raise ValueError(f"Invalid direction: {direction}")

        with self._connect() as conn:
            conn.execute(
                "INSERT INTO logs (timestamp, camera_id, direction) VALUES (?, ?, ?)",
                (timestamp, camera_id, direction),
            )

    def record_events(self, events: Sequence[Tuple[str, str, str]]) -> None:
        if not events:
            return

        normalized = []
        for timestamp, camera_id, direction in events:
            direction = direction.upper()
            if direction not in {"IN", "OUT"}:
                raise ValueError(f"Invalid direction: {direction}")
            normalized.append((timestamp, camera_id, direction))

        with self._connect() as conn:
            conn.executemany(
                "INSERT INTO logs (timestamp, camera_id, direction) VALUES (?, ?, ?)",
                normalized,
            )

    def fetch_time_series(self, start_at: str, interval_minutes: int = 15) -> List[Dict[str, Any]]:
        interval_minutes = max(1, int(interval_minutes))
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT timestamp, direction
                FROM logs
                WHERE timestamp >= ?
                ORDER BY timestamp ASC
                """,
                (start_at,),
            ).fetchall()

        buckets: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"total_in": 0, "total_out": 0})

        for row in rows:
            bucket = self._bucket_timestamp(row["timestamp"], interval_minutes)
            bucket_data = buckets[bucket]
            if row["direction"] == "IN":
                bucket_data["total_in"] += 1
            else:
                bucket_data["total_out"] += 1

        ordered: List[Dict[str, Any]] = []
        for bucket in sorted(buckets.keys()):
            item = buckets[bucket]
            ordered.append(
                {
                    "interval_start": bucket,
                    "total_in": item["total_in"],
                    "total_out": item["total_out"],
                    "net_in": item["total_in"] - item["total_out"],
                }
            )
        return ordered

    def fetch_camera_summary(self, start_at: str) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT camera_id,
                       SUM(CASE WHEN direction = 'IN' THEN 1 ELSE 0 END) AS total_in,
                       SUM(CASE WHEN direction = 'OUT' THEN 1 ELSE 0 END) AS total_out,
                       COUNT(*) AS total_events,
                       MAX(timestamp) AS last_event_at
                FROM logs
                WHERE timestamp >= ?
                GROUP BY camera_id
                ORDER BY camera_id ASC
                """,
                (start_at,),
            ).fetchall()

        return [
            {
                "camera_id": row["camera_id"],
                "total_in": int(row["total_in"] or 0),
                "total_out": int(row["total_out"] or 0),
                "total_events": int(row["total_events"] or 0),
                "last_event_at": row["last_event_at"],
            }
            for row in rows
        ]

    def _bucket_timestamp(self, timestamp: str, interval_minutes: int) -> str:
        parsed = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
        bucket_minute = (parsed.minute // interval_minutes) * interval_minutes
        bucket = parsed.replace(minute=bucket_minute, second=0, microsecond=0)
        return bucket.strftime("%Y-%m-%d %H:%M:%S")
