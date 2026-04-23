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
import pytz

logger = logging.getLogger(__name__)
WIB_TZ = pytz.timezone("Asia/Jakarta")


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
                    timestamp_wib TEXT NOT NULL,
                    camera_id TEXT NOT NULL,
                    direction TEXT NOT NULL CHECK(direction IN ('IN', 'OUT')),
                    session_id TEXT NOT NULL
                )
                """
            )

            # Lightweight migration path for earlier schema versions.
            log_columns = {row["name"] for row in conn.execute("PRAGMA table_info(logs)").fetchall()}
            if "timestamp_wib" not in log_columns:
                conn.execute("ALTER TABLE logs ADD COLUMN timestamp_wib TEXT")
                if "timestamp" in log_columns:
                    conn.execute("UPDATE logs SET timestamp_wib = timestamp WHERE timestamp_wib IS NULL")
                conn.execute("UPDATE logs SET timestamp_wib = ? WHERE timestamp_wib IS NULL", (self._wib_now(),))
            if "session_id" not in log_columns:
                conn.execute("ALTER TABLE logs ADD COLUMN session_id TEXT")
                conn.execute("UPDATE logs SET session_id = 'legacy' WHERE session_id IS NULL OR session_id = ''")

            conn.execute("CREATE INDEX IF NOT EXISTS idx_logs_timestamp_wib ON logs(timestamp_wib)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_logs_camera_timestamp_wib ON logs(camera_id, timestamp_wib)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_logs_session_timestamp_wib ON logs(session_id, timestamp_wib)")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS app_state (
                    id INTEGER PRIMARY KEY CHECK(id = 1),
                    total_in INTEGER NOT NULL DEFAULT 0,
                    total_out INTEGER NOT NULL DEFAULT 0,
                    session_id TEXT NOT NULL,
                    last_reset_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )

            state_columns = {row["name"] for row in conn.execute("PRAGMA table_info(app_state)").fetchall()}
            if "session_id" not in state_columns:
                conn.execute("ALTER TABLE app_state ADD COLUMN session_id TEXT")
                conn.execute("UPDATE app_state SET session_id = 'legacy' WHERE session_id IS NULL OR session_id = ''")

            row = conn.execute("SELECT id FROM app_state WHERE id = 1").fetchone()
            if row is None:
                now = self._wib_now()
                conn.execute(
                    """
                    INSERT OR REPLACE INTO app_state (id, total_in, total_out, session_id, last_reset_at, updated_at)
                    VALUES (1, 0, 0, ?, ?, ?)
                    """,
                    ("legacy", now, now),
                )

    @staticmethod
    def _wib_now() -> str:
        return datetime.now(WIB_TZ).strftime("%Y-%m-%d %H:%M:%S")

    def load_state(self) -> Dict[str, Any]:
        with self._connect() as conn:
            row = conn.execute("SELECT total_in, total_out, session_id, last_reset_at FROM app_state WHERE id = 1").fetchone()
            if row is None:
                return {
                    "total_in": 0,
                    "total_out": 0,
                    "current_inside": 0,
                    "session_id": "legacy",
                    "last_reset_at": self._wib_now(),
                }

            total_in = int(row["total_in"])
            total_out = int(row["total_out"])
            return {
                "total_in": total_in,
                "total_out": total_out,
                "current_inside": total_in - total_out,
                "session_id": row["session_id"] or "legacy",
                "last_reset_at": row["last_reset_at"],
            }

    def save_state(self, total_in: int, total_out: int, session_id: str, last_reset_at: str) -> None:
        updated_at = self._wib_now()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO app_state (id, total_in, total_out, session_id, last_reset_at, updated_at)
                VALUES (1, ?, ?, ?, ?, ?)
                """,
                (int(total_in), int(total_out), str(session_id), last_reset_at, updated_at),
            )

    def record_event(self, timestamp_wib: str, camera_id: str, direction: str, session_id: str) -> None:
        direction = direction.upper()
        if direction not in {"IN", "OUT"}:
            raise ValueError(f"Invalid direction: {direction}")

        with self._connect() as conn:
            conn.execute(
                "INSERT INTO logs (timestamp_wib, camera_id, direction, session_id) VALUES (?, ?, ?, ?)",
                (timestamp_wib, camera_id, direction, session_id),
            )

    def record_events(self, events: Sequence[Tuple[str, str, str, str]]) -> None:
        if not events:
            return

        normalized = []
        for timestamp_wib, camera_id, direction, session_id in events:
            direction = direction.upper()
            if direction not in {"IN", "OUT"}:
                raise ValueError(f"Invalid direction: {direction}")
            normalized.append((timestamp_wib, camera_id, direction, str(session_id)))

        with self._connect() as conn:
            conn.executemany(
                "INSERT INTO logs (timestamp_wib, camera_id, direction, session_id) VALUES (?, ?, ?, ?)",
                normalized,
            )

    def fetch_time_series(self, start_at: str, session_id: str, interval_minutes: int = 15) -> List[Dict[str, Any]]:
        interval_minutes = max(1, int(interval_minutes))
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT timestamp_wib, direction
                FROM logs
                WHERE timestamp_wib >= ?
                  AND session_id = ?
                ORDER BY timestamp_wib ASC
                """,
                (start_at, session_id),
            ).fetchall()

        buckets: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"total_in": 0, "total_out": 0})

        for row in rows:
            bucket = self._bucket_timestamp(row["timestamp_wib"], interval_minutes)
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

    def fetch_camera_summary(self, start_at: str, session_id: str) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT camera_id,
                       SUM(CASE WHEN direction = 'IN' THEN 1 ELSE 0 END) AS total_in,
                       SUM(CASE WHEN direction = 'OUT' THEN 1 ELSE 0 END) AS total_out,
                       COUNT(*) AS total_events,
                       MAX(timestamp_wib) AS last_event_at
                FROM logs
                WHERE timestamp_wib >= ?
                  AND session_id = ?
                GROUP BY camera_id
                ORDER BY camera_id ASC
                """,
                (start_at, session_id),
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

    def fetch_logs(self, session_id: Optional[str] = None) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            if session_id:
                rows = conn.execute(
                    """
                    SELECT id, timestamp_wib, camera_id, direction, session_id
                    FROM logs
                    WHERE session_id = ?
                    ORDER BY timestamp_wib ASC, id ASC
                    """,
                    (session_id,),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT id, timestamp_wib, camera_id, direction, session_id
                    FROM logs
                    ORDER BY timestamp_wib ASC, id ASC
                    """
                ).fetchall()

        return [dict(row) for row in rows]

    def _bucket_timestamp(self, timestamp: str, interval_minutes: int) -> str:
        parsed = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
        bucket_minute = (parsed.minute // interval_minutes) * interval_minutes
        bucket = parsed.replace(minute=bucket_minute, second=0, microsecond=0)
        return bucket.strftime("%Y-%m-%d %H:%M:%S")
