"""
database/logger.py  —  Detection Logging
=========================================
Menyimpan setiap deteksi stimming ke SQLite dan menyediakan
query untuk dashboard monitoring.
"""

import sqlite3
import json
from datetime import datetime
from typing import List, Dict


CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS detections (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp   TEXT    NOT NULL,
    label       TEXT    NOT NULL,
    confidence  REAL    NOT NULL,
    all_probs   TEXT    NOT NULL,
    session_id  TEXT
);
"""

CREATE_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_timestamp ON detections (timestamp);
CREATE INDEX IF NOT EXISTS idx_label     ON detections (label);
"""


class DetectionLogger:
    def __init__(self, db_path: str = "detections.db"):
        self._db_path = db_path
        self._conn    = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(CREATE_TABLE_SQL + CREATE_INDEX_SQL)
        self._conn.commit()

    def log(self, result: Dict, session_id: str = None):
        """Simpan satu deteksi stimming."""
        self._conn.execute(
            """INSERT INTO detections
               (timestamp, label, confidence, all_probs, session_id)
               VALUES (?, ?, ?, ?, ?)""",
            (
                datetime.now().isoformat(),
                result["label"],
                result["confidence"],
                json.dumps(result.get("all_probs", {})),
                session_id,
            ),
        )
        self._conn.commit()

    def get_recent(self, limit: int = 100) -> List[Dict]:
        """Ambil N deteksi terbaru."""
        rows = self._conn.execute(
            "SELECT * FROM detections ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_stats(self) -> Dict:
        """Statistik agregat: jumlah per label."""
        rows = self._conn.execute(
            """SELECT label, COUNT(*) as count, AVG(confidence) as avg_conf
               FROM detections GROUP BY label"""
        ).fetchall()
        return {
            r["label"]: {
                "count":    r["count"],
                "avg_conf": round(r["avg_conf"], 3),
            }
            for r in rows
        }

    def get_timeline(self, hours: int = 24) -> List[Dict]:
        """Deteksi per jam dalam N jam terakhir."""
        rows = self._conn.execute(
            """SELECT strftime('%Y-%m-%dT%H:00:00', timestamp) as hour,
                      label,
                      COUNT(*) as count
               FROM detections
               WHERE timestamp >= datetime('now', ?)
               GROUP BY hour, label
               ORDER BY hour""",
            (f"-{hours} hours",),
        ).fetchall()
        return [dict(r) for r in rows]

    def close(self):
        self._conn.close()
