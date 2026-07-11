"""Alert engine: dedupe, persistence, snapshots, and notification fan-out."""

import json
import sqlite3
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2

from .config import DB_PATH, SNAPSHOT_DIR, settings
from .threat import Threat


class Notifier:
    """Base notifier. Real integrations (SMS, push, call) plug in here."""

    def send(self, alert: dict) -> None:
        raise NotImplementedError


class ConsoleNotifier(Notifier):
    def send(self, alert: dict) -> None:
        print(f"[ALERT:{alert['level']}] {alert['message']}")


class WebhookNotifier(Notifier):
    """Placeholder: POST alerts to a webhook (e.g. Twilio/FCM bridge)."""

    def __init__(self, url: str):
        self.url = url

    def send(self, alert: dict) -> None:
        # Prototype stub — wire to requests.post(self.url, json=alert)
        pass


class AlertEngine:
    def __init__(self):
        SNAPSHOT_DIR.mkdir(exist_ok=True)
        self._lock = threading.Lock()
        self._last_fired: dict[tuple, float] = {}  # (level, kind) -> ts
        self.notifiers: list[Notifier] = [ConsoleNotifier()]
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(DB_PATH) as db:
            db.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts TEXT NOT NULL,
                    level TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    message TEXT NOT NULL,
                    snapshot TEXT
                )
            """)

    def process(self, threats: list[Threat], annotated_frame) -> list[dict]:
        """Fire alerts for threats not in cooldown. Returns fired alerts."""
        fired = []
        now = time.time()

        for threat in threats:
            key = (threat.level, threat.kind)
            with self._lock:
                last = self._last_fired.get(key, 0)
                if now - last < settings.alert_cooldown_seconds:
                    continue
                self._last_fired[key] = now

            snapshot_name = None
            if annotated_frame is not None:
                snapshot_name = (
                    f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    f"_{threat.level}_{threat.kind}.jpg"
                )
                cv2.imwrite(str(SNAPSHOT_DIR / snapshot_name), annotated_frame)

            alert = {
                "ts": datetime.now().isoformat(timespec="seconds"),
                "level": threat.level,
                "kind": threat.kind,
                "message": threat.message,
                "snapshot": snapshot_name,
            }

            with sqlite3.connect(DB_PATH) as db:
                db.execute(
                    "INSERT INTO events (ts, level, kind, message, snapshot) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (alert["ts"], alert["level"], alert["kind"],
                     alert["message"], alert["snapshot"]),
                )

            for notifier in self.notifiers:
                try:
                    notifier.send(alert)
                except Exception as e:
                    print(f"Notifier failed: {e}")

            fired.append(alert)

        return fired

    def recent_events(self, limit: int = 50) -> list[dict]:
        with sqlite3.connect(DB_PATH) as db:
            db.row_factory = sqlite3.Row
            rows = db.execute(
                "SELECT ts, level, kind, message, snapshot FROM events "
                "ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]
