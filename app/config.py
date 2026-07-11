"""Central configuration for the AI CCTV Threat Detection prototype."""

from dataclasses import dataclass, field
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

WEAPON_MODEL_PATH = str(
    BASE_DIR
    / "Weapons-and-Knives-Detector-with-YOLOv8-main"
    / "runs" / "detect" / "Normal" / "weights" / "best.pt"
)

# COCO-pretrained model for people / vehicles / bags (downloads on first run)
GENERAL_MODEL_PATH = "yolov8n.pt"

SNAPSHOT_DIR = BASE_DIR / "snapshots"
DB_PATH = BASE_DIR / "events.db"

# COCO class ids we care about
PERSON_CLASSES = {"person"}
VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle", "bicycle"}
PACKAGE_CLASSES = {"backpack", "handbag", "suitcase"}


@dataclass
class Settings:
    # Detection
    weapon_confidence: float = 0.45
    general_confidence: float = 0.40
    inference_width: int = 640

    # Threat rules (seconds)
    loiter_seconds: float = 15.0        # person mostly stationary this long -> LOW
    vehicle_lurk_seconds: float = 8.0   # person lingering beside a vehicle -> MEDIUM
    weapon_confirm_frames: int = 3      # consecutive weapon frames before HIGH
    weapon_critical_seconds: float = 5.0  # weapon persisting this long -> CRITICAL

    # Alert engine
    alert_cooldown_seconds: float = 30.0  # per (level, kind) dedupe window

    # Restricted zone as fractions of frame (x1, y1, x2, y2); None = disabled
    restricted_zone: tuple | None = None

    # Night hours (for "unusual hours" context on alerts)
    night_start_hour: int = 22
    night_end_hour: int = 5


settings = Settings()
