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


# COCO objects that commonly get mistaken for weapons: if one of these
# confidently overlaps a weapon box, the weapon detection is rejected
DISTRACTOR_CLASSES = {
    "cell phone", "remote", "laptop", "tv", "keyboard", "mouse", "book",
    "bottle", "cup", "umbrella", "toothbrush", "hair drier", "clock",
    "teddy bear", "banana",
}


@dataclass
class Settings:
    # Detection
    general_confidence: float = 0.40
    inference_width: int = 640

    # Weapon fusion: candidates below these bars are rejected as noise.
    weapon_candidate_conf: float = 0.30   # raw model threshold (candidates)
    weapon_conf_near_person: float = 0.55  # weapon overlapping a person
    weapon_conf_agree: float = 0.40        # custom + COCO 'knife' agree
    weapon_conf_alone: float = 0.70        # no person anywhere near
    weapon_max_area_frac: float = 0.30     # bigger than this = misfire
    weapon_person_iou_veto: float = 0.50   # box ≈ person box = misfire
    distractor_iou: float = 0.40
    distractor_min_conf: float = 0.40

    # Threat rules (seconds)
    loiter_seconds: float = 15.0        # person mostly stationary this long -> LOW
    vehicle_lurk_seconds: float = 8.0   # person lingering beside a vehicle -> MEDIUM
    # Temporal vote: weapon must appear in >= weapon_votes of the last
    # weapon_window frames before HIGH fires (kills one-frame flickers)
    weapon_window: int = 8
    weapon_votes: int = 5
    weapon_critical_seconds: float = 5.0  # weapon persisting this long -> CRITICAL

    # Alert engine
    alert_cooldown_seconds: float = 30.0  # per (level, kind) dedupe window

    # Restricted zone as fractions of frame (x1, y1, x2, y2); None = disabled
    restricted_zone: tuple | None = None

    # Night hours (for "unusual hours" context on alerts)
    night_start_hour: int = 22
    night_end_hour: int = 5


settings = Settings()
