"""Central configuration for the AI CCTV Threat Detection prototype."""

from dataclasses import dataclass, field
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# Weapons ensemble: the two strongest variants from the trained runs
# (Normal mAP50 0.868, Normal_Compressed 0.860). A detection that both
# models agree on is far more trustworthy than either alone.
_RUNS = BASE_DIR / "Weapons-and-Knives-Detector-with-YOLOv8-main" / "runs" / "detect"
WEAPON_MODEL_PATHS = [
    str(_RUNS / "Normal" / "weights" / "best.pt"),
    str(_RUNS / "Normal_Compressed" / "weights" / "best.pt"),
]

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

# COCO objects that aren't weapons but are worth a MEDIUM alert when
# someone is carrying them
DANGEROUS_OBJECTS = {"baseball bat", "scissors"}


@dataclass
class Settings:
    # Detection
    general_confidence: float = 0.40
    inference_width: int = 640

    # Weapon fusion: candidates below these bars are rejected as noise.
    weapon_candidate_conf: float = 0.30   # raw model threshold (candidates)
    weapon_conf_near_person: float = 0.55  # weapon overlapping a person
    weapon_conf_ensemble: float = 0.45     # both weapons models agree
    weapon_conf_agree: float = 0.40        # custom + COCO 'knife' agree
    weapon_conf_alone: float = 0.70        # no person anywhere near
    weapon_max_area_frac: float = 0.30     # bigger than this = misfire
    weapon_person_iou_veto: float = 0.50   # box ≈ person box = misfire
    distractor_iou: float = 0.40
    distractor_min_conf: float = 0.40
    ensemble_iou: float = 0.55             # same-object match across models

    # Altercation heuristic: two people moving fast in close quarters
    fight_speed_px: float = 140.0          # px/s each person must exceed
    fight_window: int = 6                  # frames in the vote window
    fight_votes: int = 3                   # frames that must look like a fight

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
