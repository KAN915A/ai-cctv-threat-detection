"""Detection engine: runs a general COCO model + a custom weapons model."""

import time
from dataclasses import dataclass, field

from ultralytics import YOLO

from .config import (
    GENERAL_MODEL_PATH,
    PACKAGE_CLASSES,
    PERSON_CLASSES,
    VEHICLE_CLASSES,
    WEAPON_MODEL_PATH,
    settings,
)


@dataclass
class Detection:
    label: str
    kind: str          # person | vehicle | package | weapon | other
    confidence: float
    box: tuple         # x1, y1, x2, y2 in frame pixels
    track_id: int | None = None
    meta: dict = field(default_factory=dict)

    @property
    def center(self):
        x1, y1, x2, y2 = self.box
        return ((x1 + x2) / 2, (y1 + y2) / 2)


def _kind_for(label: str) -> str:
    l = label.lower()
    if l in PERSON_CLASSES:
        return "person"
    if l in VEHICLE_CLASSES:
        return "vehicle"
    if l in PACKAGE_CLASSES:
        return "package"
    return "other"


class DetectionEngine:
    def __init__(self):
        print("Loading general model (people/vehicles)...")
        self.general = YOLO(GENERAL_MODEL_PATH)
        print("Loading weapons model...")
        self.weapons = YOLO(WEAPON_MODEL_PATH)
        self.weapon_labels = {
            name for name in self.weapons.names.values()
        }
        print(f"Weapon classes: {self.weapon_labels}")

    def detect(self, frame) -> tuple[list[Detection], float]:
        """Run both models on a frame. Returns (detections, inference_ms)."""
        start = time.time()
        detections: list[Detection] = []

        general_results = self.general(
            frame, conf=settings.general_confidence, verbose=False
        )
        for result in general_results:
            for box in result.boxes:
                label = self.general.names[int(box.cls[0])]
                kind = _kind_for(label)
                if kind == "other":
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detections.append(Detection(
                    label=label, kind=kind,
                    confidence=float(box.conf[0]), box=(x1, y1, x2, y2),
                ))

        weapon_results = self.weapons(
            frame, conf=settings.weapon_confidence, verbose=False
        )
        for result in weapon_results:
            for box in result.boxes:
                label = self.weapons.names[int(box.cls[0])]
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detections.append(Detection(
                    label=label, kind="weapon",
                    confidence=float(box.conf[0]), box=(x1, y1, x2, y2),
                ))

        elapsed_ms = (time.time() - start) * 1000
        return detections, elapsed_ms
