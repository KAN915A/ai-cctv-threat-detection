"""Detection engine: COCO context model + an ensemble of weapons models.

The weapons models alone false-positive heavily (laptop screens, pens,
whole-room boxes). Two defenses:
  * ensemble agreement — the same box found by both weapons models is far
    more trustworthy than either alone;
  * ``fuse_weapons`` — cross-checks every candidate against the COCO model
    and scene geometry before it is allowed through.
"""

import time
from dataclasses import dataclass, field

from ultralytics import YOLO

from .config import (
    DANGEROUS_OBJECTS,
    DISTRACTOR_CLASSES,
    EXCLUDED_WEAPON_LABELS,
    GENERAL_MODEL_PATH,
    HF_WEAPON_MODEL_URL,
    PACKAGE_CLASSES,
    PERSON_CLASSES,
    VEHICLE_CLASSES,
    WEAPON_LABEL_MAP,
    WEAPON_MODEL_PATHS,
    settings,
)


@dataclass
class Detection:
    label: str
    kind: str          # person | vehicle | package | weapon | danger | other
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
    if l in DANGEROUS_OBJECTS:
        return "danger"
    return "other"


def _iou(a: tuple, b: tuple) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter + 1e-9)


def _near(a: tuple, b: tuple, margin: int = 40) -> bool:
    return not (a[2] + margin < b[0] or b[2] + margin < a[0] or
                a[3] + margin < b[1] or b[3] + margin < a[1])


def merge_candidates(per_model: list[list[Detection]]) -> list[Detection]:
    """Collapse same-label overlapping boxes from different weapons models
    into one detection that remembers how many models agreed."""
    merged: list[Detection] = []
    for model_idx, candidates in enumerate(per_model):
        for cand in candidates:
            match = next(
                (m for m in merged
                 if m.label == cand.label
                 and _iou(m.box, cand.box) > settings.ensemble_iou),
                None)
            if match is not None:
                match.meta["models"].add(model_idx)
                if cand.confidence > match.confidence:
                    match.confidence = cand.confidence
                    match.box = cand.box
            else:
                cand.meta["models"] = {model_idx}
                merged.append(cand)
    return merged


def fuse_weapons(
    candidates: list[Detection],
    general: list[Detection],
    frame_shape: tuple,
) -> list[Detection]:
    """Filter weapon candidates using the COCO model + scene geometry.

    Returns the accepted subset, with ``meta['basis']`` explaining why each
    survived (agrees_coco / ensemble_agree / near_person / high_conf).
    """
    h, w = frame_shape[:2]
    frame_area = float(h * w)

    persons = [d for d in general if d.kind == "person"]
    coco_knives = [d for d in general if d.label.lower() == "knife"]
    distractors = [
        d for d in general
        if d.label.lower() in DISTRACTOR_CLASSES
        and d.confidence >= settings.distractor_min_conf
    ]

    accepted = []
    for cand in candidates:
        x1, y1, x2, y2 = cand.box
        area_frac = ((x2 - x1) * (y2 - y1)) / frame_area

        # Huge boxes are misfires — real handheld weapons are small in frame
        if area_frac > settings.weapon_max_area_frac:
            continue

        # A "weapon" box that coincides with a person box IS the person
        if any(_iou(cand.box, p.box) > settings.weapon_person_iou_veto
               for p in persons):
            continue

        # Confident everyday object in the same spot (laptop, phone, ...)
        if any(_iou(cand.box, d.box) > settings.distractor_iou
               and d.confidence > 0.8 * cand.confidence
               for d in distractors):
            continue

        # Tiered acceptance: use the lowest bar this candidate qualifies for
        bars = [("high_conf", settings.weapon_conf_alone)]
        if any(_near(cand.box, p.box) for p in persons):
            bars.append(("near_person", settings.weapon_conf_near_person))
        if len(cand.meta.get("models", ())) >= 2:
            bars.append(("ensemble_agree", settings.weapon_conf_ensemble))
        if (cand.label.lower() == "knife"
                and any(_iou(cand.box, k.box) > 0.3 for k in coco_knives)):
            bars.append(("agrees_coco", settings.weapon_conf_agree))

        basis, bar = min(bars, key=lambda item: item[1])
        if cand.confidence >= bar:
            cand.meta["basis"] = basis
            accepted.append(cand)

    return accepted


def _ensure_hf_model(path: str) -> bool:
    """Download the HF ensemble member on first run. Returns availability."""
    from pathlib import Path
    p = Path(path)
    if p.exists():
        return True
    try:
        import requests
        print(f"Downloading weapons model from Hugging Face -> {p.name}")
        r = requests.get(HF_WEAPON_MODEL_URL, timeout=120)
        r.raise_for_status()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(r.content)
        return True
    except Exception as e:
        print(f"HF model unavailable ({e}) — continuing without it")
        return False


class DetectionEngine:
    def __init__(self):
        print("Loading general model (people/vehicles)...")
        self.general = YOLO(GENERAL_MODEL_PATH)
        self.weapons = []
        for path in WEAPON_MODEL_PATHS:
            if "threat-detection" in path and not _ensure_hf_model(path):
                continue
            try:
                print(f"Loading weapons model: {path}")
                self.weapons.append(YOLO(path))
            except Exception as e:
                print(f"Skipping weapons model {path}: {e}")
        print(f"Weapons ensemble: {len(self.weapons)} models")
        self._weapon_turn = 0
        self._weapon_cache: dict[int, tuple[float, list[Detection]]] = {}

    def _run_general(self, frame) -> list[Detection]:
        detections = []
        for result in self.general(frame, conf=settings.general_confidence,
                                   verbose=False):
            for box in result.boxes:
                label = self.general.names[int(box.cls[0])]
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detections.append(Detection(
                    label=label, kind=_kind_for(label),
                    confidence=float(box.conf[0]), box=(x1, y1, x2, y2),
                ))
        return detections

    def _detect_one(self, model, frame) -> list[Detection]:
        candidates = []
        for result in model(frame, conf=settings.weapon_candidate_conf,
                            verbose=False):
            for box in result.boxes:
                raw = model.names[int(box.cls[0])].lower()
                if raw in EXCLUDED_WEAPON_LABELS:
                    continue
                label = WEAPON_LABEL_MAP.get(raw, raw)
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                candidates.append(Detection(
                    label=label, kind="weapon",
                    confidence=float(box.conf[0]), box=(x1, y1, x2, y2),
                ))
        return candidates

    def _run_weapons(self, frame, all_models: bool = False) -> list[Detection]:
        """Run the weapons ensemble.

        Live pipeline (interleave on): one model per frame, round-robin,
        merged with the other model's result from the previous frame — the
        agreement check spans adjacent frames at near-2x FPS. Pass
        ``all_models=True`` (evals, single images) to run every model on
        this exact frame.
        """
        interleave = (settings.ensemble_interleave and not all_models
                      and len(self.weapons) > 1)
        now = time.time()
        per_model = []
        for idx, model in enumerate(self.weapons):
            if interleave and idx != self._weapon_turn:
                ts, cached = self._weapon_cache.get(idx, (0.0, []))
                per_model.append(
                    cached if now - ts < settings.ensemble_cache_seconds
                    else [])
                continue
            candidates = self._detect_one(model, frame)
            self._weapon_cache[idx] = (now, candidates)
            per_model.append(candidates)
        if interleave:
            self._weapon_turn = (self._weapon_turn + 1) % len(self.weapons)
        return merge_candidates(per_model)

    def detect(self, frame) -> tuple[list[Detection], float]:
        """Run all models + fusion. Returns (detections, inference_ms)."""
        start = time.time()

        general = self._run_general(frame)
        candidates = self._run_weapons(frame)
        weapons = fuse_weapons(candidates, general, frame.shape)

        detections = [d for d in general if d.kind != "other"] + weapons
        elapsed_ms = (time.time() - start) * 1000
        return detections, elapsed_ms
