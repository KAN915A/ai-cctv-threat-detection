"""Compare old weapon rule (conf >= 0.45, no checks) vs new fusion logic.

Negatives: real false-positive snapshots from a live session + bus.jpg.
Positives: validation mosaics from the weapons model training run (real
guns/knives) + the repo's example image.
"""

import glob

import cv2

from app.detection import DetectionEngine, fuse_weapons

NEGATIVES = sorted(
    glob.glob("snapshots/*HIGH_weapon.jpg")
    + glob.glob("snapshots/*CRITICAL*.jpg")
)[:12] + ["bus.jpg"]

POSITIVES = [
    "Weapons-and-Knives-Detector-with-YOLOv8-main/runs/detect/Normal/val_batch0_labels.jpg",
    "Weapons-and-Knives-Detector-with-YOLOv8-main/runs/detect/Normal/val_batch1_labels.jpg",
    "Weapons-and-Knives-Detector-with-YOLOv8-main/runs/detect/Normal/val_batch2_labels.jpg",
    "Weapons-and-Knives-Detector-with-YOLOv8-main/Results/teste.jpg",
]

engine = DetectionEngine()


def run(path):
    frame = cv2.imread(path)
    if frame is None:
        return None
    general = engine._run_general(frame)
    candidates = engine._run_weapons(frame, all_models=True)
    old = [c for c in candidates if c.confidence >= 0.45]
    new = fuse_weapons(candidates, general, frame.shape)
    return old, new


def report(title, paths, want_weapons):
    print(f"\n=== {title} ===")
    old_hits = new_hits = 0
    for p in paths:
        result = run(p)
        if result is None:
            continue
        old, new = result
        old_hits += bool(old)
        new_hits += bool(new)
        name = p.split("\\")[-1].split("/")[-1]
        fmt = lambda dets: ", ".join(
            f"{d.label} {d.confidence:.0%}"
            + (f" [{d.meta.get('basis')}]" if d.meta.get('basis') else "")
            for d in dets) or "-"
        print(f"{name:48s} old: {fmt(old):40s} new: {fmt(new)}")
    n = len(paths)
    kind = "with weapons flagged" if not want_weapons else "still detected"
    print(f"--> old: {old_hits}/{n} {kind} | new: {new_hits}/{n} {kind}")


report("NEGATIVES (weapon alerts should disappear)", NEGATIVES, False)
report("POSITIVES (real weapons should survive)", POSITIVES, True)
