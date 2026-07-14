"""Sample frames from the weapons repo demo video (real weapons on camera)
and check old rule vs fusion on each."""

import cv2

from app.detection import DetectionEngine, fuse_weapons

VIDEO = "Weapons-and-Knives-Detector-with-YOLOv8-main/Results/detected_objects_video.mp4"

engine = DetectionEngine()
cap = cv2.VideoCapture(VIDEO)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"video frames: {total}")

old_hits = new_hits = n = 0
for i in range(0, total, max(1, total // 12)):
    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
    ret, frame = cap.read()
    if not ret:
        continue
    n += 1
    general = engine._run_general(frame)
    candidates = engine._run_weapons(frame, all_models=True)
    old = [c for c in candidates if c.confidence >= 0.45]
    new = fuse_weapons(candidates, general, frame.shape)
    old_hits += bool(old)
    new_hits += bool(new)
    fmt = lambda dets: ", ".join(
        f"{d.label} {d.confidence:.0%}"
        + (f"[{d.meta.get('basis','')}]" if d.meta.get('basis') else "")
        for d in dets) or "-"
    print(f"frame {i:5d}  old: {fmt(old):45s} new: {fmt(new)}")

cap.release()
print(f"\n--> weapon found: old {old_hits}/{n} frames, new {new_hits}/{n} frames")
