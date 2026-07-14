"""A/B benchmark: full ensemble every frame vs interleaved round-robin."""

import time

import cv2

from app.config import settings
from app.detection import DetectionEngine

engine = DetectionEngine()
cap = cv2.VideoCapture("test_clip.mp4")
frames = []
for _ in range(20):
    ret, frame = cap.read()
    if not ret:
        break
    frames.append(frame)
cap.release()

# warm-up
engine.detect(frames[0])

for mode in (False, True):
    settings.ensemble_interleave = mode
    start = time.time()
    for frame in frames:
        engine.detect(frame)
    per_frame = (time.time() - start) / len(frames) * 1000
    label = "interleaved" if mode else "full ensemble"
    print(f"{label:14s}: {per_frame:6.0f} ms/frame  ({1000/per_frame:.1f} fps)")
