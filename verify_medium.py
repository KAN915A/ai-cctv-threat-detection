"""Offline check of the MEDIUM threat path (person lingering near vehicle)
using test_clip.mp4, without touching the running server."""

import time

import cv2

from app.detection import DetectionEngine
from app.threat import ThreatClassifier
from app.tracker import CentroidTracker

engine = DetectionEngine()
tracker = CentroidTracker()
classifier = ThreatClassifier()

cap = cv2.VideoCapture("test_clip.mp4")
seen_levels = {}
start = time.time()

while time.time() - start < 25:
    ret, frame = cap.read()
    if not ret:
        break
    detections, _ = engine.detect(frame)
    persons = [d for d in detections if d.kind == "person"]
    tracks = tracker.update(persons)
    threats = classifier.classify(detections, tracks, frame.shape)
    for t in threats:
        if (t.level, t.kind) not in seen_levels:
            seen_levels[(t.level, t.kind)] = t.message
            print(f"[{time.time()-start:5.1f}s] {t.level}: {t.message}")

cap.release()
kinds = {d.kind for d in detections}
print(f"\nFinal frame object kinds: {kinds}")
print(f"Threat kinds seen: {sorted(seen_levels)}")
