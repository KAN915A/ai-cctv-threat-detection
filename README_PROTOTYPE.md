# AI CCTV Threat Detection — Full Edition (Python)

Working prototype of the platform described in
`AI_CCTV_Threat_Detection_Project_Brief.pdf`: live CCTV analysis → threat
classification → real-time alerts on a dashboard. This document covers the
local Python edition; the repo [README](README.md) covers both editions and
the hosted browser demo.

## Run it

```bash
# double-click start_dashboard.bat, or:
.venv\Scripts\python.exe -m uvicorn app.main:app --port 8000
```

Open http://localhost:8000, enter a source, press **Start**:

| Source | Example |
|---|---|
| Webcam | `0` |
| Video file | `test_clip.mp4` |
| IP camera | `rtsp://user:pass@192.168.1.100:554/stream` |

## Architecture (maps to the brief)

```
Camera/RTSP/file ──► Pipeline thread (app/pipeline.py)
                        │  DetectionEngine (app/detection.py)
                        │    ├─ yolov8n.pt COCO   → people, vehicles, packages,
                        │    │                      context objects for fusion
                        │    └─ weapons ensemble  → guns, knives
                        │         Normal (mAP50 .868) + Normal_Compressed (.860),
                        │         cross-model merge in merge_candidates(),
                        │         fusion vetoes + tiered bars in fuse_weapons()
                        │  CentroidTracker (app/tracker.py)
                        │    └─ stable person IDs, dwell time, movement, speed
                        │  ThreatClassifier (app/threat.py)
                        │    └─ rules → LOW / MEDIUM / HIGH / CRITICAL
                        │       (weapon temporal vote, altercation heuristic)
                        │  AlertEngine (app/alerts.py)
                        │    └─ cooldown dedupe, SQLite (events.db),
                        │       snapshots/, notifier fan-out
                        ▼
                 FastAPI (app/main.py)
                   ├─ /video_feed   MJPEG annotated stream
                   ├─ /ws           WebSocket: state + alerts (2 Hz)
                   ├─ /api/*        start/stop/status/events/zone
                   └─ /             dashboard (app/static/index.html)
```

## Alert levels (as specified in the brief)

| Level | Rule in prototype |
|---|---|
| **LOW** | Person mostly stationary ≥ 15 s (loitering) |
| **MEDIUM** | Person beside a vehicle ≥ 8 s · person inside restricted zone · baseball bat/scissors near a person |
| **HIGH** | Weapon accepted by fusion, present in ≥ 5 of last 8 frames · two people moving > 140 px/s in close quarters (3-of-6 vote) |
| **CRITICAL** | Weapon persisting in scene ≥ 5 s |

Night hours (22:00–05:00) are annotated on alerts. All thresholds live in
`app/config.py`.

## Weapon false-positive defenses

The raw weapons models flag laptop screens, pens, and whole-room boxes.
Candidates only count when they survive, in order:

1. size veto (box > 30 % of frame area)
2. person-coincidence veto (IoU with a person box > 0.5)
3. distractor veto (confident COCO laptop/phone/remote/… in the same spot)
4. tiered confidence — lowest applicable bar: COCO-knife agreement 0.40,
   ensemble agreement 0.45, near a person 0.55, alone 0.70
5. temporal vote — weapon in ≥ 5 of the last 8 frames before HIGH fires

## Verified end-to-end

- Weapons demo video (real knives): 13/13 sampled frames detected via
  ensemble agreement (`eval_video.py`).
- 13 real false-positive snapshots from a live session + bus street scene:
  0 weapon alerts (`eval_accuracy.py`).
- Live webcam: person tracking, loitering LOW, knife HIGH → CRITICAL
  escalation, snapshot per alert.
- Test clip: vehicle detection, MEDIUM vehicle-lurking at 8 s, LOW loitering
  (`verify_medium.py`).
- ~2.5–3 FPS on CPU with all three models (~8 FPS with a single weapons
  model — trim `WEAPON_MODEL_PATHS` in `app/config.py` if speed matters
  more than the ensemble).

## Known prototype limitations

- Single camera, single dashboard viewer (alerts are consumed by the first
  reader of `/api/status` or the WebSocket).
- Weapon recall is limited by the training data: rifles at distance and
  heavily occluded weapons are often missed; pen-like objects held close to
  the camera can still sneak past the filters. The real fix is fine-tuning
  with rifle images and hard negatives.
- SQLite instead of PostgreSQL; notifiers are stubs (`WebhookNotifier` is the
  hook for Twilio SMS / FCM push).
- Not yet implemented from the brief: pose-based fighting recognition (the
  altercation rule is motion-heuristic only), package-theft logic,
  masked-person detection, license-plate OCR, mobile app.
