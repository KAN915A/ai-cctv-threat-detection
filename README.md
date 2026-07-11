# AI CCTV Threat Monitor

AI-powered security monitoring prototype: analyzes CCTV/webcam video in real
time with a **multi-model detection ensemble**, classifies threats into four
alert levels, and raises alerts with snapshots — based on the *AI CCTV Threat
Detection System* project brief.

**Live demo:** https://ai-cctv-threat-detection.vercel.app — runs entirely in
your browser, no video ever leaves your device.

Two editions share the same detection models and threat rules:

| | **Browser demo** (`web/`) | **Full edition** (`app/`) |
|---|---|---|
| Runs on | Vercel / any static host | Local machine (Python) |
| Inference | ONNX Runtime Web (WebGPU/WASM), in-browser | PyTorch + OpenCV, server-side |
| Sources | Webcam, image upload | Webcam, video files, RTSP IP cameras |
| Alerts | Local feed + browser storage | SQLite history, snapshots, notifier hooks (SMS/push) |
| Privacy | Video never leaves the device | Video never leaves your network |

## Detection pipeline (3 models + fusion)

Every frame passes through **three YOLOv8 models**:

1. **COCO context model** (`yolov8n`) — people, vehicles, packages, plus
   context objects used for cross-checking.
2. **Weapons ensemble** — the two strongest custom-trained guns/knife
   detectors from the training runs (Normal, mAP50 0.868 and
   Normal_Compressed, mAP50 0.860). Same-label boxes that overlap across
   models are merged and remember how many models agreed.

Raw weapons models false-positive heavily (laptop screens, pens, whole-room
boxes), so every weapon candidate must survive a **fusion filter**:

- **Size veto** — boxes covering > 30 % of the frame are misfires
- **Person-coincidence veto** — a "weapon" box that ≈ equals a person box *is* the person
- **Distractor veto** — a confident everyday COCO object (laptop, phone,
  remote, book…) in the same spot rejects the candidate
- **Tiered confidence** — the lowest bar the candidate qualifies for:
  COCO-knife agreement 0.40 · ensemble agreement 0.45 · near a person 0.55 ·
  alone in frame 0.70

A **temporal vote** (weapon present in ≥ 5 of the last 8 frames) then
suppresses single-frame flickers on live video; uploaded still images bypass
the vote. A centroid tracker adds per-person identity, dwell time, and speed.

## Threat levels

| Level | Rules |
|---|---|
| **LOW** | Person mostly stationary ≥ 15 s (loitering) |
| **MEDIUM** | Person beside a vehicle ≥ 8 s · person inside a restricted zone · suspicious object (baseball bat, scissors) near a person |
| **HIGH** | Weapon confirmed by fusion + temporal vote · possible altercation (two people moving > 140 px/s in close quarters, 3-of-6 frame vote) |
| **CRITICAL** | Weapon persisting in scene ≥ 5 s |

Measured results: on the weapons demo video (real knives held on camera) the
ensemble keeps **13/13 sampled frames** detected via ensemble agreement; on
13 real false-positive snapshots from a live webcam session, **0** weapon
alerts fire. Regression assets: `web/test_weapon.jpg`, `eval_video.py`,
`eval_accuracy.py`.

## Browser demo

Deployed on Vercel as a static site; the three ONNX models (~36 MB total,
cached after first visit) load via CDN and run client-side (WebGPU when
available, ~300 ms/frame).

```bash
# local preview
python -m http.server 8090 --directory web
```

## Full edition (FastAPI dashboard)

```bash
python -m venv .venv
.venv\Scripts\pip install ultralytics opencv-python fastapi "uvicorn[standard]"
.venv\Scripts\python -m uvicorn app.main:app --port 8000
```

Open http://localhost:8000 and enter a source: `0` (webcam), a video file
path, or an `rtsp://` camera URL. Architecture and tuning knobs:
[README_PROTOTYPE.md](README_PROTOTYPE.md); all thresholds live in
`app/config.py`. Running three models on CPU yields ~2.5–3 FPS; trim
`WEAPON_MODEL_PATHS` to one model for ~2× speed.

## Repo layout

```
app/          FastAPI + pipeline (detection ensemble, tracking, threat rules, alerts)
web/          static browser edition + ONNX models (served to Vercel via CDN)
Weapons-and-Knives-Detector-with-YOLOv8-main/
              trained weapons weights (only weights + attribution committed)
eval_*.py     accuracy evaluation harnesses (positives + negatives)
```

## Licensing & attribution

- Weapons model weights from
  [Weapons-and-Knives-Detector-with-YOLOv8](https://github.com/JoaoAssalim/Weapons-and-Knives-Detector-with-YOLOv8).
- Built with [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) (AGPL-3.0).
- This repository is licensed **AGPL-3.0** (see [LICENSE](LICENSE)).

Prototype for education/research. Weapon detection has false positives and
misses — do not rely on it for real security decisions. Check local laws
before pointing cameras at public spaces or using face recognition.
