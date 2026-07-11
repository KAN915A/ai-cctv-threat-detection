# AI CCTV Threat Detection

AI-powered security monitoring prototype: analyzes CCTV/webcam video in real
time, classifies threats into four alert levels, and raises alerts with
snapshots — based on the *AI CCTV Threat Detection System* project brief.

Two editions share the same detection models and threat rules:

| | **Browser demo** (`web/`) | **Full edition** (`app/`) |
|---|---|---|
| Runs on | Vercel / any static host | Local machine (Python) |
| Inference | ONNX Runtime Web (WebGPU/WASM), in-browser | PyTorch + OpenCV, server-side |
| Sources | Webcam, image upload | Webcam, video files, RTSP IP cameras |
| Alerts | Local feed + browser storage | SQLite history, snapshots, notifier hooks (SMS/push) |
| Privacy | Video never leaves the device | Video never leaves your network |

## Threat model

Two YOLOv8 detectors run on every frame — a COCO model (people, vehicles,
packages) and a custom-trained guns/knife model — feeding a centroid tracker
and rule engine.

Because the raw weapons model false-positives heavily (laptop screens, pens,
whole-room boxes), weapon candidates pass a **fusion filter** before they
count: oversized boxes are rejected, boxes coinciding with a person are
rejected, boxes overlapping a confident everyday COCO object (laptop, phone,
remote…) are rejected, and the confidence bar depends on context (lower when
a person is nearby or COCO's own `knife` class agrees, higher for weapons
floating alone). A temporal vote (weapon in ≥ 5 of the last 8 frames) then
suppresses single-frame flickers. Measured on the weapons model's own demo
video: 13/13 frames with real knives still detect after filtering.

| Level | Rule |
|---|---|
| **LOW** | Person mostly stationary ≥ 15 s (loitering) |
| **MEDIUM** | Person beside a vehicle ≥ 8 s, or inside a restricted zone |
| **HIGH** | Weapon detected across ≥ 3 consecutive frames |
| **CRITICAL** | Weapon persisting in scene ≥ 5 s |

## Browser demo

Deployed on Vercel as a static site; models (~24 MB, cached) load via CDN and
run entirely client-side.

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
[README_PROTOTYPE.md](README_PROTOTYPE.md); thresholds in `app/config.py`.

## Repo layout

```
app/          FastAPI + pipeline (detection, tracking, threat rules, alerts)
web/          static browser edition + ONNX models (served to Vercel via CDN)
Weapons-and-Knives-Detector-with-YOLOv8-main/
              trained weapons weights (only weights + attribution committed)
```

## Licensing & attribution

- Weapons model weights from
  [Weapons-and-Knives-Detector-with-YOLOv8](https://github.com/JoaoAssalim/Weapons-and-Knives-Detector-with-YOLOv8) (GPL-3.0).
- Built with [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) (AGPL-3.0).
- This repository is licensed **AGPL-3.0** (see [LICENSE](LICENSE)).

Prototype for education/research. Weapon detection has false positives and
misses — do not rely on it for real security decisions. Check local laws
before pointing cameras at public spaces or using face recognition.
