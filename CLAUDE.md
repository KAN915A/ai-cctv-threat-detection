# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

AI CCTV threat-detection prototype. Analyzes CCTV/webcam video with a three-model YOLOv8
ensemble, classifies threats into LOW / MEDIUM / HIGH / CRITICAL, and raises alerts with
snapshots. There are **three editions that share one detection design**:

- `app/` — Python/FastAPI dashboard, server-side PyTorch + OpenCV inference (the reference implementation).
- `web/` — static browser edition; the *same* fusion + threat rules re-implemented in JS on ONNX Runtime Web.
- `android/` — native Kotlin client that talks to the `app/` server over HTTP (it is not a detector itself).

## Critical: the detection logic is duplicated in two languages

`web/app.js` is a hand-port of the Python pipeline (`app/detection.py`, `app/threat.py`,
`app/tracker.py`). **Every threshold and rule exists twice.** When you change fusion
constants, threat timings, distractor/dangerous class sets, or vote windows in
`app/config.py`, you must mirror the change in the constants block at the top of
`web/app.js` (`WEAPON_CONF_*`, `LOITER_SECONDS`, `WEAPON_WINDOW`, `DISTRACTORS`, etc.), or
the two editions will disagree. There is no shared source of truth and no test that catches drift.

## Commands

Python edition (run from repo root; a `.venv` already exists):
```bash
.venv\Scripts\python.exe -m uvicorn app.main:app --port 8000   # dashboard at http://localhost:8000
# or double-click start_dashboard.bat
# for the Android app / LAN access, add: --host 0.0.0.0
```
Dashboard source input: `0` (webcam), a video path like `test_clip.mp4`, or an `rtsp://` URL.

Browser edition:
```bash
python -m http.server 8090 --directory web   # then open http://localhost:8090
```

Evaluation / verification harnesses (no pytest; these are standalone scripts run from repo
root, importing `app.*` as a package). They depend on runtime artifacts that are
`.gitignore`d (`snapshots/`, `test_clip.mp4`, `bus.jpg`) and on the weapons demo video, so
they only work on a machine that has produced/downloaded those assets:
```bash
.venv\Scripts\python.exe eval_accuracy.py   # positives survive / negatives get suppressed
.venv\Scripts\python.exe eval_video.py      # weapons demo video, old rule vs fusion per frame
.venv\Scripts\python.exe verify_medium.py   # MEDIUM vehicle-lurking path on test_clip.mp4
```

Android build (from `android/`):
```powershell
$env:JAVA_HOME = "C:\Program Files\Android\Android Studio1\jbr"
$env:JDK_JAVA_OPTIONS = "-Djdk.net.unixdomain.tmpdir=C:\Temp"   # this machine: default TMP breaks JDK AF_UNIX loopback
.\gradlew.bat assembleDebug      # -> app/build/outputs/apk/debug/
.\gradlew.bat bundleRelease      # signed AAB -> app/build/outputs/bundle/release/
```

## Python pipeline architecture

`Pipeline` (`app/pipeline.py`) runs a background capture→detect→track→classify→alert thread
and hands the web layer the latest annotated JPEG plus a state dict. `app/main.py` is a thin
FastAPI shell over the singleton `pipeline`: `/video_feed` (MJPEG), `/ws` (state at 2 Hz),
`/api/start|stop|status|events|zone`. Only one viewer is really supported — `new_alerts` are
consumed on read in `snapshot_state()`.

`DetectionEngine.detect()` (`app/detection.py`) is the heart of the false-positive defense:
1. COCO model (`yolov8n.pt`) → people, vehicles, packages, and context objects.
2. Weapons **ensemble** — two custom models (`WEAPON_MODEL_PATHS`). `merge_candidates()`
   collapses same-label overlapping boxes across models and records how many agreed.
3. `fuse_weapons()` gates every candidate: size veto → person-coincidence veto → distractor
   veto → **tiered confidence** (the lowest bar it qualifies for: coco-agree 0.40, ensemble
   0.45, near-person 0.55, alone 0.70). Survivors carry `meta['basis']`.

`ThreatClassifier` (`app/threat.py`) adds the **temporal vote** — a weapon must appear in
≥ `weapon_votes` of the last `weapon_window` frames before HIGH fires; persisting ≥
`weapon_critical_seconds` escalates to CRITICAL. Loitering/vehicle-lurking use `CentroidTracker`
dwell state; the altercation rule is a motion heuristic (two tracks fast + close, voted over a window).

`AlertEngine` (`app/alerts.py`) dedupes per `(level, kind)` on a cooldown, writes `events.db`
(SQLite) + a snapshot JPEG, and fans out to `Notifier`s. Only `ConsoleNotifier` is live;
`WebhookNotifier` is the stub for Twilio SMS / FCM push.

**All tunable thresholds live in `app/config.py`** (`Settings` dataclass + the class-set constants).

## Weapons models and how they get into each edition

Trained weights come from the vendored `Weapons-and-Knives-Detector-with-YOLOv8/` repo. The
ensemble uses the `Normal` and `Normal_Compressed` runs (`runs/detect/<name>/weights/best.pt`);
other runs (Haar, Symlet, Db…) exist but are not wired in. `.gitignore` keeps only `best.pt`
weights + attribution from that directory, not the full dataset.

The browser edition needs ONNX, not `.pt`. Export scripts like `export_weapons2.py` convert a
`best.pt` to `web/models/*.onnx`. **The hosted/CDN demo does not load `web/models/` locally** —
`web/app.js` fetches models from a jsDelivr URL pinned to `KAN915A/ai-cctv-threat-detection@main`.
So a changed ONNX model only reaches the live demo after it is committed **and pushed** to that
GitHub repo's `main`; a local `web/` server likewise needs those files present under `web/models/`.

## Notes

- Licensed AGPL-3.0 (uses Ultralytics YOLOv8 + third-party weapons weights). Keep attribution intact.
- Prototype only — weapon detection has real false positives/misses; not for production security decisions.
