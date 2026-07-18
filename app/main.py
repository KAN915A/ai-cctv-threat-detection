"""FastAPI backend: dashboard, MJPEG video feed, WebSocket alerts, REST API.

Run with:
    uvicorn app.main:app --port 8000
"""

import asyncio
import base64
import time
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from .config import SNAPSHOT_DIR, settings
from .pipeline import pipeline
from .threat import ThreatClassifier, highest_level
from .tracker import CentroidTracker

app = FastAPI(title="AI CCTV Threat Detection — Prototype")

STATIC_DIR = Path(__file__).parent / "static"


# ------------------------------------------------------------------ pages --
@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/live")
def live_page():
    """Browser-camera live test: open on any phone/laptop on the network,
    grant camera access, and its frames are analyzed by this server."""
    return FileResponse(STATIC_DIR / "live.html")


@app.get("/snapshots/{name}")
def snapshot(name: str):
    path = (SNAPSHOT_DIR / name).resolve()
    if path.parent != SNAPSHOT_DIR.resolve() or not path.exists():
        return {"error": "not found"}
    return FileResponse(path)


# ---------------------------------------------------------------- control --
class StartRequest(BaseModel):
    source: str = "0"


@app.post("/api/start")
def start(req: StartRequest):
    src: int | str = int(req.source) if req.source.isdigit() else req.source
    pipeline.start(src)
    # give the capture thread a moment to surface immediate errors
    time.sleep(1.0)
    return {"running": pipeline.running, "error": pipeline.error}


@app.post("/api/stop")
def stop():
    pipeline.stop()
    return {"running": False}


@app.get("/api/status")
def status():
    return pipeline.snapshot_state()


@app.get("/api/events")
def events(limit: int = 50):
    return pipeline.alerts.recent_events(limit)


class ZoneRequest(BaseModel):
    zone: list[float] | None = None  # [x1, y1, x2, y2] fractions, or null


@app.post("/api/zone")
def set_zone(req: ZoneRequest):
    if req.zone is not None and len(req.zone) == 4:
        settings.restricted_zone = tuple(req.zone)
    else:
        settings.restricted_zone = None
    return {"restricted_zone": settings.restricted_zone}


# --------------------------------------------------------------- telegram --
class TelegramConfigRequest(BaseModel):
    token: str | None = None       # empty string clears the token
    chat_id: str | None = None
    min_level: str | None = None   # LOW / MEDIUM / HIGH / CRITICAL


@app.get("/api/telegram")
def telegram_status():
    return pipeline.alerts.telegram.status()


@app.post("/api/telegram")
def telegram_configure(req: TelegramConfigRequest):
    pipeline.alerts.telegram.configure(
        token=req.token, chat_id=req.chat_id, min_level=req.min_level)
    return pipeline.alerts.telegram.status()


@app.post("/api/telegram/test")
def telegram_test():
    return pipeline.alerts.telegram.send_test()


# ------------------------------------------------------------------ video --
@app.get("/video_feed")
def video_feed():
    def generate():
        while True:
            jpeg = pipeline.jpeg
            if jpeg is None:
                if not pipeline.running:
                    break
                time.sleep(0.05)
                continue
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                   + jpeg + b"\r\n")
            time.sleep(0.03)
            if not pipeline.running:
                break
    return StreamingResponse(
        generate(), media_type="multipart/x-mixed-replace; boundary=frame")


# -------------------------------------------------------------- websocket --
@app.websocket("/ws/live")
async def ws_live(websocket: WebSocket):
    """Receives JPEG frames from a browser camera, runs the full pipeline
    (detection ensemble, tracking, threat rules, alerts) and streams the
    annotated result back. Each connection gets its own tracker/classifier;
    the models and alert engine are shared with the main pipeline."""
    await websocket.accept()
    tracker = CentroidTracker()
    classifier = ThreatClassifier()

    def process(frame):
        with pipeline.infer_lock:
            detections, inference_ms = pipeline.engine.detect(frame)
        persons = [d for d in detections if d.kind == "person"]
        tracks = tracker.update(persons)
        threats = classifier.classify(detections, tracks, frame.shape)
        annotated = pipeline._annotate(frame.copy(), detections, threats)
        pipeline.alerts.process(threats, annotated)
        ok, buf = cv2.imencode(".jpg", annotated,
                               [cv2.IMWRITE_JPEG_QUALITY, 75])
        return {
            "frame": base64.b64encode(buf.tobytes()).decode() if ok else None,
            "threat_level": highest_level(threats),
            "threats": [{"level": t.level, "kind": t.kind,
                         "message": t.message} for t in threats],
            "inference_ms": round(inference_ms, 1),
        }

    try:
        while True:
            data = await websocket.receive_bytes()
            arr = np.frombuffer(data, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is None:
                continue
            # Heavy CPU work off the event loop so other clients stay live
            result = await asyncio.to_thread(process, frame)
            await websocket.send_json(result)
    except WebSocketDisconnect:
        pass
    except Exception:
        pass


@app.websocket("/ws")
async def ws(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            await websocket.send_json(pipeline.snapshot_state())
            await asyncio.sleep(0.5)
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
