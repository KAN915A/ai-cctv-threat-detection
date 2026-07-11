"""FastAPI backend: dashboard, MJPEG video feed, WebSocket alerts, REST API.

Run with:
    uvicorn app.main:app --port 8000
"""

import asyncio
import time
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from .config import SNAPSHOT_DIR, settings
from .pipeline import pipeline

app = FastAPI(title="AI CCTV Threat Detection — Prototype")

STATIC_DIR = Path(__file__).parent / "static"


# ------------------------------------------------------------------ pages --
@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


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
