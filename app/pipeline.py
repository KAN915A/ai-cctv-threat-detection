"""Video pipeline: capture -> detect -> track -> classify -> alert.

Runs in a background thread and shares the latest annotated JPEG + state
snapshot with the web layer.
"""

import threading
import time

import cv2

from .alerts import AlertEngine
from .config import settings
from .detection import Detection, DetectionEngine
from .threat import Threat, ThreatClassifier, highest_level
from .tracker import CentroidTracker

LEVEL_COLORS = {           # BGR
    "LOW": (11, 180, 255),
    "MEDIUM": (0, 120, 255),
    "HIGH": (60, 60, 230),
    "CRITICAL": (200, 50, 200),
}
KIND_COLORS = {
    "person": (90, 200, 90),
    "vehicle": (230, 180, 60),
    "package": (200, 160, 120),
    "weapon": (60, 60, 230),
    "danger": (0, 140, 255),
}


class Pipeline:
    def __init__(self):
        self.engine = DetectionEngine()
        self.tracker = CentroidTracker()
        self.classifier = ThreatClassifier()
        self.alerts = AlertEngine()

        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self.running = False
        self.error: str | None = None
        self.source: int | str = 0

        self.jpeg: bytes | None = None
        self.state = {
            "running": False,
            "threat_level": None,
            "threats": [],
            "counts": {},
            "fps": 0.0,
            "inference_ms": 0.0,
            "new_alerts": [],
        }

    # ------------------------------------------------------------- control --
    def start(self, source: int | str = 0):
        if self.running:
            return
        prev = self._thread
        if prev is not None and prev.is_alive():
            self.running = False
            prev.join(timeout=5)
        self.source = source
        self.running = True
        self.error = None
        self.jpeg = None
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self.running = False

    def snapshot_state(self) -> dict:
        with self._lock:
            state = dict(self.state)
            state["running"] = self.running
            state["error"] = self.error
            # new_alerts are consumed on read so each alert is pushed once
            self.state["new_alerts"] = []
        return state

    # ---------------------------------------------------------------- loop --
    def _open_capture(self):
        src = self.source
        if isinstance(src, int):
            # CAP_DSHOW opens much faster than default MSMF on Windows
            cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap.release()
                cap = cv2.VideoCapture(src)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        else:
            cap = cv2.VideoCapture(src)
        return cap

    def _loop(self):
        cap = None
        try:
            cap = self._open_capture()
            if not cap.isOpened():
                self.error = f"Could not open video source: {self.source}"
                self.running = False
                return

            is_file = isinstance(self.source, str) and not str(
                self.source).lower().startswith(("rtsp://", "http://", "https://"))

            last_frame_ts = time.time()
            while self.running:
                ret, frame = cap.read()
                if not ret:
                    if is_file:  # loop video files for demo purposes
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        self.tracker = CentroidTracker()
                        continue
                    self.error = "Video stream ended"
                    self.running = False
                    break

                # Keep inference size bounded
                h, w = frame.shape[:2]
                if w > settings.inference_width:
                    scale = settings.inference_width / w
                    frame = cv2.resize(frame, (settings.inference_width,
                                               int(h * scale)))

                detections, inference_ms = self.engine.detect(frame)
                persons = [d for d in detections if d.kind == "person"]
                tracks = self.tracker.update(persons)
                threats = self.classifier.classify(detections, tracks,
                                                   frame.shape)

                annotated = self._annotate(frame.copy(), detections, threats)
                fired = self.alerts.process(threats, annotated)

                ok, buf = cv2.imencode(".jpg", annotated,
                                       [cv2.IMWRITE_JPEG_QUALITY, 80])

                now = time.time()
                fps = 1.0 / max(now - last_frame_ts, 1e-6)
                last_frame_ts = now

                counts = {}
                for d in detections:
                    counts[d.kind] = counts.get(d.kind, 0) + 1

                with self._lock:
                    if ok:
                        self.jpeg = buf.tobytes()
                    self.state.update({
                        "threat_level": highest_level(threats),
                        "threats": [
                            {"level": t.level, "kind": t.kind,
                             "message": t.message} for t in threats
                        ],
                        "counts": counts,
                        "fps": round(fps, 1),
                        "inference_ms": round(inference_ms, 1),
                    })
                    self.state["new_alerts"].extend(fired)

        except Exception as e:
            self.error = f"Pipeline failed: {e}"
            self.running = False
        finally:
            if cap is not None:
                cap.release()

    # ---------------------------------------------------------- annotation --
    def _annotate(self, frame, detections: list[Detection],
                  threats: list[Threat]):
        for det in detections:
            x1, y1, x2, y2 = det.box
            color = KIND_COLORS.get(det.kind, (150, 150, 150))
            thick = 3 if det.kind == "weapon" else 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thick)
            label = det.label
            if det.track_id is not None:
                label += f" #{det.track_id}"
            label += f" {det.confidence:.0%}"
            cv2.rectangle(frame, (x1, max(0, y1 - 22)),
                          (x1 + 9 * len(label), y1), color, -1)
            cv2.putText(frame, label, (x1 + 3, max(15, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Restricted zone overlay
        if settings.restricted_zone is not None:
            h, w = frame.shape[:2]
            zx1, zy1, zx2, zy2 = settings.restricted_zone
            p1 = (int(zx1 * w), int(zy1 * h))
            p2 = (int(zx2 * w), int(zy2 * h))
            cv2.rectangle(frame, p1, p2, (0, 200, 255), 2)
            cv2.putText(frame, "RESTRICTED", (p1[0] + 4, p1[1] + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 2)

        # Threat banner
        level = highest_level(threats)
        if level:
            color = LEVEL_COLORS[level]
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 34), color, -1)
            cv2.putText(frame, f"THREAT LEVEL: {level}", (10, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        return frame


pipeline = Pipeline()
