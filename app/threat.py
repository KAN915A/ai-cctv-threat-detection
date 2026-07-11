"""Threat classification engine.

Maps raw detections + track history to the alert levels from the project
brief:
    LOW      loitering
    MEDIUM   suspicious person near a vehicle, trespassing in restricted zone
    HIGH     weapon detected (confirmed across consecutive frames)
    CRITICAL weapon persisting in the scene (likely active threat)
"""

import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime

from .config import settings
from .detection import Detection
from .tracker import Track

LEVELS = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]

# Person considered "stationary" if they wandered less than this many px
LOITER_RADIUS_PX = 90


@dataclass
class Threat:
    level: str        # LOW / MEDIUM / HIGH / CRITICAL
    kind: str         # loitering / vehicle_lurking / trespassing / weapon / weapon_persistent
    message: str
    box: tuple | None = None


def _boxes_near(a: tuple, b: tuple, margin: int = 40) -> bool:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return not (ax2 + margin < bx1 or bx2 + margin < ax1 or
                ay2 + margin < by1 or by2 + margin < ay1)


def _in_zone(box: tuple, zone_px: tuple) -> bool:
    cx = (box[0] + box[2]) / 2
    cy = (box[1] + box[3]) / 2
    zx1, zy1, zx2, zy2 = zone_px
    return zx1 <= cx <= zx2 and zy1 <= cy <= zy2


def is_night() -> bool:
    hour = datetime.now().hour
    return hour >= settings.night_start_hour or hour < settings.night_end_hour


class ThreatClassifier:
    def __init__(self):
        self.weapon_window: deque = deque(maxlen=settings.weapon_window)
        self.weapon_first_seen: float | None = None

    def classify(
        self,
        detections: list[Detection],
        tracks: dict[int, Track],
        frame_shape: tuple,
    ) -> list[Threat]:
        threats: list[Threat] = []
        now = time.time()
        night = is_night()

        persons = [d for d in detections if d.kind == "person"]
        vehicles = [d for d in detections if d.kind == "vehicle"]
        weapons = [d for d in detections if d.kind == "weapon"]

        # --- Weapon rules (HIGH -> CRITICAL) --------------------------------
        # Windowed vote: a weapon must show up in most recent frames before
        # it counts, so single-frame flickers never raise an alert.
        self.weapon_window.append(bool(weapons))
        confirmed = sum(self.weapon_window) >= settings.weapon_votes
        if confirmed:
            if self.weapon_first_seen is None:
                self.weapon_first_seen = now
        elif not any(self.weapon_window):
            self.weapon_first_seen = None

        if confirmed and weapons:
            top = max(weapons, key=lambda w: w.confidence)
            persisted = now - (self.weapon_first_seen or now)
            if persisted >= settings.weapon_critical_seconds:
                threats.append(Threat(
                    level="CRITICAL", kind="weapon_persistent",
                    message=(f"{top.label.upper()} persisting in scene for "
                             f"{persisted:.0f}s — possible active threat"),
                    box=top.box,
                ))
            else:
                threats.append(Threat(
                    level="HIGH", kind="weapon",
                    message=(f"{top.label.upper()} detected "
                             f"({top.confidence:.0%} confidence)"),
                    box=top.box,
                ))

        # --- Person near vehicle (MEDIUM) -----------------------------------
        for track in tracks.values():
            near = any(_boxes_near(track.box, v.box) for v in vehicles)
            if near:
                if track.near_vehicle_since is None:
                    track.near_vehicle_since = now
                elif now - track.near_vehicle_since >= settings.vehicle_lurk_seconds:
                    suffix = " at night" if night else ""
                    threats.append(Threat(
                        level="MEDIUM", kind="vehicle_lurking",
                        message=(f"Person #{track.track_id} lingering near a "
                                 f"vehicle for "
                                 f"{now - track.near_vehicle_since:.0f}s{suffix}"),
                        box=track.box,
                    ))
            else:
                track.near_vehicle_since = None

        # --- Restricted zone (MEDIUM) ----------------------------------------
        if settings.restricted_zone is not None:
            h, w = frame_shape[:2]
            zx1, zy1, zx2, zy2 = settings.restricted_zone
            zone_px = (zx1 * w, zy1 * h, zx2 * w, zy2 * h)
            for p in persons:
                if _in_zone(p.box, zone_px):
                    tid = f"#{p.track_id}" if p.track_id else ""
                    threats.append(Threat(
                        level="MEDIUM", kind="trespassing",
                        message=f"Person {tid} inside restricted zone",
                        box=p.box,
                    ))

        # --- Loitering (LOW) --------------------------------------------------
        for track in tracks.values():
            if (track.age >= settings.loiter_seconds
                    and track.travel_radius() < LOITER_RADIUS_PX):
                suffix = " during night hours" if night else ""
                threats.append(Threat(
                    level="LOW", kind="loitering",
                    message=(f"Person #{track.track_id} loitering for "
                             f"{track.age:.0f}s{suffix}"),
                    box=track.box,
                ))

        return threats


def highest_level(threats: list[Threat]) -> str | None:
    if not threats:
        return None
    return max(threats, key=lambda t: LEVELS.index(t.level)).level
