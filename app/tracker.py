"""Lightweight centroid tracker for persons.

Assigns stable IDs across frames and records dwell time + movement so the
threat classifier can detect loitering and lingering near vehicles.
"""

import math
import time
from dataclasses import dataclass, field

from .detection import Detection


@dataclass
class Track:
    track_id: int
    center: tuple
    box: tuple
    first_seen: float
    last_seen: float
    positions: list = field(default_factory=list)  # recent centers
    near_vehicle_since: float | None = None

    @property
    def age(self) -> float:
        return self.last_seen - self.first_seen

    def travel_radius(self) -> float:
        """How far the track has wandered from its average position."""
        if len(self.positions) < 2:
            return 0.0
        xs = [p[0] for p in self.positions]
        ys = [p[1] for p in self.positions]
        cx, cy = sum(xs) / len(xs), sum(ys) / len(ys)
        return max(math.hypot(x - cx, y - cy) for x, y in self.positions)


class CentroidTracker:
    def __init__(self, max_distance: float = 120.0, max_missing_seconds: float = 2.0):
        self.max_distance = max_distance
        self.max_missing_seconds = max_missing_seconds
        self.tracks: dict[int, Track] = {}
        self._next_id = 1

    def update(self, persons: list[Detection]) -> dict[int, Track]:
        now = time.time()

        # Match detections to existing tracks greedily by distance
        unmatched = list(persons)
        for track in sorted(self.tracks.values(), key=lambda t: t.last_seen, reverse=True):
            if not unmatched:
                break
            best, best_dist = None, self.max_distance
            for det in unmatched:
                d = math.hypot(det.center[0] - track.center[0],
                               det.center[1] - track.center[1])
                if d < best_dist:
                    best, best_dist = det, d
            if best is not None:
                unmatched.remove(best)
                track.center = best.center
                track.box = best.box
                track.last_seen = now
                track.positions.append(best.center)
                if len(track.positions) > 150:
                    track.positions.pop(0)
                best.track_id = track.track_id

        # New tracks for unmatched detections
        for det in unmatched:
            track = Track(
                track_id=self._next_id, center=det.center, box=det.box,
                first_seen=now, last_seen=now, positions=[det.center],
            )
            det.track_id = track.track_id
            self.tracks[self._next_id] = track
            self._next_id += 1

        # Drop stale tracks
        stale = [tid for tid, t in self.tracks.items()
                 if now - t.last_seen > self.max_missing_seconds]
        for tid in stale:
            del self.tracks[tid]

        return self.tracks
