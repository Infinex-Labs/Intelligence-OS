"""Scene-state memory (§D — Phase 3).

For each defined location/region in the fixed frame, maintain the set of entities
present over time — an inventory over time. This is the baseline that change
detection (§H) diffs against: "new" = not in this location's prior inventory;
"gone" = was present, now absent across sufficient observations.

Locations are manual config for a fixed camera (loaded from zones.json).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

from .store import Store


def _point_in_poly(x: float, y: float, poly: Sequence[Sequence[float]]) -> bool:
    inside = False
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-9) + xi):
            inside = not inside
        j = i
    return inside


@dataclass
class Zone:
    location_id: str
    name: str
    polygon: list[list[float]]   # frame coords

    def contains_bbox(self, bbox: tuple[int, int, int, int]) -> bool:
        """An entity is "in" a zone if the bottom-center of its bbox (where it
        meets a surface/floor) falls inside the polygon."""
        x1, y1, x2, y2 = bbox
        cx, cy = (x1 + x2) / 2.0, float(y2)
        return _point_in_poly(cx, cy, self.polygon)


class SceneState:
    """Loads zones, assigns detections to zones, and writes per-location inventory
    snapshots over time."""

    def __init__(self, store: Store, zones: Optional[list[Zone]] = None):
        self.store = store
        self.zones: list[Zone] = zones or []

    @classmethod
    def from_json(cls, store: Store, path: str | Path) -> "SceneState":
        data = json.loads(Path(path).read_text())
        zones = []
        for z in data.get("zones", data if isinstance(data, list) else []):
            lid = store.upsert_location(z["name"], {"polygon": z["polygon"]},
                                        location_id=z.get("location_id"))
            zones.append(Zone(lid, z["name"], z["polygon"]))
        return cls(store, zones)

    def assign(self, bbox: tuple[int, int, int, int]) -> Optional[str]:
        """Return the location_id whose zone contains the bbox (first match)."""
        for z in self.zones:
            if z.contains_bbox(bbox):
                return z.location_id
        return None

    def snapshot(self, present_by_location: dict[str, list[str]],
                timestamp: Optional[float] = None) -> list[str]:
        """Write one inventory snapshot per location. present_by_location maps
        location_id -> list of entity_ids currently present there."""
        snap_ids = []
        for z in self.zones:
            ids = sorted(set(present_by_location.get(z.location_id, [])))
            snap_ids.append(self.store.add_snapshot(z.location_id, ids, timestamp))
        return snap_ids

    def inventory_at(self, location_id: str, timestamp: Optional[float] = None
                    ) -> list[str]:
        """What was present in a location at (or just before) a given time —
        answers 'what was on the table yesterday vs today' (§12 Phase-3)."""
        snaps = self.store.snapshots(location_id=location_id)
        if not snaps:
            return []
        if timestamp is None:
            chosen = snaps[-1]
        else:
            chosen = None
            for s in snaps:
                if s["timestamp"] <= timestamp:
                    chosen = s
                else:
                    break
            if chosen is None:
                return []
        return json.loads(chosen["present_entity_ids"])
