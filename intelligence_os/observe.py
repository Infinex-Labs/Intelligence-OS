"""Memory write + cheap relational observations (§G — Phase 4).

Turns detections + identities into structured observation records (§6): present,
near (person-object proximity / interaction). Every observation carries timestamp,
confidence, provenance, and origin. A cooldown stops continuous presence from
spamming identical observations.

Per FR-8 an LLM may act as a *translator* (frame+detections -> structured record)
but never stores anything itself — our code writes the DB. The default translator
is deterministic and grounded (it can only assert what detection/geometry shows);
an LLM translator is an optional, swappable upgrade. No inferred relations here —
those are distilled later (§H).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .config import CONFIG
from .detect import Detection
from .store import Store


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / (area_a + area_b - inter + 1e-9)


def _center_dist_norm(a: tuple[int, int, int, int], b: tuple[int, int, int, int]
                     ) -> float:
    """Center distance normalized by the mean bbox size — scale-invariant
    proximity, so 'near' means the same at different distances from camera."""
    acx, acy = (a[0] + a[2]) / 2, (a[1] + a[3]) / 2
    bcx, bcy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    import math
    d = math.hypot(acx - bcx, acy - bcy)
    scale = (abs(a[2] - a[0]) + abs(a[3] - a[1]) +
             abs(b[2] - b[0]) + abs(b[3] - b[1])) / 4.0
    return d / (scale + 1e-9)


@dataclass
class ResolvedDetection:
    entity_id: str
    det: Detection
    location_id: Optional[str] = None
    camera_id: Optional[str] = None     # M6: which camera produced this detection


class Observer:
    """Writes grounded observations with cooldown. Stateless w.r.t. the scene;
    all memory lives in the Store."""

    def __init__(self, store: Store,
                 near_iou: float = 0.02, near_dist: float = 1.5):
        self.store = store
        self.near_iou = near_iou
        self.near_dist = near_dist
        self.cooldown = CONFIG.trigger.observation_cooldown_seconds
        # register the few structural predicates the deterministic translator uses
        store.register_predicate("present", subject_type="any", object_type="none")
        # object_type is "any", not "object": the thing a person is near may be a
        # dog or a van, not only furniture. `near` stays one predicate because it
        # states one observed fact — proximity. What that proximity *means* is the
        # distiller's call, and it picks a different verb per kind (distill.py).
        store.register_predicate("near", subject_type="person", object_type="any")

    def _cooled(self, subject: str, predicate: str, object_id: Optional[str],
               ts: float) -> bool:
        last = self.store.last_observation_time(subject, predicate, object_id)
        return last is not None and (ts - last) < self.cooldown

    def observe_frame(self, resolved: list[ResolvedDetection], timestamp: float,
                     source_ref: Optional[str] = None) -> list[str]:
        """Emit 'present' for every entity and 'near' for each person↔non-person
        pair in proximity. Cooldown suppresses repeats of the same situation."""
        ids: list[str] = []
        people = [r for r in resolved if r.det.cls_name == "person"]
        others = [r for r in resolved if r.det.cls_name != "person"]

        for r in resolved:
            if self._cooled(r.entity_id, "present", None, timestamp):
                continue
            ids.append(self.store.add_observation(
                r.entity_id, "present", location_id=r.location_id,
                confidence=min(0.99, r.det.conf), source_ref=source_ref,
                origin="detector", timestamp=timestamp,
                camera_id=r.camera_id))

        for p in people:
            for o in others:
                iou = _iou(p.det.bbox, o.det.bbox)
                dist = _center_dist_norm(p.det.bbox, o.det.bbox)
                if iou < self.near_iou and dist > self.near_dist:
                    continue
                if self._cooled(p.entity_id, "near", o.entity_id, timestamp):
                    continue
                # confidence scales with how strong the proximity evidence is
                conf = min(0.9, 0.4 + iou)
                ids.append(self.store.add_observation(
                    p.entity_id, "near", object_entity_id=o.entity_id,
                    location_id=p.location_id, confidence=conf,
                    source_ref=source_ref, origin="detector", timestamp=timestamp,
                    camera_id=p.camera_id))
        return ids
