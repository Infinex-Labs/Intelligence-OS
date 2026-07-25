"""Detection + object persistence (§B, §C — Phase 2).

YOLO detects persons + configured object classes on motion frames. A tracker
(ByteTrack via ultralytics) gives the SAME object one identity across frames; an
appearance embedding + location prior re-matches it across days (location prior is
acceptable for objects — they're static — but is NEVER used for people, §C / NFR-7).

Objects become entities too, and are mergeable/splittable like people.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from .config import CONFIG
from .store import Store


def _yolo_device():
    """Ultralytics device arg: GPU 0 if a CUDA-enabled torch is installed, else CPU.
    Deploy-time choice — a CUDA torch wheel is what flips this, not the code."""
    try:
        import torch
        return 0 if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


@dataclass
class Detection:
    cls_name: str
    bbox: tuple[int, int, int, int]   # x1,y1,x2,y2
    conf: float
    track_id: Optional[int] = None    # within-stream tracker id
    appearance: Optional[np.ndarray] = None  # color-histogram signature


def _appearance_signature(crop_bgr: np.ndarray) -> np.ndarray:
    """A cheap appearance embedding for object re-ID: normalized HS color
    histogram. Not as strong as a learned embedding, but enough for the POC's
    cross-day object re-matching when combined with the location prior (§C)."""
    if crop_bgr.size == 0:
        return np.zeros(50, dtype=np.float32)
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [10, 5], [0, 180, 0, 256])
    hist = cv2.normalize(hist, hist).flatten().astype(np.float32)
    n = np.linalg.norm(hist)
    return hist / n if n > 0 else hist


class Detector:
    """YOLO wrapper that runs with the built-in tracker so detections carry a
    persistent track_id within a stream."""

    def __init__(self, weights: Optional[str] = None):
        import os
        from ultralytics import YOLO  # heavy import deferred
        w = weights or CONFIG.detect.yolo_weights
        # §10: a bare name (not an existing path) -> Ultralytics fetches + caches it
        # on first run with its own progress bar. Flag it so the wait isn't a mystery.
        if not os.path.exists(w):
            print(f"[detect] YOLO weights '{w}' not local — fetching on first run "
                  f"(cached after)…")
        self.model = YOLO(w)
        self.names = self.model.names
        self._wanted = set(CONFIG.detect.object_classes) | {"person"}
        self._device = _yolo_device()   # GPU if a CUDA torch build is installed, else CPU
        print(f"[detect] YOLO on device={self._device}")

    def track(self, frame_bgr: np.ndarray, persist: bool = True) -> list[Detection]:
        cfg = CONFIG.detect
        res = self.model.track(frame_bgr, persist=persist, conf=cfg.conf,
                               iou=cfg.iou, tracker=cfg.tracker_cfg,
                               device=self._device, verbose=False)[0]
        out: list[Detection] = []
        if res.boxes is None:
            return out
        boxes = res.boxes
        ids = boxes.id.int().cpu().tolist() if boxes.id is not None else [None] * len(boxes)
        for b, tid in zip(boxes, ids):
            cls_name = self.names[int(b.cls)]
            if cls_name not in self._wanted:
                continue
            x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].cpu().tolist()]
            crop = frame_bgr[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
            out.append(Detection(
                cls_name=cls_name, bbox=(x1, y1, x2, y2), conf=float(b.conf),
                track_id=tid,
                appearance=_appearance_signature(crop) if cls_name != "person" else None,
            ))
        return out


class ObjectRegistry:
    """Maps tracker ids and appearance signatures to durable object entities.

    Within a stream: track_id -> entity (stable).
    Across days/restarts: re-match by appearance similarity + location prior,
    else mint. Objects are entities, so the same merge/split/delete machinery
    applies (§C 'objects also get entity ids and are mergeable/splittable')."""

    def __init__(self, store: Store, appearance_threshold: float = 0.6,
                 kind: str = "object"):
        # kind="person" is the no-face path (FR-ST-3 off by default): track id
        # within a stream, appearance across restarts, no biometrics at all.
        self.store = store
        self.kind = kind
        self.appearance_threshold = appearance_threshold
        self._track_to_entity: dict[tuple[str, int], str] = {}

    def resolve(self, det: Detection) -> str:
        # 1) within-stream: tracker id we've already bound to an entity
        key = (det.cls_name, det.track_id) if det.track_id is not None else None
        if key is not None and key in self._track_to_entity:
            return self._track_to_entity[key]

        # 2) cross-day: appearance re-match against stored object signatures
        entity_id = None
        if det.appearance is not None:
            entity_id, sim = self._best_appearance(det)
            if sim < self.appearance_threshold:
                entity_id = None

        # 3) mint a fresh object entity
        if entity_id is None:
            # people stay unlabeled — the UI calls them "Unknown Person" until named
            entity_id = self.store.create_entity(
                self.kind, label=None if self.kind == "person" else det.cls_name)
            if det.appearance is not None:
                self.store.add_signature(entity_id, "appearance", det.appearance)

        if key is not None:
            self._track_to_entity[key] = entity_id
        return entity_id

    def _best_appearance(self, det: Detection) -> tuple[Optional[str], float]:
        best_id, best_sim = None, -1.0
        for entity_id, _sig_id, vec in self.store.signatures(kind="appearance"):
            ent = self.store.get_entity(entity_id)
            # only re-match within the same class (unlabeled people match by look alone)
            if ent is None or ent["type"] != self.kind or \
                    (ent["label"] or det.cls_name) != det.cls_name:
                continue
            sim = float(np.dot(det.appearance, vec))
            if sim > best_sim:
                best_sim, best_id = sim, entity_id
        return best_id, best_sim
