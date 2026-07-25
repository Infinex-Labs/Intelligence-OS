"""Identity / Re-identification — the foundation (§C, FR-4..7).

People: durable face embedding (insightface buffalo_l, 512-d, L2-normalized) is
the ONLY long-term identity signal. Body/clothing is same-day only and never used
for cross-day matching (clothing changes daily). Match probe -> gallery by cosine;
above threshold attach, else mint a new anonymous entity. Every association records
a confidence.

This module owns the single most important guarantee in the system: the same
person stays one entity across days, and two people stay distinct. If this is
unreliable, every higher layer narrates a fictional world — so it is kept small,
explicit, and tunable.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .config import CONFIG
from .store import Store


@dataclass
class FaceDetection:
    bbox: tuple[int, int, int, int]   # x1,y1,x2,y2
    embedding: np.ndarray             # 512-d, L2-normalized
    det_score: float


@dataclass
class Match:
    entity_id: str
    confidence: float                 # cosine similarity to best signature
    minted: bool                      # True if a new entity was created


def l2norm(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def _ort_device():
    """(providers, ctx_id) for insightface. GPU only if onnxruntime-gpu is the
    installed wheel (its provider list carries CUDA); else CPU. ctx_id<0 = CPU in
    insightface's convention — the old code passed ctx_id=0 (GPU) with a CPU-only
    provider list, which only worked because onnxruntime silently fell back."""
    try:
        import onnxruntime as ort
        if "CUDAExecutionProvider" in ort.get_available_providers():
            return ["CUDAExecutionProvider", "CPUExecutionProvider"], 0
    except Exception:
        pass
    return ["CPUExecutionProvider"], -1


class FaceEmbedder:
    """Wraps insightface. Lazy-loads the model so importing this module is cheap
    and tests that don't need real faces stay fast."""

    def __init__(self, det_size: int = 640):
        self._app = None
        self._det_size = det_size

    def _ensure(self):
        if self._app is None:
            from insightface.app import FaceAnalysis  # heavy import, deferred
            providers, ctx_id = _ort_device()
            print(f"[identity] InsightFace providers={providers} ctx_id={ctx_id}")
            app = FaceAnalysis(name="buffalo_l", providers=providers)
            app.prepare(ctx_id=ctx_id, det_size=(self._det_size, self._det_size))
            self._app = app
        return self._app

    def detect(self, frame_bgr: np.ndarray) -> list[FaceDetection]:
        app = self._ensure()
        out = []
        for f in app.get(frame_bgr):
            if f.det_score < CONFIG.identity.min_face_det_score:
                continue
            emb = l2norm(np.asarray(f.normed_embedding, dtype=np.float32))
            x1, y1, x2, y2 = [int(v) for v in f.bbox]
            out.append(FaceDetection((x1, y1, x2, y2), emb, float(f.det_score)))
        return out


class IdentityResolver:
    """Match-or-mint against the persistent gallery in the Store."""

    def __init__(self, store: Store, threshold: Optional[float] = None):
        self.store = store
        self.threshold = (threshold if threshold is not None
                          else CONFIG.identity.face_match_threshold)

    def _best_match(self, emb: np.ndarray) -> tuple[Optional[str], float]:
        """Cosine similarity is a dot product on L2-normalized vectors. Score an
        entity by its single best-matching signature (max), not the mean, so one
        odd-angle signature doesn't drag a true match below threshold."""
        best_id, best_sim = None, -1.0
        for entity_id, _sig_id, vec in self.store.signatures(kind="face"):
            sim = float(np.dot(emb, vec))
            if sim > best_sim:
                best_sim, best_id = sim, entity_id
        return best_id, best_sim

    def resolve(self, det: FaceDetection) -> Match:
        emb = l2norm(det.embedding)
        entity_id, sim = self._best_match(emb)
        if entity_id is not None and sim >= self.threshold:
            self._maybe_store_signature(entity_id, emb)
            return Match(entity_id, sim, minted=False)
        # No confident match -> mint a new anonymous entity (entity_N).
        new_id = self.store.create_entity("person")
        self.store.add_signature(new_id, "face", emb)
        return Match(new_id, max(0.0, sim), minted=True)

    def _maybe_store_signature(self, entity_id: str, emb: np.ndarray) -> None:
        """Grow the gallery with genuinely *new* appearances only (pose/lighting),
        capped, so we capture variation without storing N near-duplicates (§C)."""
        cfg = CONFIG.identity
        existing = self.store.entity_signatures(entity_id, kind="face")
        if existing:
            max_sim = max(float(np.dot(emb, v)) for v in existing)
            if (1.0 - max_sim) < cfg.signature_novelty_min:
                return  # too similar to what we already have; skip
        self.store.add_signature(entity_id, "face", emb)
        self.store.prune_signatures(entity_id, "face", cfg.max_signatures_per_entity)
