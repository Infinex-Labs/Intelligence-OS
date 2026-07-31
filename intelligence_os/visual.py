"""Visual re-ranking over stored keyframes (search plan Phase 8).

Alone among the phases this one answers no confirmed gap in the audit — it was
speculative when written, and stayed speculative right up to being measured.

Phase 8 was specified as a third recall index — CLIP vectors over keyframes,
fused beside the lexical and semantic lists, so "the red van" could be found in
a memory where nobody wrote the words down. This is not that, and the reason is
in `config.VisualConfig`: measured on this system's own retained frames, CLIP
cannot say NO. "A hospital bed" scores 0.953 on a dim room containing a dog and
a sofa, beating nine of the twelve things genuinely in shot. No threshold
separates present from absent, under any of the three calibrations tried.

A ranker that cannot detect absence is still a good ranker. So the rule here is
one line, and everything else follows from it:

    visual may reorder candidates another index retrieved.
    visual may never introduce one.

That keeps three earlier guarantees intact rather than trading them for reach.
Phase 4's hard filters still decide what is eligible. Phase 6's fusion still
only ever ADDS, and this narrows to only ever REORDERING. Phase 7's ladder still
sees a genuinely empty result when there is one — an index that always returns
its nearest frame would make every dead end look like an answer, which is the
exact failure the ladder exists to make visible.

What it buys: when the describer wrote "dog" on forty rows across a day, the
picture where the dog is actually the subject sorts to the front, and that is
the frame the answer cites.

    pip install sentence-transformers      # already the Phase 6 dependency
    INTELLIGENCE_OS_VISUAL=1 python -m intelligence_os.visual   # build the index

Off unless asked for. Nothing here reaches the network at query time.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from functools import lru_cache
from typing import Callable, Optional, Sequence

import numpy as np

from .config import CONFIG, FRAMES_DIR, retained_keyframe
from .store import Store

# Two encoders, because CLIP is two towers over one space. Images in, text in,
# vectors out — and the contract is callables for the same reason the semantic
# layer's is: a test can inject deterministic arithmetic and exercise the
# backlog, the de-duplication, the storage and the re-rank arithmetic without
# ever downloading 350MB.
ImageEncoder = Callable[[Sequence[str]], np.ndarray]      # file paths -> (n, dim)
TextEncoder = Callable[[Sequence[str]], np.ndarray]       # strings   -> (n, dim)

_images: Optional[ImageEncoder] = None
_texts: Optional[TextEncoder] = None
_encoder_name: Optional[str] = None
_load_failed = False


def set_encoder(images: Optional[ImageEncoder], texts: Optional[TextEncoder] = None,
                name: Optional[str] = None) -> None:
    """Install encoders, bypassing the lazy load. `None` restores the default.

    Both towers or neither. A stub that encodes images but leaves the real text
    tower in place would compare two unrelated spaces and return confident
    nonsense, which is the one failure mode a shared vector table makes easy.
    """
    global _images, _texts, _encoder_name, _load_failed
    _images = images
    _texts = texts if images is not None else None
    _encoder_name = name if images is not None else None
    _load_failed = False
    _encode_query.cache_clear()


def model_name() -> str:
    """The name the vectors currently being produced are stored under."""
    return _encoder_name or CONFIG.visual.model


def _load() -> bool:
    """Load the real CLIP towers once. False if this machine cannot.

    Latched on failure, exactly as the semantic loader is: a missing package or
    a model that will not download fails identically on every subsequent call,
    and retrying per question would turn one absent dependency into seconds of
    latency on every search.
    """
    global _images, _texts, _load_failed
    if _images is not None:
        return True
    if _load_failed:
        return False
    try:
        from PIL import Image                              # optional dep
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(CONFIG.visual.model)

        def encode_images(paths: Sequence[str]) -> np.ndarray:
            imgs = [Image.open(p).convert("RGB") for p in paths]
            try:
                return model.encode(imgs, convert_to_numpy=True,
                                    normalize_embeddings=True,
                                    show_progress_bar=False)
            finally:
                for im in imgs:
                    im.close()

        def encode_texts(texts: Sequence[str]) -> np.ndarray:
            return model.encode(list(texts), convert_to_numpy=True,
                                normalize_embeddings=True, show_progress_bar=False)

        _images, _texts = encode_images, encode_texts
    except Exception:
        # Deliberately broad, and for the same reason as the semantic layer's:
        # this is an optional stage, and no import error, download failure or
        # unreadable JPEG may take a search down when the ranking that was
        # already computed is sitting right there.
        _load_failed = True
        return False
    return True


def available() -> bool:
    """Whether this machine can actually produce a visual vector right now.

    Proven by encoding, not by importing — "the package is installed" and "the
    weights are on this disk" are different claims, and on a box with no network
    they differ in the direction that matters.
    """
    if not CONFIG.visual.enabled or not _load():
        return False
    try:
        v = _texts(["probe"])                              # type: ignore[misc]
        return v is not None and len(np.asarray(v)) > 0
    except Exception:
        return False


def _normalise(vecs) -> np.ndarray:
    """`(n, dim)` L2-normalised float32, so cosine is a dot product downstream.

    Done here rather than demanded of the encoder, so no caller has to remember
    which convention the one it installed followed.
    """
    v = np.asarray(vecs, dtype=np.float32)
    if v.ndim == 1:
        v = v.reshape(1, -1)
    n = np.linalg.norm(v, axis=1, keepdims=True)
    return v / np.where(n > 0, n, 1.0)


def encode_images(paths: Sequence[str]) -> Optional[np.ndarray]:
    """Vectors for these image files, or None if there is no encoder."""
    if not paths or not CONFIG.visual.enabled or not _load():
        return None
    try:
        return _normalise(_images(list(paths)))            # type: ignore[misc]
    except Exception:
        return None


@lru_cache(maxsize=256)
def _encode_query(text: str) -> Optional[np.ndarray]:
    """One query string through CLIP's TEXT tower, remembered.

    Not `semantic._encode_query`, and the distinction is the whole safety of the
    shared table: that one returns a MiniLM vector. Dotting a MiniLM query
    against a CLIP keyframe is meaningless where the dimensions differ and, far
    worse, merely wrong where they happen to agree.
    """
    if not CONFIG.visual.enabled or not (text or "").strip() or not _load():
        return None
    try:
        return _normalise(_texts([text]))[0]               # type: ignore[misc]
    except Exception:
        return None


def _frame_path(source_ref: Optional[str]) -> Optional[str]:
    """The readable file for a `source_ref`, or None if retention took it.

    Goes through `retained_keyframe` rather than trusting the stored path: the
    ref is absolute and recorded on the machine that wrote it, so it is the
    basename plus today's `FRAMES_DIR` that says whether there is a picture here
    to encode.
    """
    name = retained_keyframe(source_ref)
    return str(FRAMES_DIR / name) if name else None


# --- building the index ------------------------------------------------------

def backfill(store: Store, *, limit: Optional[int] = None,
             verbose: bool = False) -> int:
    """Embed keyframes that have no vector yet. Returns how many rows were written.

    Newest first, batch-committed and resumable, like the semantic backfill —
    and with one extra concern that one does not have. An observation's picture
    can be deleted by retention while the row itself lives on, so some of the
    backlog is permanently unembeddable. Those rows are stepped over with an
    offset rather than skipped-and-retried, or the newest-first walk would hand
    back the same dead batch until the end of time.

    Distinct FILES are encoded once and the vector written to every observation
    that cited them. A busy minute produces many rows against one picture, and
    a forward pass per row would pay for the same image dozens of times.
    """
    if not CONFIG.visual.enabled:
        return 0
    model = model_name()
    batch = max(1, int(CONFIG.visual.batch_size))
    written, offset = 0, 0
    while limit is None or written < limit:
        rows = store.keyframe_backlog(model, limit=batch, offset=offset)
        if not rows:
            break
        # Group the batch by the file it points at, keeping only pictures that
        # are still on disk.
        by_path: dict[str, list[str]] = {}
        for r in rows:
            path = _frame_path(r["source_ref"])
            if path:
                by_path.setdefault(path, []).append(r["observation_id"])
        if not by_path:
            offset += len(rows)          # a whole batch of pruned frames
            continue
        vecs = encode_images(list(by_path))
        if vecs is None:
            return written               # no encoder: nothing to do, not a failure
        payload: list[tuple[str, np.ndarray]] = []
        for vec, oids in zip(vecs, by_path.values()):
            payload.extend((oid, vec) for oid in oids)
        if limit is not None:
            payload = payload[:limit - written]
        written += store.add_embeddings(store.EMBED_KEYFRAME, model, payload)
        # Rows in this batch that could not be embedded stay in the backlog, so
        # the offset has to clear them or the next pass re-reads them forever.
        offset += len(rows) - sum(len(v) for v in by_path.values())
        if verbose:
            print(f"[visual] embedded {written} row(s) "
                  f"({len(by_path)} distinct frame(s) this batch)", flush=True)
        if len(rows) < batch:
            break
    return written


# --- re-ranking --------------------------------------------------------------

def rerank(store: Store, query: str, scores: dict[str, float]) -> dict[str, float]:
    """Reorder already-retrieved candidates by how well their picture fits `query`.

    Takes the fused scores and returns them adjusted. The contract is stated as
    an invariant rather than a tendency, and it is asserted in the tests:

        set(result) == set(scores)

    Nothing enters, nothing leaves. Only the order changes. That is what makes a
    ranker safe to use when it cannot detect absence — the row was already
    vouched for by an index that CAN, and all this decides is which vouched-for
    row is shown first.

    The boost is RRF-shaped: `weight / (rrf_k + visual_rank)`, the same form a
    fused list contributes, so agreement adds to a row's standing on the same
    scale the lexical and semantic lists were added on. With the default weight
    of 0.5 a re-rank cannot overturn two indexes that agree with each other; it
    breaks ties and settles the cases where they do not.

    Three gates, cheapest first, so a question asked of a memory with no visual
    index never pays for a model load to be told so.
    """
    cfg = CONFIG.visual
    if not cfg.enabled or not scores or not (query or "").strip():
        return scores
    if store.embedded_count(store.EMBED_KEYFRAME, model_name()) == 0:
        return scores

    # Only the head of the list is worth re-ranking: the tail is not going to
    # reach an answer whatever its picture looks like, and `top_k` is what keeps
    # a broad question from being a broad matrix multiply.
    head = sorted(scores, key=lambda oid: (-scores[oid], oid))[:max(1, int(cfg.top_k))]
    ids, mat = store.keyframe_vectors(model_name(), head)
    if mat is None:
        return scores
    q = _encode_query(query)
    if q is None or q.shape[-1] != mat.shape[-1]:
        # A dimension mismatch means two models have shared one name — the exact
        # corruption the `model` column exists to prevent, caught here rather
        # than silently scored.
        return scores

    sims = mat @ q
    order = sorted(range(len(ids)), key=lambda i: (-float(sims[i]), ids[i]))
    k = float(CONFIG.semantic.rrf_k)
    out = dict(scores)
    for rank, i in enumerate(order, start=1):
        out[ids[i]] = out[ids[i]] + float(cfg.weight) / (k + rank)
    return out


def reranked_ids(store: Store, query: str, scores: dict[str, float]) -> set[str]:
    """Which ids `rerank` would actually move. For the trace, and for tests.

    Recomputed rather than returned alongside the scores, because the caller
    that wants the trace and the caller that wants the ranking are different
    callers, and threading a second return value through the one that does not
    care would put it in every signature between here and there.
    """
    if not CONFIG.visual.enabled or not scores:
        return set()
    head = sorted(scores, key=lambda oid: (-scores[oid], oid))[
        :max(1, int(CONFIG.visual.top_k))]
    ids, _mat = store.keyframe_vectors(model_name(), head)
    return set(ids)


# --- CLI ---------------------------------------------------------------------

def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="intelligence_os.visual",
        description="Build the CLIP index over retained keyframes (Phase 8).")
    p.add_argument("--limit", type=int, default=None,
                   help="stop after this many rows (default: the whole backlog)")
    p.add_argument("--reindex", action="store_true",
                   help="drop this model's keyframe vectors first and rebuild")
    p.add_argument("--rank", metavar="QUERY",
                   help="rank every embedded keyframe against QUERY and print "
                        "the top of the list, with the caveat below")
    args = p.parse_args(argv)

    if not CONFIG.visual.enabled:
        print("Visual re-ranking is off. Turn it on with "
              "INTELLIGENCE_OS_VISUAL=1, or `visual_reranking: true` in "
              "config.yaml.", file=sys.stderr)
        return 1

    store = Store()
    try:
        if args.rank:
            # Deliberately NOT called `--query`, unlike the semantic CLI's, and
            # deliberately not a search. This prints an ORDER over the frames
            # that are indexed; it does not claim any of them contains what was
            # asked for, and the measurement in `config.VisualConfig` is why
            # nothing in this system is allowed to make that claim from a CLIP
            # score alone.
            ids, mat = store.keyframe_vectors(
                model_name(),
                [r["observation_id"] for r in store.conn.execute(
                    "SELECT ref_id AS observation_id FROM text_embeddings "
                    "WHERE kind=? AND model=?",
                    (store.EMBED_KEYFRAME, model_name()))])
            if mat is None:
                print("no keyframe vectors yet — run this without --rank first")
                return 0
            q = _encode_query(args.rank)
            if q is None:
                print("no CLIP text encoder available here", file=sys.stderr)
                return 1
            sims = mat @ q
            print(f"closest frames to {args.rank!r} — an ORDER, not a claim "
                  f"that any of them shows it:")
            # By PICTURE, not by row. The vectors are stored per observation, so
            # a frame a busy minute cited forty times would otherwise be forty
            # identical lines and push every other frame off the list.
            seen: dict[str, int] = {}
            for i in np.argsort(-sims):
                row = store.get_observation(ids[i])
                ref = os.path.basename((row["source_ref"] or "")) if row else "?"
                if ref in seen:
                    seen[ref] += 1
                    continue
                seen[ref] = 1
                print(f"  {sims[i]:.3f}  {ref}")
                if len(seen) >= 15:
                    break
            return 0

        if not available():
            print("No local CLIP model. Install it with:\n"
                  "  pip install sentence-transformers\n"
                  "Search is unaffected — visual re-ranking only ever changes "
                  "the ORDER of results the other indexes already found.",
                  file=sys.stderr)
            return 1
        if args.reindex:
            dropped = store.drop_embeddings(kind=store.EMBED_KEYFRAME,
                                            model=model_name())
            print(f"[visual] dropped {dropped} vector(s) for {model_name()}")
        started = time.perf_counter()
        n = backfill(store, limit=args.limit, verbose=True)
        held = store.embedded_count(store.EMBED_KEYFRAME, model_name())
        print(f"[visual] embedded {n} row(s) in "
              f"{time.perf_counter() - started:.1f}s; {held} vector(s) held")
        return 0
    finally:
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())
