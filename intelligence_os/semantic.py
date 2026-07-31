"""Meaning-based retrieval (search plan Phase 6, gap G1).

Phase 2 fixed word *variants*. Porter stems "cigarettes" to the same root as
"cigarette", so one finds the other. What no amount of stemming reaches is
different *vocabulary*: "loitering" and "standing around, waiting" share not one
character sequence, and a question phrased in the asker's words has to find a
sentence written in the model's.

So a second index, over the same prose, keyed by meaning rather than by term.
The layer is deliberately narrow:

  * It only ever ADDS candidates. Fusion is Reciprocal Rank Fusion over the
    lexical list and this one, so a row bm25 already found cannot be pushed out
    of an answer by a semantic ranker that disagrees.
  * It never crosses a hard filter. The window, the zone, the camera and the
    subject are in the SQL that produces the candidates, exactly as they are for
    the lexical search. A ranker may reorder what the filters allowed; it may
    not widen it.
  * It gates on an absolute score. `min_similarity` is the floor below which a
    "nearest" row is not a match at all — without it, every question returns the
    closest sentence in memory, which is how a semantic layer invents evidence
    for a thing that never happened.

The dependency is optional in the same sense `insightface` is. With no encoder
installed every function here returns nothing and search is exactly what it was
at Phase 2: fewer answers, none of them wrong.

    pip install sentence-transformers      # ~90MB of weights, downloaded once
    python -m intelligence_os.semantic     # backfill the index

Nothing here reaches the network at query time. The model runs locally, so the
"no cloud keys" property of every stage except the assistant's planner holds.
"""
from __future__ import annotations

import argparse
import sys
import time
from functools import lru_cache
from typing import Callable, Optional, Sequence

import numpy as np

from .config import CONFIG
from .store import Store

# What the encoder contract is: a callable taking a list of strings and
# returning an (n, dim) float array in the same order. That is all. It is a
# plain callable rather than a class so a test can inject twenty lines of
# deterministic arithmetic and exercise the fusion, the floor, the collapsing
# and the storage without ever downloading a model — which is what makes the
# rest of this file testable offline.
Encoder = Callable[[Sequence[str]], np.ndarray]

_encoder: Optional[Encoder] = None
_encoder_name: Optional[str] = None
_load_failed = False


def set_encoder(fn: Optional[Encoder], name: Optional[str] = None) -> None:
    """Install an encoder, bypassing the lazy load. `None` restores the default.

    `name` is stored beside every vector it produces and defaults to the
    configured model name. A test injecting a stub should pass its own name:
    two encoders under one name is exactly the failure the `model` column exists
    to prevent, and a test that ignores it would be building the corruption it
    is supposed to detect.
    """
    global _encoder, _encoder_name, _load_failed
    _encoder = fn
    _encoder_name = name if fn is not None else None
    _load_failed = False
    # Not optional. The query cache is keyed on the text alone, so leaving it
    # populated across an encoder swap would answer the next question with
    # vectors from the previous model — in the one space where nothing can tell
    # you that is what happened.
    _encode_query.cache_clear()


def model_name() -> str:
    """The name the vectors currently being produced are stored under."""
    return _encoder_name or CONFIG.semantic.model


def _load() -> Optional[Encoder]:
    """The real encoder, loaded once. `None` if this machine cannot.

    Failure is latched. A missing package, a model that will not download on a
    box with no network, an incompatible torch — all of them fail the same way
    and all of them would fail again on the next question. Retrying per query
    would turn one missing dependency into seconds of latency on every search.
    """
    global _encoder, _load_failed
    if _encoder is not None:
        return _encoder
    if _load_failed:
        return None
    try:
        from sentence_transformers import SentenceTransformer  # optional dep
        model = SentenceTransformer(CONFIG.semantic.model)

        def encode_batch(texts: Sequence[str]) -> np.ndarray:
            return model.encode(list(texts), convert_to_numpy=True,
                                normalize_embeddings=True, show_progress_bar=False)

        _encoder = encode_batch
    except Exception:
        # Deliberately broad. This is an optional stage: no import error, no
        # download failure and no CUDA misconfiguration may take a search down,
        # because the lexical answer is still there to return.
        _load_failed = True
        return None
    return _encoder


def available() -> bool:
    """Whether an encoder can actually produce a vector here, right now.

    Proven rather than declared — it encodes something. "The package imports"
    is not the same claim as "the weights are on this disk", and on a machine
    with no network those two answers differ. Tooling and tests read this;
    the query path does not, because it has a cheaper question to ask (are
    there any vectors?) and should not pay for a model load to find out there
    is nothing to search.
    """
    return encode(["probe"]) is not None


def encode(texts: Sequence[str]) -> Optional[np.ndarray]:
    """`(n, dim)` L2-normalised float32 vectors, or None if there is no encoder.

    Normalisation happens here rather than being required of the encoder, so
    cosine similarity is a dot product everywhere downstream and no caller has
    to remember which convention its encoder followed.
    """
    texts = [t or "" for t in texts]
    if not texts or not CONFIG.semantic.enabled:
        return None
    fn = _load()
    if fn is None:
        return None
    try:
        vecs = np.asarray(fn(texts), dtype=np.float32)
    except Exception:
        return None
    if vecs.ndim == 1:
        vecs = vecs.reshape(1, -1)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    # A zero vector has no direction, so it has no similarity to anything. The
    # 1.0 keeps the division finite and leaves it at zero, which scores below
    # any floor — the right answer for text that carried no signal.
    return vecs / np.where(norms > 0, norms, 1.0)


@lru_cache(maxsize=256)
def _encode_query(text: str) -> Optional[np.ndarray]:
    """One query string, encoded and remembered.

    A forward pass through the model is ~75ms on a CPU, which is the entire cost
    of a semantic search — the scan that follows it is microseconds. Questions
    repeat far more than they look like they do: the suggested prompts on an
    empty thread, a follow-up re-planned into the same words, an operator asking
    the same thing of two cameras, and every re-run of the eval.

    Cleared by `set_encoder`, because the cache is keyed on the text and knows
    nothing about which model produced the vector.
    """
    vecs = encode([text])
    return None if vecs is None else vecs[0]


# --- building the index ------------------------------------------------------

def backfill(store: Store, *, limit: Optional[int] = None,
             verbose: bool = False) -> int:
    """Embed prose that has no vector yet. Returns how many were written.

    Idempotent and resumable, because it will be interrupted: it walks the
    backlog in batches and each batch is committed on its own, so a kill signal
    costs the current batch rather than the run. Re-running it does no work for
    rows already covered — the backlog query is defined by absence, not by a
    cursor that could drift.

    Newest first. A memory being indexed for the first time is usable for recent
    questions long before the whole history is done, and recent questions are
    what people ask while they wait.
    """
    if not CONFIG.semantic.enabled:
        return 0
    model = model_name()
    batch = max(1, int(CONFIG.semantic.batch_size))
    written = 0
    while limit is None or written < limit:
        want = batch if limit is None else min(batch, limit - written)
        rows = store.embedding_backlog(model, limit=want)
        if not rows:
            break
        vecs = encode([r["text"] for r in rows])
        if vecs is None:
            return written        # no encoder: not a failure, just nothing to do
        written += store.add_embeddings(
            store.EMBED_OBSERVATION, model,
            list(zip([r["observation_id"] for r in rows], vecs)))
        if verbose:
            print(f"[semantic] embedded {written} rows", flush=True)
        if len(rows) < want:
            break
    return written


# --- searching ---------------------------------------------------------------

def search(store: Store, query: str, *, since: Optional[float] = None,
           until: Optional[float] = None,
           location_ids: Optional[Sequence[str]] = None,
           camera_ids: Optional[Sequence[str]] = None,
           entity_ids: Optional[Sequence[str]] = None,
           limit: Optional[int] = None,
           min_similarity: Optional[float] = None) -> list[tuple[str, float]]:
    """`[(observation_id, cosine)]` for the best matches, strongest first.

    Three gates, cheapest first, and the order is the point: a question asked of
    a memory with no semantic index must not pay for a model load to be told so.

      1. the layer is switched off                    -> a config read
      2. nothing is embedded under this model         -> one COUNT
      3. no encoder on this machine                   -> a latched import

    Only past all three does anything cost. The scan itself is brute force over
    the filtered candidates, held to a running top-k so memory is bounded by k
    rather than by the corpus — exact, because a maximum over chunks is the
    maximum over their union.
    """
    cfg = CONFIG.semantic
    if not cfg.enabled or not (query or "").strip():
        return []
    if store.embedded_count(store.EMBED_OBSERVATION, model_name()) == 0:
        return []
    q = _encode_query(query)
    if q is None:
        return []

    k = int(limit if limit is not None else cfg.top_k)
    floor = float(min_similarity if min_similarity is not None else cfg.min_similarity)

    best: list[tuple[str, float]] = []
    for ids, mat in store.embedded_chunks(
            model_name(), since=since, until=until, location_ids=location_ids,
            camera_ids=camera_ids, entity_ids=entity_ids,
            chunk=max(1, int(cfg.scan_chunk))):
        sims = mat @ q
        for i in np.flatnonzero(sims >= floor):
            best.append((ids[i], float(sims[i])))
        if len(best) > 4 * k:
            # Trim as we go rather than at the end. Without this a broad query
            # over a large memory accumulates every row above the floor before
            # cutting to k, which is the unbounded list the chunking was for.
            best.sort(key=lambda r: (-r[1], r[0]))
            del best[k:]
    # Ties broken by id so two runs over the same memory return the same order.
    # Ranked output feeds RRF, where an unstable order is an unstable answer.
    best.sort(key=lambda r: (-r[1], r[0]))
    return best[:k]


# --- CLI ---------------------------------------------------------------------

def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="intelligence_os.semantic",
        description="Build the semantic index over stored prose.")
    p.add_argument("--limit", type=int, default=None,
                   help="stop after this many rows (default: the whole backlog)")
    p.add_argument("--reindex", action="store_true",
                   help="drop this model's vectors first and rebuild them")
    p.add_argument("--query", help="search instead of indexing, and print the hits")
    args = p.parse_args(argv)

    store = Store()
    try:
        if args.query:
            hits = search(store, args.query)
            if not hits:
                print("no semantic hits (index empty, or nothing above the floor)")
                return 0
            for oid, score in hits[:20]:
                row = store.get_observation(oid)
                print(f"  {score:.3f}  {row['text'] if row else '(row gone)'}")
            return 0

        if not available():
            print("No local embedding model. Install it with:\n"
                  "  pip install sentence-transformers\n"
                  "Search still works — it falls back to the lexical index.",
                  file=sys.stderr)
            return 1
        if args.reindex:
            dropped = store.drop_embeddings(kind=store.EMBED_OBSERVATION,
                                            model=model_name())
            print(f"[semantic] dropped {dropped} vector(s) for {model_name()}")
        started = time.perf_counter()
        n = backfill(store, limit=args.limit, verbose=True)
        held = store.embedded_count(store.EMBED_OBSERVATION, model_name())
        print(f"[semantic] embedded {n} row(s) in "
              f"{time.perf_counter() - started:.1f}s; {held} vector(s) held")
        return 0
    finally:
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())
