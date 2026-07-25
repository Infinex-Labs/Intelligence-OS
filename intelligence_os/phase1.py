"""Phase 1 runner — Capture + Identity foundation (§4 Phase 1, §12).

Proves the make-or-break layer: the same person stays one entity across runs/days
and two people stay distinct, persisted across restart.

Usage:
  # Watch a webcam and accumulate identities into memory.db
  python -m intelligence_os.phase1 run --webcam 0

  # Run over a recorded clip
  python -m intelligence_os.phase1 run --video test_scripts/test.mp4

  # Inspect what identities memory has formed
  python -m intelligence_os.phase1 entities

  # Tune: report the cosine-similarity matrix between formed entities
  python -m intelligence_os.phase1 separation
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict

import numpy as np

from .capture import Capture, MotionGate
from .config import CONFIG
from .identity import FaceEmbedder, IdentityResolver
from .store import Store


def _short(eid: str) -> str:
    return eid[-6:]


def _draw_overlay(img, motion_present, results, processed_this_frame, frame_idx):
    import cv2
    h, w = img.shape[:2]
    # header bar
    cv2.rectangle(img, (0, 0), (w, 30), (0, 0, 0), -1)
    status = "MOTION" if motion_present else "still"
    color = (0, 200, 0) if motion_present else (120, 120, 120)
    cv2.putText(img, f"frame {frame_idx}  {status}  faces:{len(results)}",
                (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    cv2.putText(img, "press q to quit", (w - 150, 21),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    for det, m in results:
        x1, y1, x2, y2 = det.bbox
        box_col = (0, 165, 255) if m.minted else (0, 220, 0)  # orange=new, green=known
        cv2.rectangle(img, (x1, y1), (x2, y2), box_col, 2)
        tag = ("NEW " if m.minted else "") + _short(m.entity_id) + f" {m.confidence:.2f}"
        cv2.rectangle(img, (x1, y1 - 20), (x1 + 11 * len(tag), y1), box_col, -1)
        cv2.putText(img, tag, (x1 + 2, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    return img


def run(args) -> None:
    import cv2
    store = Store()
    embedder = FaceEmbedder()
    resolver = IdentityResolver(store, threshold=args.threshold)
    gate = MotionGate()

    source = args.webcam if args.webcam is not None else args.video
    realtime = args.webcam is not None
    cap = Capture(source, realtime=realtime)

    # Show a live window by default for a webcam (so you can SEE it work); off by
    # default for a video file. --show / --no-show overrides.
    show = args.show if args.show is not None else realtime

    seen_counts: dict[str, int] = defaultdict(int)
    minted = 0
    processed = 0
    skipped_no_motion = 0
    win = "Intelligence OS — phase1 identity"
    if show:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    print(f"[phase1] source={source!r} threshold={resolver.threshold} "
          f"show={show} (motion-gated face identity; q to quit)")
    try:
        for frame in cap.frames():
            # Always feed the gate (cheap) so the background model stays current.
            motion = gate.update(frame.image)
            # Run identity on motion OR as a periodic heartbeat, so a person who
            # is sitting STILL (no motion) is still recognized instead of the
            # window looking frozen. Heartbeat every ~10 frames.
            heartbeat = (frame.index % 10 == 0)
            do_detect = motion.present or heartbeat
            if not do_detect:
                skipped_no_motion += 1
                if show:
                    _draw_overlay(frame.image, motion.present, [], False, frame.index)
                    cv2.imshow(win, frame.image)
                    if (cv2.waitKey(1) & 0xFF) in (ord('q'), 27):
                        break
                continue

            faces = embedder.detect(frame.image)
            results = []
            if faces:
                processed += 1
                for det in faces:
                    m = resolver.resolve(det)
                    results.append((det, m))
                    seen_counts[m.entity_id] += 1
                    if m.minted:
                        minted += 1
                        print(f"  frame {frame.index}: MINTED {m.entity_id} "
                              f"(best sim {m.confidence:.3f})")
                    elif args.verbose:
                        print(f"  frame {frame.index}: matched {m.entity_id} "
                              f"(sim {m.confidence:.3f})")

            if show:
                _draw_overlay(frame.image, motion.present, results, bool(faces), frame.index)
                cv2.imshow(win, frame.image)
                if (cv2.waitKey(1) & 0xFF) in (ord('q'), 27):
                    break

            if args.max_frames and frame.index >= args.max_frames:
                break
    finally:
        cap.release()
        if show:
            cv2.destroyAllWindows()

    print(f"\n[phase1] done. processed {processed} frames with faces, "
          f"{skipped_no_motion} skipped (no motion).")
    print(f"[phase1] entities now in memory: {len(store.list_entities('person'))} "
          f"({minted} minted this run)")
    for eid, n in sorted(seen_counts.items(), key=lambda x: -x[1]):
        ent = store.get_entity(eid)
        label = ent["label"] or "(unnamed)"
        print(f"  {eid}  {label:18s}  seen {n}x  "
              f"signatures={len(store.entity_signatures(eid, 'face'))}")
    store.close()


def entities(args) -> None:
    store = Store()
    rows = store.list_entities(active_only=not args.all)
    if not rows:
        print("(no entities yet)")
        return
    for e in rows:
        nsig = len(store.entity_signatures(e["entity_id"], "face"))
        nobs = len(store.observations(e["entity_id"]))
        print(f"{e['entity_id']}  type={e['type']:6s} "
              f"label={e['label'] or '-':12s} status={e['status']:14s} "
              f"sigs={nsig} obs={nobs}")
    store.close()


def separation(args) -> None:
    """Print the cross-entity cosine-similarity matrix using each entity's mean
    face signature. Distinct people should be LOW; this is how you tune the
    threshold and verify two people stay separate (§12 Phase-1)."""
    store = Store()
    people = store.list_entities("person")
    means = []
    ids = []
    for e in people:
        sigs = store.entity_signatures(e["entity_id"], "face")
        if not sigs:
            continue
        mean = np.mean(np.stack(sigs), axis=0)
        mean = mean / (np.linalg.norm(mean) or 1.0)
        means.append(mean)
        ids.append(e["entity_id"])
    if len(ids) < 2:
        print("need >=2 entities with face signatures to compare")
        store.close()
        return
    print("cross-entity cosine similarity (lower = better separated):")
    print("        " + "  ".join(i[-6:] for i in ids))
    for i, a in enumerate(means):
        row = "  ".join(f"{float(np.dot(a, b)):+.2f}" for b in means)
        print(f"{ids[i][-6:]:8s}{row}")
    store.close()


def _greedy_cluster(embs: list, threshold: float) -> int:
    """Count clusters: each embedding joins the first existing cluster whose
    centroid is within threshold (cosine), else starts a new one. Mirrors the
    online match-or-mint, so the count predicts how many entities that threshold
    would form."""
    centroids: list = []
    for e in embs:
        best, bi = -1.0, -1
        for i, c in enumerate(centroids):
            s = float(np.dot(e, c))
            if s > best:
                best, bi = s, i
        if best >= threshold:
            centroids[bi] = centroids[bi] + (e - centroids[bi]) * 0.3
            centroids[bi] = centroids[bi] / (np.linalg.norm(centroids[bi]) or 1.0)
        else:
            centroids.append(e.copy())
    return len(centroids)


def tune(args) -> None:
    """Extract face embeddings from a clip once, then sweep the match threshold
    so you can pick the value that yields ~the true number of people (§12)."""
    from .capture import Capture, MotionGate
    embedder = FaceEmbedder()
    gate = MotionGate()
    cap = Capture(args.video)
    embs = []
    print(f"[tune] extracting face embeddings from {args.video} (every {args.stride} frames)...")
    for frame in cap.frames():
        gate.update(frame.image)
        if frame.index % args.stride != 0:
            continue
        for det in embedder.detect(frame.image):
            embs.append(det.embedding)
        if args.max_frames and frame.index >= args.max_frames:
            break
    cap.release()
    print(f"[tune] {len(embs)} face embeddings extracted.")
    if not embs:
        return
    print("[tune] threshold -> #entities formed (pick the one matching #people you know are in the clip):")
    for thr in [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:
        print(f"  threshold {thr:.2f} -> {_greedy_cluster(embs, thr):3d} entities")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="intelligence_os.phase1")
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("tune", help="sweep match threshold over a clip's faces")
    t.add_argument("--video", required=True)
    t.add_argument("--stride", type=int, default=5)
    t.add_argument("--max-frames", type=int, default=0)
    t.set_defaults(func=tune)

    r = sub.add_parser("run", help="capture + identity into memory")
    g = r.add_mutually_exclusive_group(required=True)
    g.add_argument("--webcam", type=int, help="webcam index, e.g. 0")
    g.add_argument("--video", type=str, help="path to a video file")
    r.add_argument("--threshold", type=float, default=None,
                   help=f"face match cosine threshold (default {CONFIG.identity.face_match_threshold})")
    r.add_argument("--max-frames", type=int, default=0)
    r.add_argument("--verbose", action="store_true")
    r.add_argument("--show", dest="show", action="store_true", default=None,
                   help="show a live preview window (default: on for --webcam)")
    r.add_argument("--no-show", dest="show", action="store_false",
                   help="force headless")
    r.set_defaults(func=run)

    e = sub.add_parser("entities", help="list formed entities")
    e.add_argument("--all", action="store_true", help="include merged/deleted")
    e.set_defaults(func=entities)

    s = sub.add_parser("separation", help="cross-entity similarity matrix")
    s.set_defaults(func=separation)

    args = p.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
