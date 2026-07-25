"""Full pipeline orchestrator (§3 cascade, integrates Phases 1-5).

camera -> [A motion gate] -> [B detect+track] -> [C identity/re-id] ->
[D scene-state] -> [E vlm trigger] -> [F vlm describe] -> [G memory write]

Cheap stages gate expensive ones. The VLM fires only on meaningful change. Run the
distillation pass (Phase 6) separately/nightly via `python -m intelligence_os.distill`.

M6: one thread per camera, shared detector/models. `cameras:` list in config.yaml.

Usage:
  python -m intelligence_os.run --webcam 0 --zones intelligence_os/data/zones.json
  python -m intelligence_os.run --video clip.mp4 --zones zones.json --snapshot-every 30
"""
from __future__ import annotations

import argparse
import sys
import threading
import time
from collections import defaultdict

from .capture import Capture, MotionGate, SettleDetector, _is_stream
from .config import CONFIG, FRAMES_DIR, load_app_config, resolve_cameras
from .detect import Detector, ObjectRegistry
from .identity import FaceEmbedder, IdentityResolver
from .observe import Observer, ResolvedDetection
from .rules import RuleEngine
from .scene_state import SceneState
from .store import Store
from .vlm import Describer, TriggerContext, TriggerDecider


def _scene_signature(resolved: list[ResolvedDetection]) -> str:
    return "|".join(sorted(f"{r.det.cls_name}:{r.location_id}" for r in resolved))


def resolve_source(args) -> dict:
    """§10 no-config boot: fill camera/zones/sensitivity from config.yaml when they
    weren't passed on the CLI. Precedence: CLI flag > config.yaml > default (webcam 0),
    so `python -m intelligence_os.web` boots with nothing to read. Mutates args in place;
    returns the loaded config (for logging).

    M6: also resolves the cameras list onto args._cameras (used by run())."""
    cfg = load_app_config()

    # M6: resolve cameras list from config, but CLI --webcam/--video overrides
    if getattr(args, "webcam", None) is not None:
        args._cameras = [{"name": "default", "source": args.webcam}]
    elif getattr(args, "video", None) is not None:
        args._cameras = [{"name": "default", "source": args.video}]
    else:
        args._cameras = resolve_cameras(cfg)
        # backward compat: set webcam/video from first camera for any code that reads them
        first = args._cameras[0]["source"]
        if isinstance(first, int):
            args.webcam = first
        else:
            args.video = str(first)

    if getattr(args, "zones", None) is None and cfg.get("zones"):
        args.zones = str(cfg["zones"])
    if getattr(args, "sensitivity", None) is None and cfg.get("sensitivity"):
        args.sensitivity = cfg["sensitivity"]
    return cfg


def _draw(img, motion_present, resolved, vlm_calls, vlm_flash, frame_idx, cam_name=None):
    import cv2
    h, w = img.shape[:2]
    cv2.rectangle(img, (0, 0), (w, 30), (0, 0, 0), -1)
    status = "MOTION" if motion_present else "still"
    col = (0, 200, 0) if motion_present else (120, 120, 120)
    prefix = f"[{cam_name}] " if cam_name and cam_name != "default" else ""
    cv2.putText(img, f"{prefix}f{frame_idx} {status} dets:{len(resolved)} vlm:{vlm_calls}",
                (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)
    cv2.putText(img, "q to quit", (w - 95, 21),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    if vlm_flash:
        cv2.putText(img, "VLM described frame", (w // 2 - 110, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
    for r in resolved:
        x1, y1, x2, y2 = r.det.bbox
        person = r.det.cls_name == "person"
        c = (0, 220, 0) if person else (255, 180, 0)
        cv2.rectangle(img, (x1, y1), (x2, y2), c, 2)
        tag = (r.entity_id[-6:] if person else r.det.cls_name)
        cv2.putText(img, tag, (x1 + 2, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 2)
    return img


# ---------------------------------------------------------------------------
# Shared resources loaded once and used by all camera threads
# ---------------------------------------------------------------------------
class _SharedModels:
    """Heavy models that are loaded once and shared across camera threads.
    Detector and FaceEmbedder are thread-safe (Ultralytics/InsightFace run
    under their own locks). Describer is stateless HTTP calls."""
    def __init__(self, store: Store):
        self.store = store
        self.detector = Detector()
        self.face_embedder = FaceEmbedder()
        self.identity = IdentityResolver(store)
        self.objects = ObjectRegistry(store)
        self.people = ObjectRegistry(store, kind="person")   # used when faces are off
        self.observer = Observer(store)
        self.describer = Describer()
        self.rules = RuleEngine(store, verifier=self.describer.verify)
        # Thread lock for detector (Ultralytics is not guaranteed thread-safe)
        self.detect_lock = threading.Lock()


def _camera_loop(cam_name: str, source, shared: _SharedModels,
                 cam_state: dict, args,
                 on_frame=None, show: bool = False) -> None:
    """M6: per-camera capture+detect loop. Each camera has its own Capture,
    MotionGate, SettleDetector, SceneState; heavy models are shared."""
    import cv2
    from .distill import Distiller

    store = shared.store
    realtime = _is_stream(source)
    cap = Capture(source, realtime=realtime)
    cam_state["is_stream"] = cap.is_stream

    gate = MotionGate()
    settle = SettleDetector()
    scene = (SceneState.from_json(store, args.zones) if args.zones
             else SceneState(store))
    trigger = TriggerDecider()

    win = f"Intelligence OS — {cam_name}" if show else None
    if show:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    last_signature = None
    last_snapshot_t = 0.0
    last_distill_t = time.time()
    vlm_calls = 0
    frames_seen = 0
    pending_change = False

    print(f"[run:{cam_name}] source={source!r} zones={'yes' if scene.zones else 'none'} "
          f"vlm={'on' if CONFIG.vlm.enabled else 'off'} show={show}")
    try:
        for frame in cap.frames():
            cam_state["last_frame_ts"] = time.time()
            if cam_state.get("paused", False):
                if on_frame:
                    img = frame.image.copy()
                    cv2.putText(img, "CAMERA PAUSED", (img.shape[1]//2 - 100, img.shape[0]//2),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2)
                    on_frame(img)
                time.sleep(0.1)
                continue

            frames_seen += 1

            # [H] live distillation on a timer (only one camera thread does this)
            interval = CONFIG.distill.live_interval_s
            if interval and (time.time() - last_distill_t) >= interval:
                try:
                    Distiller(store).run()
                except Exception as e:
                    print(f"[run:{cam_name}] live distill failed: {e}")
                last_distill_t = time.time()

            motion = gate.update(frame.image)
            settled = settle.update(motion.present)
            heartbeat = (frame.index % 10 == 0)
            if not motion.present and not settled and not heartbeat:
                img = frame.image.copy()
                _draw(img, False, [], vlm_calls, False, frame.index, cam_name)
                if on_frame:
                    on_frame(img)
                if show:
                    cv2.imshow(win, img)
                    if (cv2.waitKey(1) & 0xFF) in (ord('q'), 27):
                        break
                continue

            def _present(resolved, flash):
                img = frame.image.copy()
                _draw(img, motion.present, resolved, vlm_calls, flash, frame.index, cam_name)
                if on_frame:
                    on_frame(img)
                if not show:
                    return False
                cv2.imshow(win, img)
                return (cv2.waitKey(1) & 0xFF) in (ord('q'), 27)

            # [B] detect + track (shared detector, locked)
            with shared.detect_lock:
                dets = shared.detector.track(frame.image)
            if not dets:
                if _present([], False):
                    break
                continue

            # [C] identity / re-id + [D] scene-state assignment
            resolved: list[ResolvedDetection] = []
            new_entity = False
            for d in dets:
                if d.cls_name == "person" and CONFIG.identity.enabled:
                    x1, y1, x2, y2 = d.bbox
                    crop = frame.image[max(0, y1):y2, max(0, x1):x2]
                    faces = shared.face_embedder.detect(crop) if crop.size else []
                    if faces:
                        m = shared.identity.resolve(faces[0])
                        eid, new_entity = m.entity_id, new_entity or m.minted
                    else:
                        continue
                elif d.cls_name == "person":
                    eid = shared.people.resolve(d)   # track/appearance, no face
                else:
                    eid = shared.objects.resolve(d)
                loc = scene.assign(d.bbox)
                resolved.append(ResolvedDetection(eid, d, loc, camera_id=cam_name))

            if not resolved:
                if _present([], False):
                    break
                continue

            # [rules] cascade stage 3-4: zone/dwell/cooldown gate -> VLM verify
            for ev in shared.rules.feed(resolved, frame.image, frame.timestamp):
                print(f"[rule:{cam_name}] FIRED {ev.rule} · entity={ev.entity_id[-6:]} · "
                      f"{time.strftime('%H:%M:%S', time.localtime(ev.timestamp))} · {ev.keyframe}")
                try:
                    from .deliver import deliver_realtime_event
                    deliver_realtime_event(store, ev)
                except Exception as e:
                    print(f"[deliver] Real-time delivery failed: {e}")

            # [G] memory write
            src_ref = None
            if settled:
                kf = FRAMES_DIR / f"kf_{cam_name}_{frame.index}.jpg"
                if not kf.exists():
                    cv2.imwrite(str(kf), frame.image)
                src_ref = str(kf)
            shared.observer.observe_frame(resolved, frame.timestamp, source_ref=src_ref)

            # [D] periodic scene-state snapshot
            if scene.zones and (frame.timestamp - last_snapshot_t) >= args.snapshot_every:
                present_by_loc = defaultdict(list)
                for r in resolved:
                    if r.location_id:
                        present_by_loc[r.location_id].append(r.entity_id)
                scene.snapshot(present_by_loc, frame.timestamp)
                last_snapshot_t = frame.timestamp

            # [E] VLM trigger decision
            sig = _scene_signature(resolved)
            scene_diff = sig != last_signature
            interaction = any(r.det.cls_name == "person" for r in resolved) and \
                          any(r.det.cls_name != "person" for r in resolved)
            pending_change = pending_change or new_entity or scene_diff or interaction
            ctx = TriggerContext(motion.present, settled, pending_change, scene_diff,
                                 interaction, sig)
            vlm_flash = False
            if trigger.should_fire(ctx, now=frame.timestamp):
                pending_change = False
                vlm_flash = True
                keyframe_path = src_ref or str(FRAMES_DIR / f"kf_{cam_name}_{frame.index}.jpg")
                if not (FRAMES_DIR / f"kf_{cam_name}_{frame.index}.jpg").exists():
                    cv2.imwrite(keyframe_path, frame.image)
                ctx_entities = {}
                for r in resolved:
                    if not r.entity_id.startswith("ent_"):
                        continue
                    ent = store.get_entity(r.entity_id)
                    if not ent:
                        continue
                    rels = [rel for rel in store.relations(r.entity_id) if rel["status"] == "confirmed"]
                    habits = [rel["predicate"] + (f" @{rel['location_id']}" if rel["location_id"] else "")
                              for rel in rels if rel["kind"] == "habit"]
                    relations = [rel["predicate"] + (f" {rel['object_entity_id']}" if rel["object_entity_id"] else "")
                                 for rel in rels if rel["kind"] == "relation"]
                    events = [rel["predicate"] + (f" {rel['object_entity_id']}" if rel["object_entity_id"] else "")
                              for rel in rels if rel["kind"] == "event"]
                    ctx_entities[r.entity_id] = {
                        "label": ent["label"] or "Unknown",
                        "type": ent["type"],
                        "relations": relations,
                        "habits": habits,
                        "recent_events": events
                    }

                desc = shared.describer.describe(frame.image, context={
                    "entities": ctx_entities,
                    "locations": [z.name for z in scene.zones],
                })
                vlm_calls += 1
                if desc:
                    for subj_ref, predicate in desc.states():
                        subj = (resolved[0].entity_id if subj_ref in ("person", "scene")
                                and resolved else subj_ref)
                        if not subj.startswith("ent_"):
                            continue
                        store.add_observation(subj, predicate, confidence=0.6,
                                              source_ref=keyframe_path, origin="vlm",
                                              timestamp=frame.timestamp,
                                              camera_id=cam_name)
            last_signature = sig

            if _present(resolved, vlm_flash):
                break

            if args.max_frames and frame.index >= args.max_frames:
                break
    finally:
        cap.release()
        if show:
            cv2.destroyAllWindows()

    cam_state["frames_seen"] = frames_seen
    cam_state["vlm_calls"] = vlm_calls


def run(args, on_frame=None, state=None) -> None:
    """M6: orchestrate one thread per camera. Shared models load once.
    For single-camera configs this is exactly one thread (no behavior change)."""
    import cv2

    if args.sensitivity:
        CONFIG.trigger.sensitivity = args.sensitivity
    from .distill import prune_old_keyframes
    pruned = prune_old_keyframes()
    if pruned:
        print(f"[run] retention: pruned {pruned} keyframe(s) older than "
              f"{CONFIG.raw_retention_days}d")

    cameras = getattr(args, '_cameras', None)
    if not cameras:
        # fallback for callers that haven't gone through resolve_source yet
        source = args.webcam if args.webcam is not None else args.video
        cameras = [{"name": "default", "source": source}]

    store = Store()
    shared = _SharedModels(store)

    if shared.rules.rules:
        print(f"[run] {len(shared.rules.rules)} rule(s) armed: "
              f"{', '.join(r['name'] for r in shared.rules.rules)}")
        for name, zone in shared.rules.zone_warnings:
            print(f"[run] WARNING rule '{name}' targets zone '{zone}' which is not "
                  f"defined — it can NEVER fire. Add the zone (--zones / draw it) or "
                  f"remove the zone from the rule.")

    # M6: prepare per-camera state
    if state is None:
        state = {}
    if "cameras" not in state:
        state["cameras"] = {}

    show = args.show if args.show is not None else _is_stream(cameras[0]["source"])

    # M9: Start background delivery scheduler thread (PLAN M9.1)
    from .deliver import start_scheduler_thread
    start_scheduler_thread()

    def spawn(cam, solo: bool = False) -> threading.Thread:
        """Start one camera thread against the shared models. Reused for the
        initial cameras and for cameras hot-added later via state['_spawn']."""
        cam_name = cam["name"]
        source = cam["source"]
        cam_state = {"paused": state.get("paused", False), "source": str(source)}
        state["cameras"][cam_name] = cam_state
        # M6: on_frame is per-camera (web.py passes cam_name to pick the right buffer)
        cam_on_frame = (lambda img, _n=cam_name: on_frame(img, cam_name=_n)) if on_frame else None
        t = threading.Thread(
            target=_camera_loop,
            args=(cam_name, source, shared, cam_state, args),
            kwargs={"on_frame": cam_on_frame, "show": show and solo},
            daemon=True,
            name=f"cam-{cam_name}",
        )
        t.start()
        return t

    # expose the spawner so the web layer can add a camera without a restart
    state["_spawn"] = spawn

    threads: list[threading.Thread] = []
    for cam in cameras:
        threads.append(spawn(cam, solo=len(cameras) == 1))

    # Backward compat: expose first camera's state at top level too
    first_cam = cameras[0]["name"]
    if first_cam in state["cameras"]:
        # state dict is mutable, so web.py can read state["last_frame_ts"] etc.
        # We proxy reads from the first camera for backward compat
        state.setdefault("is_stream", _is_stream(cameras[0]["source"]))

    print(f"[run] {len(cameras)} camera(s) started: "
          f"{', '.join(c['name'] for c in cameras)}")

    # Wait for all camera threads to finish
    for t in threads:
        t.join()

    # consolidate same-person fragments formed during this session (reversible)
    if not args.no_auto_merge:
        merged = store.auto_merge_people()
        if merged:
            print(f"[run] auto-merged {merged} same-person fragment(s) "
                  f"(undo with `operator split`)")

    n_people = len(store.list_entities("person"))
    n_objs = len(store.list_entities("object"))
    n_obs = len(store.observations())
    total_frames = sum(state["cameras"][c["name"]].get("frames_seen", 0) for c in cameras)
    total_vlm = sum(state["cameras"][c["name"]].get("vlm_calls", 0) for c in cameras)
    print(f"\n[run] frames={total_frames}  entities: {n_people} people / {n_objs} objects  "
          f"observations={n_obs}  vlm_calls={total_vlm}")
    print("[run] next: `python -m intelligence_os.distill` to mine relations/habits, "
          "`python -m intelligence_os.operator list` to inspect.")
    store.close()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="intelligence_os.run")
    # not required: falls back to config.yaml, then webcam 0 (§10 no-config boot)
    g = p.add_mutually_exclusive_group(required=False)
    g.add_argument("--webcam", type=int)
    g.add_argument("--video", type=str)
    p.add_argument("--zones", type=str, default=None)
    p.add_argument("--snapshot-every", type=float, default=30.0,
                   help="seconds between scene-state snapshots")
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--sensitivity", choices=["lazy", "balanced", "eager"], default=None,
                   help="VLM trigger sensitivity (default: lazy from config; "
                        "use 'balanced' for a responsive live demo)")
    p.add_argument("--no-auto-merge", action="store_true",
                   help="skip end-of-session consolidation of same-person fragments")
    p.add_argument("--show", dest="show", action="store_true", default=None,
                   help="live preview window (default: on for --webcam)")
    p.add_argument("--no-show", dest="show", action="store_false")
    args = p.parse_args(argv)
    resolve_source(args)
    run(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
