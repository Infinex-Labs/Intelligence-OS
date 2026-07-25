"""Operator interface (§I — Phase 7).

Query memory and run the correction primitives. The query/inspect side doubles as
the privacy transparency tool ("what does the system know about this person", §11);
the mutate side (name/merge/split/cascade-delete) is how re-ID errors and bad
enrollments get fixed — without it, memory corrupts permanently (§7).

Usage:
  python -m intelligence_os.operator list
  python -m intelligence_os.operator inspect <entity_id>
  python -m intelligence_os.operator name <entity_id> "Raj"
  python -m intelligence_os.operator merge <src_id> <dst_id>
  python -m intelligence_os.operator split <entity_id> --sigs s1 s2 --obs o1 o2
  python -m intelligence_os.operator delete <entity_id>
  python -m intelligence_os.operator location <location_id> [--at <epoch>]
"""
from __future__ import annotations

import argparse
import json
import sys
import time

from .scene_state import SceneState
from .store import Store


def _fmt_time(t: float) -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t))


def cmd_list(store: Store, args) -> None:
    for e in store.list_entities(type_=args.type, active_only=not args.all):
        nobs = len(store.observations(e["entity_id"]))
        nrel = len(store.relations(e["entity_id"]))
        print(f"{e['entity_id']}  {e['type']:6s}  {e['label'] or '(unnamed)':16s}  "
              f"status={e['status']:14s}  obs={nobs}  rel={nrel}")


def cmd_inspect(store: Store, args) -> None:
    e = store.get_entity(args.entity_id)
    if not e:
        print("no such entity"); return
    print(f"Entity {e['entity_id']}  type={e['type']}  label={e['label'] or '(unnamed)'}  "
          f"status={e['status']}  created={_fmt_time(e['created_at'])}")
    sigs = {k: len(store.entity_signatures(e['entity_id'], k))
            for k in ("face", "body", "appearance")}
    print(f"  signatures: {sigs}")

    obs = store.observations(e["entity_id"])
    print(f"  observations: {len(obs)}")
    for o in obs[-args.limit:]:
        tgt = f" -> {o['object_entity_id']}" if o["object_entity_id"] else ""
        loc = f" @{o['location_id']}" if o["location_id"] else ""
        print(f"    [{_fmt_time(o['timestamp'])}] {o['predicate']}{tgt}{loc} "
              f"(conf={o['confidence']:.2f}, {o['origin']})")

    rels = store.relations(e["entity_id"])
    print(f"  relations/habits/events: {len(rels)}")
    for r in rels:
        tgt = f" -> {r['object_entity_id']}" if r["object_entity_id"] else ""
        loc = f" @{r['location_id']}" if r["location_id"] else ""
        supp = json.loads(r["supporting_observation_ids"])
        print(f"    [{r['kind']}] {r['predicate']}{tgt}{loc}  weight={r['weight']:.2f}  "
              f"{r['status']}  (evidence: {len(supp)} obs)")


def cmd_name(store: Store, args) -> None:
    store.set_label(args.entity_id, args.label)
    print(f"named {args.entity_id} -> {args.label!r}")


def cmd_merge(store: Store, args) -> None:
    store.merge(args.src, args.dst)
    print(f"merged {args.src} into {args.dst} (re-pointed signatures/obs/relations/snapshots)")


def cmd_split(store: Store, args) -> None:
    new = store.split(args.entity_id, sig_ids=args.sigs or [], observation_ids=args.obs or [])
    print(f"split {args.entity_id}: carved {len(args.sigs or [])} sigs / "
          f"{len(args.obs or [])} obs into new entity {new}")


def cmd_delete(store: Store, args) -> None:
    counts = store.cascade_delete(args.entity_id)
    print(f"cascade-deleted {args.entity_id}: {counts}")


def cmd_dedupe(store: Store, args) -> None:
    merged = store.auto_merge_people(threshold=args.threshold)
    print(f"auto-merged {merged} same-person fragment(s) "
          f"(threshold={args.threshold or 'config default'}; undo with split)")


def cmd_location(store: Store, args) -> None:
    scene = SceneState(store)
    inv = scene.inventory_at(args.location_id, args.at)
    when = _fmt_time(args.at) if args.at else "latest"
    print(f"inventory of {args.location_id} ({when}):")
    for eid in inv:
        e = store.get_entity(eid)
        print(f"  {eid}  {e['label'] or e['type'] if e else '?'}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="intelligence_os.operator")
    sub = p.add_subparsers(dest="cmd", required=True)

    pl = sub.add_parser("list"); pl.add_argument("--type"); pl.add_argument("--all", action="store_true")
    pl.set_defaults(func=cmd_list)
    pi = sub.add_parser("inspect"); pi.add_argument("entity_id"); pi.add_argument("--limit", type=int, default=15)
    pi.set_defaults(func=cmd_inspect)
    pn = sub.add_parser("name"); pn.add_argument("entity_id"); pn.add_argument("label")
    pn.set_defaults(func=cmd_name)
    pm = sub.add_parser("merge"); pm.add_argument("src"); pm.add_argument("dst")
    pm.set_defaults(func=cmd_merge)
    ps = sub.add_parser("split"); ps.add_argument("entity_id")
    ps.add_argument("--sigs", nargs="*"); ps.add_argument("--obs", nargs="*")
    ps.set_defaults(func=cmd_split)
    pd = sub.add_parser("delete"); pd.add_argument("entity_id"); pd.set_defaults(func=cmd_delete)
    pdd = sub.add_parser("dedupe", help="merge same-person fragments by face similarity")
    pdd.add_argument("--threshold", type=float, default=None)
    pdd.set_defaults(func=cmd_dedupe)
    plo = sub.add_parser("location"); plo.add_argument("location_id"); plo.add_argument("--at", type=float)
    plo.set_defaults(func=cmd_location)

    args = p.parse_args(argv)
    store = Store()
    args.func(store, args)
    store.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
