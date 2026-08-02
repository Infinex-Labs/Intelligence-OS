"""The digest — graph diff vs. learned habits (FinalPRD §8.1 — M2).

"Since you last looked": three bands, most severe first.

  rule_fired  — compiled rules that fired (origin='rule' observations)
  unusual     — deviations from the distilled baseline: first-ever entities,
                presence at an hour with no matching habit, new mined events
  routine     — everything else, collapsed to counts

Every item carries a signature (stable id), the entity it's about, and a keyframe
when one exists. Corrections are observations (predicate 'digest:dismissed:<sig>'
or 'digest:confirmed:<sig>', origin='operator') — the same provenance machinery
as everything else, and the training data for M5's salience learning. Dismissed
and confirmed items stop appearing: the digest is a triage queue, not a log.

All deviation checks are cheap heuristics over existing rows — no new ML. The
'Not interesting' signal is what eventually makes salience learned, not computed.
"""
from __future__ import annotations

import argparse
import time
from collections import defaultdict
from typing import Optional

from .ask import _kf
from .distill import bucket_hour
from .store import Store


def build(store: Store, since: float, now: Optional[float] = None) -> dict:
    now = now or time.time()
    locs = {r["location_id"]: r["name"] for r in store.locations()}
    triaged = _triaged_signatures(store)

    rows = [o for o in store.observations(since=since) if o["timestamp"] <= now]
    rule_items, unusual, routine_count = [], [], 0

    # --- band 1: rule fired ---------------------------------------------------
    for o in rows:
        if o["origin"] != "rule":
            continue
        rule = o["predicate"].removeprefix("rule_fired:")
        sig = f"rule:{rule}:{o['subject_entity_id']}:{int(o['timestamp'])}"
        if sig in triaged:
            continue
        rule_items.append(_item(store, sig, o["subject_entity_id"],
                                title=f"Rule fired: {rule}",
                                timestamp=o["timestamp"],
                                location=locs.get(o["location_id"]),
                                keyframe=_kf(o["source_ref"])))

    # --- band 2a: first-ever entities ------------------------------------------
    for e in store.list_entities():
        if e["created_at"] >= since and e["created_at"] <= now:
            sig = f"first_seen:{e['entity_id']}"
            if sig in triaged:
                continue
            eobs = store.observations(e["entity_id"])
            first_obs = next(iter(eobs), None)
            # evidence: earliest of the entity's own keyframed observations, else
            # any scene keyframe within ±10s of first sighting (keyframes are only
            # written on settled frames, so many entities have none of their own)
            with_kf = next((o for o in eobs if o["source_ref"]), None)
            if with_kf is None and first_obs is not None:
                with_kf = next((o for o in rows if o["source_ref"]
                                and abs(o["timestamp"] - first_obs["timestamp"]) <= 10), None)
            unusual.append(_item(store, sig, e["entity_id"],
                                 title="First time seen",
                                 timestamp=e["created_at"],
                                 location=locs.get(first_obs["location_id"]) if first_obs else None,
                                 keyframe=_kf(with_kf["source_ref"]) if with_kf else None))

    # --- band 2b: presence at an hour with no matching habit -------------------
    # An entity that HAS habit history at a location (the baseline exists) but is
    # present in an hour bucket none of its habits cover. No habits at all -> no
    # baseline -> not flagged (the empty-week rule: value before baseline, §13).
    habit_hours: dict[tuple, set] = defaultdict(set)   # (eid, lid) -> {hours}
    for r in store.relations(kind="habit"):
        # `present_around_14h` only. Phase 5 also mines `present_tue_around_14h`,
        # which does not carry this prefix and so is skipped rather than
        # mis-parsed — deliberate: a weekday habit says nothing about whether
        # *this* hour is unusual, and folding it in would make every Tuesday
        # regular look like a baseline for Wednesday.
        if r["predicate"].startswith("present_around_") and r["location_id"]:
            habit_hours[(r["subject_entity_id"], r["location_id"])].add(
                int(r["predicate"][len("present_around_"):-1]))
    flagged: set[tuple] = set()
    for o in rows:
        if o["predicate"] != "present" or not o["location_id"]:
            continue
        key = (o["subject_entity_id"], o["location_id"])
        hours = habit_hours.get(key)
        h = bucket_hour(o["timestamp"])
        if not hours or (key, h) in flagged:
            continue
        if any(abs(h - hh) <= 1 or abs(h - hh) >= 23 for hh in hours):  # ±1h, wraps
            continue
        flagged.add((key, h))
        sig = f"unusual_hour:{o['subject_entity_id']}:{o['location_id']}:{h:02d}"
        if sig in triaged:
            continue
        unusual.append(_item(store, sig, o["subject_entity_id"],
                             title=f"Present at unusual hour ({h:02d}h local; "
                                   f"habit hours: {sorted(hours)})",
                             timestamp=o["timestamp"],
                             location=locs.get(o["location_id"]),
                             keyframe=_kf(o["source_ref"])))

    # --- band 2c: newly mined events (acquired / removed) ----------------------
    for r in store.relations(kind="event"):
        if not (since <= r["created_at"] <= now):
            continue
        sig = f"event:{r['relation_id']}"
        if sig in triaged:
            continue
        unusual.append(_item(store, sig, r["subject_entity_id"],
                             title=f"{r['predicate'].capitalize()} ({r['status']})",
                             timestamp=r["created_at"],
                             location=locs.get(r["location_id"]),
                             keyframe=None))

    # --- band 3: routine = the rest, collapsed ---------------------------------
    routine_count = sum(1 for o in rows if o["origin"] != "rule")
    entities_seen = len({o["subject_entity_id"] for o in rows})

    # --- histogram calculation (PLAN M8.2) -------------------------------------
    num_buckets = 30
    bucket_width = (now - since) / num_buckets if now > since else 1.0
    obs_by_bucket = defaultdict(int)
    for o in rows:
        b_idx = min(num_buckets - 1, max(0, int((o["timestamp"] - since) / bucket_width)))
        obs_by_bucket[b_idx] += 1

    fired_by_bucket = set()
    for item in rule_items:
        b_idx = min(num_buckets - 1, max(0, int((item["timestamp"] - since) / bucket_width)))
        fired_by_bucket.add(b_idx)

    unusual_by_bucket = set()
    for item in unusual:
        b_idx = min(num_buckets - 1, max(0, int((item["timestamp"] - since) / bucket_width)))
        unusual_by_bucket.add(b_idx)

    histogram = []
    for i in range(num_buckets):
        count = obs_by_bucket[i]
        b_type = "r"
        if i in fired_by_bucket:
            b_type = "f"
        elif i in unusual_by_bucket:
            b_type = "u"
        histogram.append({
            "count": count,
            "type": b_type
        })

    merge_sug = get_merge_suggestions(store)

    unusual.sort(key=lambda i: i["timestamp"], reverse=True)
    rule_items.sort(key=lambda i: i["timestamp"], reverse=True)
    return {
        "since": since, "now": now,
        "rule_fired": rule_items,
        "unusual": unusual,
        "routine": {"observations": routine_count, "entities": entities_seen},
        "histogram": histogram,
        "merge_suggestions": merge_sug
    }


def _item(store, sig, eid, *, title, timestamp, location=None, keyframe=None) -> dict:
    e = store.get_entity(eid)
    label = (e["label"] if e and e["label"]
             else f"{(e['type'] if e else 'entity').capitalize()} {eid[-6:]}")
    return {"signature": sig, "entity_id": eid, "label": label, "title": title,
            "timestamp": timestamp, "location": location, "keyframe": keyframe}


def _triaged_signatures(store: Store) -> set:
    out = set()
    for o in store.observations():
        if o["predicate"].startswith(("digest:dismissed:", "digest:confirmed:")):
            out.add(o["predicate"].split(":", 2)[2])
    return out


def get_merge_suggestions(store: Store, threshold: Optional[float] = None) -> list[dict]:
    """Find person entities with similar signatures to suggest merging (PLAN M8.6)."""
    import numpy as np
    from intelligence_os.config import CONFIG
    threshold = threshold or getattr(getattr(CONFIG, "identity", None), "face_match_threshold", 0.7)
    people = store.list_entities("person")   # already active-only; Rows have no .get
    sigs = {e["entity_id"]: store.entity_signatures(e["entity_id"], "face")
            for e in people}
    ids = [e["entity_id"] for e in people if sigs[e["entity_id"]]]

    suggestions = []
    seen = set()
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a, b = ids[i], ids[j]
            best = max((float(np.dot(x, y)) for x in sigs[a] for y in sigs[b]),
                       default=-1.0)
            if best >= threshold:
                pair = tuple(sorted([a, b]))
                if pair not in seen:
                    seen.add(pair)
                    la = store.get_entity(a)
                    lb = store.get_entity(b)
                    label_a = la["label"] if la and la["label"] else f"Person {a[-6:]}"
                    label_b = lb["label"] if lb and lb["label"] else f"Person {b[-6:]}"
                    suggestions.append({
                        "source_id": a,
                        "source_label": label_a,
                        "target_id": b,
                        "target_label": label_b,
                        "confidence": round(best, 2)
                    })
    return suggestions


def feedback(store: Store, signature: str, entity_id: str, action: str,
             user_id: Optional[str] = None) -> str:
    """Record a triage action with optional user attribution. 'dismiss' = not
    interesting / false positive (the salience training signal); 'confirm' =
    true positive. Both remove the item from future digests."""
    if action not in ("dismiss", "confirm"):
        raise ValueError(f"unknown digest action: {action!r}")
    kind = "dismissed" if action == "dismiss" else "confirmed"
    return store.add_observation(entity_id, f"digest:{kind}:{signature}",
                                 confidence=1.0, origin="operator", user_id=user_id)


def render_text(d: dict) -> str:
    t = lambda ts: time.strftime("%a %H:%M", time.localtime(ts))
    lines = [f"Digest · {t(d['since'])} -> {t(d['now'])}"]
    for band, items in (("RULE FIRED", d["rule_fired"]), ("UNUSUAL", d["unusual"])):
        lines.append(f"\n{band} — {len(items)} item(s)")
        for i in items:
            loc = f" @ {i['location']}" if i["location"] else ""
            kf = f"  [{i['keyframe']}]" if i["keyframe"] else ""
            lines.append(f"  {t(i['timestamp'])}  {i['label']}{loc} — {i['title']}{kf}")
    r = d["routine"]
    lines.append(f"\nROUTINE — {r['observations']} observation(s), "
                 f"{r['entities']} entit(y/ies)")
    return "\n".join(lines)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="intelligence_os.digest")
    p.add_argument("--since-hours", type=float, default=24.0)
    args = p.parse_args(argv)
    store = Store()
    try:
        print(render_text(build(store, since=time.time() - args.since_hours * 3600)))
    finally:
        store.close()
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
