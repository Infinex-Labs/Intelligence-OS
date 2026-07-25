"""Periodic distillation — change detection, relations, habits (§H — Phase 6).

The batch pass that turns an observation log into knowledge. Runs on a schedule
(e.g. nightly). This is where "this bedsheet seems new" and "they use the laptop"
finally emerge — as WEIGHTED, DECAYING, AUDITABLE edges, never as flat facts.

Order matters (§H):
  0. normalize raw open-vocabulary predicates -> canonical (else aggregation scatters)
  1. change detection: diff scene-state inventories over time -> new/changed/gone
  2. event reasoning (VLM reasoner, optional): change+context -> events ('acquired')
  3. habit mining: recurring temporal patterns -> habits
  4. relation mining: co-occurrence/proximity over time -> relations (uses/frequents)

Corroboration gating (NFR-4): a single overconfident VLM call never becomes a
confirmed fact. Events stay 'candidate' until the change is independently
supported (scene-diff agreement). Relations confirm only past the weight threshold,
which requires repeated evidence by construction (§7).
"""
from __future__ import annotations

import json
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Optional

from .config import CONFIG, FRAMES_DIR
from .store import Store

DAY = 86400.0


def prune_old_keyframes(days: Optional[float] = None) -> int:
    """Drop raw keyframes older than the retention window (§11: raw frames kept
    only as long as needed for audit, then dropped; memory keeps the gist)."""
    days = CONFIG.raw_retention_days if days is None else days
    if not FRAMES_DIR.exists():
        return 0
    cutoff = time.time() - days * DAY
    n = 0
    for f in FRAMES_DIR.glob("*.jpg"):
        if f.stat().st_mtime < cutoff:
            f.unlink()
            n += 1
    return n

# Built-in normalization seeds (the table GROWS as new predicates are canonicalized).
CANONICAL_SEED = {
    "asleep": "sleeping", "napping": "sleeping", "eyes_closed": "sleeping",
    "on_phone": "using_phone", "on phone": "using_phone",
    "state:sleeping": "sleeping", "state:asleep": "sleeping",
}


@dataclass
class ChangeEvent:
    location_id: str
    entity_id: str
    kind: str          # new | gone
    first_seen: float
    evidence: list[str]


class Distiller:
    def __init__(self, store: Store, reasoner=None):
        self.store = store
        self.reasoner = reasoner  # optional callable(change, context)->dict

    # --- 0. normalization ----------------------------------------------------
    def normalize(self) -> None:
        for raw, canon in CANONICAL_SEED.items():
            self.store.register_predicate(raw, canonical=canon)

    def _canon(self, predicate: str) -> str:
        return self.store.canonical_predicate(predicate)

    # --- 1. change detection -------------------------------------------------
    def change_detection(self, gone_tail: int = 3) -> list[ChangeEvent]:
        """Walk each location's inventory snapshots chronologically.
        'new'  = the first snapshot an entity appears in (after the baseline
                 snapshot) when it was absent from every earlier snapshot.
        'gone' = present in some earlier snapshot but absent across the last
                 `gone_tail` snapshots (absent for sufficient observations)."""
        events: list[ChangeEvent] = []
        for loc in self.store.locations():
            lid = loc["location_id"]
            snaps = self.store.snapshots(location_id=lid)
            if len(snaps) < 2:
                continue
            inv = [set(json.loads(s["present_entity_ids"])) for s in snaps]
            seen_before: set = set(inv[0])  # baseline; appearances here aren't "new"

            for i in range(1, len(snaps)):
                for eid in inv[i] - seen_before:
                    events.append(ChangeEvent(lid, eid, "new", snaps[i]["timestamp"],
                                              [snaps[i]["snapshot_id"]]))
                seen_before |= inv[i]

            ever = set().union(*inv)
            tail = inv[-gone_tail:]
            tail_present = set().union(*tail) if tail else set()
            for eid in ever - tail_present:
                events.append(ChangeEvent(lid, eid, "gone", snaps[-1]["timestamp"],
                                          [s["snapshot_id"] for s in snaps[-gone_tail:]]))
        return events

    # --- 2. event reasoning (corroboration-gated) ----------------------------
    def reason_events(self, changes: list[ChangeEvent]) -> list[str]:
        """Turn change candidates into event edges. Deterministic mapping
        (new object -> 'acquired', gone object -> 'removed') with an OPTIONAL LLM
        reasoner to refine. Always written as 'candidate' until corroborated."""
        rel_ids = []
        for ch in changes:
            ent = self.store.get_entity(ch.entity_id)
            if ent is None:
                continue
            predicate = None
            if ent["type"] == "object":
                predicate = "acquired" if ch.kind == "new" else "removed"
            elif ent["type"] == "person" and ch.kind == "new":
                predicate = "appeared"
            if predicate is None:
                continue
            if self.reasoner is not None:
                refined = self.reasoner(ch, {"entity": dict(ent)})
                if refined and refined.get("predicate"):
                    predicate = refined["predicate"]
            # low obs_confidence -> stays candidate; needs repeated evidence to confirm
            rid = self.store.reinforce_relation(
                "event", ch.entity_id, predicate, location_id=ch.location_id,
                supporting_observation_ids=ch.evidence, obs_confidence=0.5)
            rel_ids.append(rid)
        return rel_ids

    # --- 3. habit mining -----------------------------------------------------
    def mine_habits(self, min_days: int = 2) -> list[str]:
        """Recurring temporal patterns: an entity repeatedly present in a location
        within the same hour-of-day bucket across >= min_days distinct days."""
        rel_ids = []
        obs = [o for o in self.store.observations() if o["predicate"] == "present"
               and o["location_id"]]
        # (entity, location, hour-bucket) -> set of day indices
        buckets: dict[tuple, set] = defaultdict(set)
        evidence: dict[tuple, list] = defaultdict(list)
        for o in obs:
            t = o["timestamp"]
            day = int(t // DAY)
            hour = time.gmtime(t).tm_hour
            key = (o["subject_entity_id"], o["location_id"], hour)
            buckets[key].add(day)
            evidence[key].append(o["observation_id"])
        for (eid, lid, hour), days in buckets.items():
            if len(days) >= min_days:
                rid = self.store.reinforce_relation(
                    "habit", eid, f"present_around_{hour:02d}h",
                    location_id=lid, supporting_observation_ids=evidence[(eid, lid, hour)],
                    obs_confidence=min(1.0, len(days) / 5.0))
                rel_ids.append(rid)
        return rel_ids

    # --- 4. relation mining --------------------------------------------------
    def mine_relations(self, min_count: int = 3) -> list[str]:
        """Co-occurrence/proximity over time -> relations.
        'near' (person,object) repeated -> 'uses'; frequent presence in a location
        -> 'frequents'."""
        rel_ids = []
        near_counts: Counter = Counter()
        near_evidence: dict[tuple, list] = defaultdict(list)
        freq_counts: Counter = Counter()
        freq_evidence: dict[tuple, list] = defaultdict(list)

        for o in self.store.observations():
            pred = self._canon(o["predicate"])
            if pred == "near" and o["object_entity_id"]:
                key = (o["subject_entity_id"], o["object_entity_id"])
                near_counts[key] += 1
                near_evidence[key].append(o["observation_id"])
            elif pred == "present" and o["location_id"]:
                key = (o["subject_entity_id"], o["location_id"])
                freq_counts[key] += 1
                freq_evidence[key].append(o["observation_id"])

        for (subj, obj), n in near_counts.items():
            if n < min_count:
                continue
            for _ in range(n):  # weight grows with evidence count
                rid = self.store.reinforce_relation(
                    "relation", subj, "uses", object_entity_id=obj,
                    supporting_observation_ids=near_evidence[(subj, obj)],
                    obs_confidence=0.8)
            rel_ids.append(rid)

        for (subj, lid), n in freq_counts.items():
            if n < min_count:
                continue
            for _ in range(n):
                rid = self.store.reinforce_relation(
                    "relation", subj, "frequents", location_id=lid,
                    supporting_observation_ids=freq_evidence[(subj, lid)],
                    obs_confidence=0.7)
            rel_ids.append(rid)
        return rel_ids

    # --- orchestration -------------------------------------------------------
    def run(self) -> dict:
        self.store.register_predicate("uses", "person", "object")
        self.store.register_predicate("frequents", "person", "any")
        self.store.register_predicate("acquired", "object", "any")
        self.store.register_predicate("removed", "object", "any")
        self.normalize()
        self.store.decay_relations()          # age existing edges first (§7)
        changes = self.change_detection()
        events = self.reason_events(changes)
        habits = self.mine_habits()
        relations = self.mine_relations()
        pruned = prune_old_keyframes()
        return {
            "changes": len(changes),
            "events": len(events),
            "habits": len(habits),
            "relations": len(relations),
            "keyframes_pruned": pruned,
        }


def main(argv=None) -> int:
    """Run the nightly distillation pass over the live memory.db."""
    store = Store()
    result = Distiller(store).run()
    print(f"[distill] {result}")
    confirmed = [r for r in store.relations() if r["status"] == "confirmed"]
    print(f"[distill] {len(confirmed)} confirmed edges (of {len(store.relations())} total):")
    for r in confirmed[:20]:
        tgt = f" -> {r['object_entity_id']}" if r["object_entity_id"] else ""
        loc = f" @{r['location_id']}" if r["location_id"] else ""
        print(f"  [{r['kind']}] {r['subject_entity_id']} {r['predicate']}{tgt}{loc} "
              f"w={r['weight']:.2f}")
    store.close()
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
