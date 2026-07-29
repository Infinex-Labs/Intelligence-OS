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

from .config import CONFIG, FRAMES_DIR, kind_for_class
from .store import Store

DAY = 86400.0

# --- when a habit happened (search plan Phase 5, gap G8) ---------------------
#
# Habits were bucketed in UTC and rendered in local time, which is a bug you
# cannot see until the two disagree: a warehouse in Delhi mining "present around
# 09h" from footage everyone there remembers as half past two. The buckets are
# a claim about the *working day*, and the working day is local.
#
# So the deployment's local clock is the one definition, and it lives here
# because habit mining is what cuts the buckets. `digest.py` compares an
# observation's hour against those buckets and so must cut them the same way;
# `ask.py` groups sightings by weekday and must agree with both. One function,
# three callers — that is the whole point of it being here rather than inlined.
#
# `time.localtime` reads TZ, so a deployment states its timezone the same way
# every other unix service does, and a test pins it the same way too.
WEEKDAY_ABBR = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")


def bucket_hour(ts: float) -> int:
    """Hour-of-day a timestamp falls in, on the deployment's clock."""
    return time.localtime(ts).tm_hour


def bucket_weekday(ts: float) -> str:
    """Weekday a timestamp falls on, as 'tue'. Local, for the same reason."""
    return WEEKDAY_ABBR[time.localtime(ts).tm_wday]


def bucket_day(ts: float) -> str:
    """The calendar day, as 'YYYY-MM-DD'.

    A local date string, not `int(ts // 86400)`. The integer version counts UTC
    days, so an evening sighting in a positive-offset zone lands on tomorrow —
    which silently splits one habit across two "days" and can push a real
    pattern under the `min_days` floor.
    """
    return time.strftime("%Y-%m-%d", time.localtime(ts))


# What repeated proximity is allowed to become, per kind of thing (§7: the
# distiller interprets, but only as far as the evidence reaches). Proximity to a
# laptop is use; proximity to a dog is company, not use. Anything unmapped keeps
# the original verb, so a new kind cannot silently acquire a claim.
PROXIMITY_VERB = {"object": "uses", "vehicle": "uses", "animal": "accompanied_by"}


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
        self._verb_cache: dict[str, str] = {}   # object entity -> proximity verb

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
    def mine_habits(self, min_days: int = 2, *, since: Optional[float] = None,
                    until: Optional[float] = None) -> list[str]:
        """Recurring temporal patterns, at two granularities.

            present_around_14h        this hour, on any day
            present_tue_around_14h    this hour, on Tuesdays

        Both are mined, and the coarse one is not a summary of the fine ones —
        it is the answer to a different question. "Does the van come by in the
        afternoon" is about the hour; "does it come on Tuesdays" is about the
        week, and a van that visits every weekday at 14:00 has a strong hour
        habit and no weekday habit at all. Reporting only the fine buckets would
        make every daily pattern look like five weak weekly ones.

        The weekday predicate is a NEW string, so nothing that reads the old one
        changes meaning: `digest.py` matches the `present_around_` prefix, which
        `present_tue_around_14h` does not have, and the weekday habits are simply
        invisible to it rather than mis-parsed.

        `since`/`until` bound the scan (G9). Without them a nightly pass re-reads
        all of history to re-derive edges it already has — the cost grows with
        the age of the deployment, on a job that runs forever.
        """
        rel_ids = []
        obs = [o for o in self.store.observations(
                   since=since, until=until, predicate_prefixes=["present"])
               if o["predicate"] == "present" and o["location_id"]]
        # (entity, location, predicate) -> set of local calendar days
        buckets: dict[tuple, set] = defaultdict(set)
        evidence: dict[tuple, list] = defaultdict(list)
        for o in obs:
            t = o["timestamp"]
            day, hour = bucket_day(t), bucket_hour(t)
            for predicate in (f"present_around_{hour:02d}h",
                              f"present_{bucket_weekday(t)}_around_{hour:02d}h"):
                key = (o["subject_entity_id"], o["location_id"], predicate)
                buckets[key].add(day)
                evidence[key].append(o["observation_id"])
        for key, days in buckets.items():
            if len(days) >= min_days:
                eid, lid, predicate = key
                rid = self.store.reinforce_relation(
                    "habit", eid, predicate, location_id=lid,
                    supporting_observation_ids=evidence[key],
                    obs_confidence=min(1.0, len(days) / 5.0))
                rel_ids.append(rid)
        return rel_ids

    def _proximity_verb(self, object_entity_id: str) -> str:
        """Which verb repeated `near` earns, from what the object actually is.

        The detected class is carried as the object entity's label (detect.py mints
        it that way), so a renamed entity falls back to the generic 'uses' — an
        under-claim, which is the safe direction.
        """
        verb = self._verb_cache.get(object_entity_id)
        if verb is None:
            ent = self.store.get_entity(object_entity_id)
            kind = kind_for_class((ent["label"] or "").lower() if ent else "")
            verb = self._verb_cache[object_entity_id] = PROXIMITY_VERB.get(kind, "uses")
        return verb

    # --- 4. relation mining --------------------------------------------------
    def mine_relations(self, min_count: int = 3, *, since: Optional[float] = None,
                       until: Optional[float] = None) -> list[str]:
        """Co-occurrence/proximity over time -> relations.
        'near' repeated -> 'uses' for a thing, 'accompanied_by' for an animal;
        frequent presence in a location -> 'frequents'.

        `since`/`until` bound the scan, for the same reason `mine_habits` takes
        them: a nightly job must not get slower every night forever (G9).
        """
        self._verb_cache.clear()   # labels can change between passes
        rel_ids = []
        near_counts: Counter = Counter()
        near_evidence: dict[tuple, list] = defaultdict(list)
        freq_counts: Counter = Counter()
        freq_evidence: dict[tuple, list] = defaultdict(list)

        for o in self.store.observations(since=since, until=until):
            pred = self._canon(o["predicate"])
            if pred == "near" and o["object_entity_id"]:
                key = (o["subject_entity_id"], o["object_entity_id"])
                near_counts[key] += 1
                near_evidence[key].append(o["observation_id"])
            elif pred == "present" and o["location_id"]:
                key = (o["subject_entity_id"], o["location_id"])
                freq_counts[key] += 1
                freq_evidence[key].append(o["observation_id"])

        # One write per edge, weighted by the evidence count (G9). This used to
        # be `for _ in range(n)`, which issued n UPDATEs to compute a number the
        # first one could have written: the increment is bounded-additive and the
        # cap is a min, so n applications of it are the same arithmetic as one
        # application scaled by n. Identical edges, identical weights — but an
        # entity with 100k sightings cost 100k statements, and the pass that was
        # meant to summarise the log was reading and rewriting it instead.
        for (subj, obj), n in near_counts.items():
            if n < min_count:
                continue
            rel_ids.append(self.store.reinforce_relation(
                "relation", subj, self._proximity_verb(obj), object_entity_id=obj,
                supporting_observation_ids=near_evidence[(subj, obj)],
                obs_confidence=0.8, times=n))

        for (subj, lid), n in freq_counts.items():
            if n < min_count:
                continue
            rel_ids.append(self.store.reinforce_relation(
                "relation", subj, "frequents", location_id=lid,
                supporting_observation_ids=freq_evidence[(subj, lid)],
                obs_confidence=0.7, times=n))
        return rel_ids

    # --- orchestration -------------------------------------------------------
    def mine_window(self) -> Optional[float]:
        """The `since` a mining pass reads from, or None for all of history.

        Anchored to the newest observation, not to `now()`. A camera that was
        offline for a fortnight would otherwise come back to a window containing
        nothing, and the pass would quietly un-reinforce every habit it has —
        turning an outage into a claim that the routine stopped.
        """
        days = CONFIG.distill.mine_window_days
        if not days:
            return None
        newest = self.store.newest_observation_at()
        return None if newest is None else newest - days * DAY

    def run(self) -> dict:
        self.store.register_predicate("uses", "person", "object")
        self.store.register_predicate("accompanied_by", "person", "object")
        self.store.register_predicate("frequents", "person", "any")
        self.store.register_predicate("acquired", "object", "any")
        self.store.register_predicate("removed", "object", "any")
        self.normalize()
        self.store.decay_relations()          # age existing edges first (§7)
        changes = self.change_detection()
        events = self.reason_events(changes)
        since = self.mine_window()
        habits = self.mine_habits(since=since)
        relations = self.mine_relations(since=since)
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
