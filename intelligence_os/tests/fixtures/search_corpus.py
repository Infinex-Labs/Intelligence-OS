"""A deterministic memory to measure search against (search plan Phase 0).

Two things live here, and keeping them apart is the whole point:

  PERCEIVED    what the vision model reported — the ground truth of what was
               *seen*. Written as VLM report dicts, exactly the shape
               `vlm.DESCRIBE_SCHEMA` produces.

  the database what actually survived the write path into SQLite.

`build()` writes the corpus by calling the REAL `SceneDescription.states()` and
the real `run.py` subject-resolution rule, not a copy of them. So the gap between
those two columns is measured, not asserted — and when Phase 1 stops discarding
objects and areas, `retention()` moves on its own without this file changing.

Timestamps are anchored to a fixed UTC Tuesday so weekday and hour buckets are
identical on every machine. Note the open question in the plan (§8.2): habits are
mined in UTC (`distill.py`) but rendered in local time (`ask.py`), so nothing here
asserts on a rendered local-time string.
"""
from __future__ import annotations

import calendar

from intelligence_os.store import Store
from intelligence_os.vlm import SceneDescription

# 2026-06-02 14:00:00 UTC — a Tuesday. Verified, not assumed.
ANCHOR = calendar.timegm((2026, 6, 2, 14, 0, 0, 0, 0, 0))
assert ANCHOR == 1780408800, "anchor drifted; the corpus is no longer deterministic"

DAY = 86400.0
WEEK = 7 * DAY

MODEL = "fixture-vlm"


# --- what the vision model saw ----------------------------------------------
# Each entry is (offset_from_anchor, camera, zone_key, keyframe, report).
# `ref` fields use corpus entity KEYS; build() swaps them for real ids.
#
# The reports deliberately carry all four sections the real schema allows —
# people, notable, objects, locations — because two of those four are dropped by
# today's write path. That loss is what `retention()` puts a number on.
PERCEIVED: list[tuple[float, str, str, str, dict]] = [
    # Dave smokes at the bay. Stored as "having a cigarette" — nobody searching
    # would guess that wording. This is gap G1's exhibit.
    (-6 * DAY + 1800, "cam_dock", "loading_bay", "kf_dave_smoke.jpg", {
        "people": [{"ref": "dave", "state": "having a cigarette"}],
        "notable": ["cigarette butts on the ground"],
        "objects": [{"label": "gate", "description": "open, unlatched"},
                    {"label": "pallet stack", "description": "three high, leaning"}],
        "locations": [{"location": "loading_bay",
                       "contents": ["open gate", "pallet stack", "puddle"]}],
    }),
    # Nadia loiters. Stored as "standing around, waiting" — no lexical overlap
    # with "loitering" at all, so only Phase 6 can reach it.
    (-5 * DAY + 64800, "cam_yard", "side_gate", "kf_nadia_wait.jpg", {
        "people": [{"ref": "nadia", "state": "standing around, waiting",
                    "context_analysis": "lingering longer than her usual pass-through"}],
        "notable": [],
        "objects": [{"label": "bicycle", "description": "unlocked, against the fence"}],
        "locations": [{"location": "side_gate", "contents": ["bicycle", "wheelie bin"]}],
    }),
    # Priya on a phone call. Stored "on phone"; asking "on the phone" fails today
    # purely on the word "the" — Phase 2 (stemming + phrase handling) fixes it.
    (-4 * DAY + 39600, "cam_dock", "aisle_3", "kf_priya_phone.jpg", {
        "people": [{"ref": "priya", "state": "on phone"}],
        "notable": ["aisle light flickering"],
        "objects": [{"label": "forklift", "description": "parked, forks raised"}],
        "locations": [{"location": "aisle_3", "contents": ["forklift", "spill"]}],
    }),
    # The courier. Plain wording; this one already works and must keep working.
    (-3 * DAY + 32400, "cam_yard", "side_gate", "kf_courier.jpg", {
        "people": [{"ref": "courier", "state": "carrying a parcel"}],
        "notable": [],
        "objects": [{"label": "gate", "description": "closed"}],
        "locations": [{"location": "side_gate", "contents": ["closed gate"]}],
    }),
    # Priya and a visitor together — the co-presence exhibit (G3).
    (-2 * DAY + 36000, "cam_yard", "side_gate", "kf_priya_visitor.jpg", {
        "people": [{"ref": "priya", "state": "holding the gate open"},
                   {"ref": "visitor", "state": "waiting to be let in"}],
        "notable": ["two people at the gate together"],
        "objects": [{"label": "gate", "description": "held open by hand"}],
        "locations": [{"location": "side_gate", "contents": ["open gate", "two people"]}],
    }),
]

# The van's recurring visits: six Tuesdays at 14:00 UTC, plus two Thursdays.
# `mine_habits` needs predicate 'present' + a location across >= 2 distinct days,
# so this is what makes "how often does the van come by?" answerable at Phase 5 —
# and "most Tuesdays" answerable only once habits carry a weekday bucket (G8).
VAN_TUESDAYS = [-i * WEEK for i in range(6)]
VAN_THURSDAYS = [-1 * WEEK + 2 * DAY, -3 * WEEK + 2 * DAY]


def _zone_of(key: str) -> str:
    return key


def build(store: Store, *, scale: int = 0) -> dict:
    """Write the corpus. Returns the key -> id maps the cases are written against.

    `scale` pads the table with filler observations for the latency measurement,
    so p95 is read off a realistically sized memory rather than a toy one.
    """
    zones = {
        "loading_bay": store.upsert_location(
            "loading_bay", {"polygon": [[0, 0], [1, 0], [1, 1]]}, camera_id="cam_dock"),
        "side_gate": store.upsert_location(
            "side_gate", {"polygon": [[2, 0], [3, 0], [3, 1]]}, camera_id="cam_yard"),
        "aisle_3": store.upsert_location(
            "aisle_3", {"polygon": [[4, 0], [5, 0], [5, 1]]}, camera_id="cam_dock"),
    }

    # Labels are explicit so the label filter is testable. `nadia` is left
    # unlabelled on purpose: ask.py falls back to "Person <hex>", and a search
    # must still find her without a name to match on.
    people = {
        "dave": store.create_entity("person", label="Dave"),
        "nadia": store.create_entity("person"),
        "priya": store.create_entity("person", label="Priya"),
        "courier": store.create_entity("person", label="Delivery Courier"),
        "visitor": store.create_entity("person", label="Visitor"),
    }
    objects = {
        "van": store.create_entity("object", label="White Van"),
        "forklift": store.create_entity("object", label="Forklift"),
    }
    ids = {**people, **objects}

    # --- detector-origin presence ------------------------------------------
    # Plain sightings, the rows that already work today.
    presence = [
        ("dave", "loading_bay", "cam_dock", -6 * DAY + 1500, "kf_dave_arrive.jpg"),
        ("dave", "loading_bay", "cam_dock", -6 * DAY + 2400, None),
        ("nadia", "side_gate", "cam_yard", -5 * DAY + 64500, "kf_nadia_arrive.jpg"),
        ("nadia", "side_gate", "cam_yard", -5 * DAY + 65400, None),
        ("priya", "aisle_3", "cam_dock", -4 * DAY + 39300, "kf_priya_arrive.jpg"),
        ("courier", "side_gate", "cam_yard", -3 * DAY + 32100, "kf_courier_arrive.jpg"),
        ("priya", "side_gate", "cam_yard", -2 * DAY + 35700, "kf_priya_gate.jpg"),
        ("visitor", "side_gate", "cam_yard", -2 * DAY + 35800, "kf_visitor.jpg"),
    ]
    for key, zone, cam, off, kf in presence:
        store.add_observation(ids[key], "present", location_id=zones[zone],
                              timestamp=ANCHOR + off, confidence=0.9,
                              source_ref=f"/kf/{kf}" if kf else None,
                              origin="detector", camera_id=cam)

    # The van's repeat visits, all at the same hour bucket.
    for off in VAN_TUESDAYS + VAN_THURSDAYS:
        store.add_observation(objects["van"], "present", location_id=zones["loading_bay"],
                              timestamp=ANCHOR + off, confidence=0.85,
                              source_ref="/kf/kf_van.jpg", origin="detector",
                              camera_id="cam_dock")

    # 'near' repeated -> distill mines a 'uses' relation (Priya + forklift).
    for i in range(4):
        store.add_observation(people["priya"], "near", object_entity_id=objects["forklift"],
                              location_id=zones["aisle_3"],
                              timestamp=ANCHOR - 4 * DAY + 39300 + i * 60,
                              confidence=0.7, origin="detector", camera_id="cam_dock")

    # --- a fired rule, so alert-shaped rows are in scope --------------------
    store.add_observation(ids["dave"], "rule_fired:loiter_at_bay",
                          location_id=zones["loading_bay"],
                          timestamp=ANCHOR - 6 * DAY + 2100, confidence=0.8,
                          source_ref="/kf/kf_rule_loiter.jpg", origin="rule",
                          camera_id="cam_dock")

    # --- VLM output, written the way run.py writes it ----------------------
    # This is the load-bearing part of the fixture: it goes through the real
    # SceneDescription, so whatever the production write path drops, this drops.
    for off, cam, zone_key, kf, report in PERCEIVED:
        resolved = _resolve_refs(report, ids)
        desc = SceneDescription(resolved, MODEL)
        for subj_ref, predicate in desc.states():
            # run.py's rule: bare 'person'/'scene' refs attach to the first
            # resolved entity; anything that isn't an entity id is dropped.
            subj = subj_ref
            if subj_ref in ("person", "scene"):
                subj = _first_entity(resolved, ids)
            if not subj or not subj.startswith("ent_"):
                continue
            store.add_observation(subj, predicate, location_id=zones[zone_key],
                                  timestamp=ANCHOR + off, confidence=0.6,
                                  source_ref=f"/kf/{kf}", origin="vlm",
                                  camera_id=cam)

    # --- co-presence snapshots (unreachable from ask today, G3) -------------
    store.add_snapshot(zones["side_gate"], [people["priya"], people["visitor"]],
                       timestamp=ANCHOR - 2 * DAY + 35900)
    store.add_snapshot(zones["loading_bay"], [ids["dave"]],
                       timestamp=ANCHOR - 6 * DAY + 1800)

    if scale:
        # A dedicated filler entity in its own zone: padding must never change
        # what a correctness case sees, only how much the engine has to sift.
        filler = store.create_entity("object", label="Filler Crate")
        filler_zone = store.upsert_location(
            "filler_store", {"polygon": [[9, 9], [10, 9], [10, 10]]}, camera_id="cam_dock")
        _pad(store, filler, filler_zone, scale)

    return {"zones": zones, "entities": ids, "people": people, "objects": objects,
            "anchor": ANCHOR}


def _resolve_refs(report: dict, ids: dict) -> dict:
    """Swap corpus keys for real entity ids, leaving the report shape untouched."""
    out = {k: v for k, v in report.items()}
    out["people"] = [{**p, "ref": ids.get(p.get("ref"), p.get("ref", "person"))}
                     for p in report.get("people", [])]
    return out


def _first_entity(report: dict, ids: dict) -> str | None:
    for p in report.get("people", []):
        ref = p.get("ref")
        if isinstance(ref, str) and ref.startswith("ent_"):
            return ref
    return None


def _pad(store: Store, entity_id: str, location_id: str, n: int) -> None:
    """Filler presence rows spread over 90 days, so latency is measured against a
    memory the size of a real deployment's rather than a fixture's."""
    span = 90 * DAY
    with store.tx() as c:
        for i in range(n):
            c.execute(
                "INSERT INTO observations(observation_id,subject_entity_id,predicate,"
                "object_entity_id,location_id,timestamp,confidence,source_ref,origin,"
                "camera_id,user_id,status) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (f"obs_pad{i:08d}", entity_id, "present", None, location_id,
                 ANCHOR - span + (i * span / max(n, 1)), 0.5, None, "detector",
                 "cam_dock", None, None))


# --- the perception-retention measurement -----------------------------------
# Words carrying no content, so a fact is judged on what it says rather than how
# it is punctuated. Kept deliberately short: over-trimming would make retention
# look better than it is.
_STOPWORDS = frozenset(
    "a an the of to is are and by on in at with for its it as from".split())


def _tokens(text: str) -> frozenset[str]:
    cleaned = "".join(ch if ch.isalnum() or ch.isspace() else " " for ch in text.lower())
    return frozenset(w for w in cleaned.split() if w and w not in _STOPWORDS)


def perceived_facts() -> list[tuple[float, str, str]]:
    """Every distinct thing the vision model reported, as (timestamp, section, text).

    The denominator for retention: if it is in here and not in the database, the
    system saw it, described it, and then threw the sentence away before the
    insert.
    """
    facts: list[tuple[float, str, str]] = []
    for off, _cam, _zone, _kf, report in PERCEIVED:
        ts = ANCHOR + off
        for p in report.get("people", []):
            if p.get("state"):
                facts.append((ts, "people.state", p["state"]))
            if p.get("context_analysis"):
                facts.append((ts, "people.context_analysis", p["context_analysis"]))
        for n in report.get("notable", []):
            facts.append((ts, "notable", n))
        for o in report.get("objects", []):
            text = o["label"] + (f" {o['description']}" if o.get("description") else "")
            facts.append((ts, "objects", text))
        for loc in report.get("locations", []):
            for c in loc.get("contents", []):
                facts.append((ts, "locations.contents", c))
    return facts


def retention(store: Store) -> dict:
    """How much of what was perceived is actually searchable.

    Scored **per report**, not against the whole database: a fact counts as
    retained only if a row written from that same keyframe carries every content
    word of it. Matching globally would score an object fact as retained because
    some unrelated frame's state happened to mention the same noun — which
    flatters the number and hides exactly the loss this is here to measure.

    Token-superset rather than substring, so it survives Phase 1 choosing a
    different separator when it renders these into `observations.text`.
    """
    by_ts: dict[float, list[frozenset[str]]] = {}
    for o in store.observations():
        by_ts.setdefault(o["timestamp"], []).append(_tokens(o["predicate"]))

    per_section: dict[str, dict] = {}
    for ts, section, text in perceived_facts():
        bucket = per_section.setdefault(section, {"perceived": 0, "retained": 0})
        bucket["perceived"] += 1
        want = _tokens(text)
        if want and any(want <= got for got in by_ts.get(ts, [])):
            bucket["retained"] += 1
    total_p = sum(b["perceived"] for b in per_section.values())
    total_r = sum(b["retained"] for b in per_section.values())
    return {
        "sections": per_section,
        "perceived": total_p,
        "retained": total_r,
        "rate": (total_r / total_p) if total_p else 1.0,
    }
