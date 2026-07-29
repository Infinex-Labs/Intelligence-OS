"""Ask — the grounded query surface (FinalPRD §8.2 — M1 step 7).

Not a chatbot. The hard constraint — "the model may not assert anything it cannot
point at" — is enforced by construction: the LLM's ONLY job is to parse the English
question into a structured store query (time window / zone / entity / predicate).
Every fact in the answer (arrivals, departures, durations, states, rule events,
keyframes) is computed deterministically from the observation rows that query
returns. There is no free-text generation, so there is nothing to hallucinate.

The query itself is returned with the answer (the inspectable trace, §8.2).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Optional

from .config import CONFIG
from .store import SCENE_PREFIX, Store

# --- the query form (search plan Phase 4) ------------------------------------
#
# What replaced the five slots, and why the shape is what it is.
#
# The old form had one zone, one person and one substring. Nothing wrong with
# any of them individually; the problem was the *cardinality*. Real questions are
# plural — "the bay or aisle 3", "anyone except the courier", "the last five" —
# and a form that can only say one of a thing cannot say those at all. Worse, it
# failed silently: an unrepresentable question came back as the closest
# representable one, which is a different question with a confident answer.
#
# `intent` is the field that does not filter anything. It says what *kind* of
# answer is wanted, because "how many" and "who" over identical rows are not the
# same reply. Phase 4 routes `count`; `how_often`, `who_with` and `relations`
# reach the aggregates they name in Phase 5. They are in the enum now so that the
# planner's vocabulary does not have to change again when that lands.
INTENTS = ("who", "when", "count", "how_often", "who_with", "timeline",
           "last", "relations")

ENTITY_TYPES = ("person", "object", "scene", "any")

QUERY_TOOL = {
    "name": "graph_query",
    "description": "Translate the user's question into a memory query.",
    "input_schema": {
        "type": "object",
        "properties": {
            "intent": {
                "type": "string", "enum": list(INTENTS),
                "description": "what kind of answer the question wants. 'who' lists "
                "the subjects seen (the default); 'count' asks how many sightings; "
                "'how_often' asks about a recurring pattern; 'who_with' asks who was "
                "present alongside someone; 'relations' asks how two things are "
                "connected; 'last' or 'timeline' ask for recent or ordered sightings",
            },
            "start": {"type": ["number", "null"],
                      "description": "window start, epoch seconds; null = no lower bound"},
            "end": {"type": ["number", "null"],
                    "description": "window end, epoch seconds; null = now"},
            "zones": {
                "type": "array", "items": {"type": "string"},
                "description": "known zone names to search, e.g. ['loading_bay', "
                "'aisle_3']. Empty = anywhere. Use every zone the question could "
                "mean, not just the closest one",
            },
            "cameras": {
                "type": "array", "items": {"type": "string"},
                "description": "known camera names, when the question is about what a "
                "particular camera saw. Empty = every camera",
            },
            "entity_labels": {
                "type": "array", "items": {"type": "string"},
                "description": "known entity labels the question is asking about. "
                "Empty = anyone",
            },
            "exclude_entity_labels": {
                "type": "array", "items": {"type": "string"},
                "description": "known entity labels to leave OUT — 'anyone except the "
                "courier'",
            },
            "entity_type": {
                "type": "string", "enum": list(ENTITY_TYPES),
                "description": "restrict to people, to objects/vehicles, or to facts "
                "recorded against a place ('scene'). 'any' is the default",
            },
            # Phase 2 put a stemming index behind this field, which inverts the
            # old advice: 'smok' used to be the safe way to reach "smoking", and
            # now it is the way to miss it — porter stems 'smoking' to 'smoke'
            # and 'smok' to itself. Whole words, as the user said them.
            "text": {
                "type": ["string", "null"],
                "description": "the words to search for, as whole words (e.g. "
                "'smoking', 'cigarettes', 'gate open'); do not truncate them. null "
                "if the question names no particular thing to look for",
            },
            "exclude_predicates": {
                "type": "array", "items": {"type": "string"},
                "description": "exact predicates to leave out. Rarely needed",
            },
            "min_confidence": {
                "type": ["number", "null"],
                "description": "0-1 floor on detection confidence; use only when the "
                "question asks for certainty ('definitely', 'for sure')",
            },
            "order": {
                "type": "string", "enum": ["asc", "desc"],
                "description": "'asc' reads as a timeline (the default); 'desc' puts "
                "the most recent first, for 'the latest' or 'the last few'",
            },
            "limit": {
                "type": ["integer", "null"],
                "description": "at most this many subjects in the answer, for 'the "
                "last two' or 'the top five'. null = no cap",
            },
        },
        "required": ["intent", "start", "end", "zones", "entity_labels", "text"],
    },
}

PLAN_SYSTEM = (
    "You translate a question about what a camera-memory system saw into a "
    "structured query. You will be given the current time, the known zone names, "
    "the known camera names, and the known entity labels. Resolve relative times "
    "('yesterday', 'around 3', 'last Tuesday 2-4pm') into epoch seconds using the "
    "current time. Map any mentioned place to the known zone names it could mean — "
    "list every one of them, not just the closest. Only use zone names, camera "
    "names and entity labels from the provided lists; a name that is not on a list "
    "does not exist in this memory, so leave the field empty rather than inventing "
    "one. Leave a field out entirely when the question does not constrain it: an "
    "empty list means 'anywhere' / 'anyone', never 'nowhere'. Earlier turns of the "
    "conversation may precede the question — use them to resolve follow-ups ('what "
    "about yesterday?', 'and her?', 'only the smoking ones') into a complete query "
    "on their own. Answer by calling graph_query."
)


def _as_list(value) -> list[str]:
    """A plan field that should be a list of names, however it arrived.

    The planner is an LLM, so a field documented as an array comes back as a bare
    string often enough to be worth handling rather than dropping. Dropping it is
    the bad failure: the filter silently disappears and the answer widens.
    """
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    return [str(v) for v in value if str(v).strip()]


def normalize_plan(query: dict) -> dict:
    """Any accepted plan shape, as the one canonical shape `execute()` reads.

    The planner is a single tool-call boundary, so the pre-Phase-4 five-slot form
    can be up-converted here instead of being a breaking change: `zone` becomes a
    one-element `zones`, `entity_label` a one-element `entity_labels`. Stored
    conversation turns keep their rendered payload, so history replays without
    re-planning — but a client, a script or a saved query may still send the old
    shape, and it costs one function to keep all of them working.

    `predicate_contains` is the one old field kept as itself rather than folded
    into `text`. They are not synonyms: `text` goes to the lexical index alone,
    while `predicate_contains` is also a literal substring test, and that is the
    machine contract 'rule_fired' is matched on. Folding it in would silently
    drop the substring half.
    """
    q = dict(query or {})

    zones = _as_list(q.get("zones"))
    if not zones and q.get("zone"):
        zones = [str(q["zone"])]

    labels = _as_list(q.get("entity_labels"))
    if not labels and q.get("entity_label"):
        labels = [str(q["entity_label"])]

    intent = str(q.get("intent") or "who").strip().lower()
    etype = str(q.get("entity_type") or "any").strip().lower()
    order = str(q.get("order") or "asc").strip().lower()

    limit = q.get("limit")
    try:
        limit = int(limit) if limit is not None else None
    except (TypeError, ValueError):
        limit = None
    if limit is not None and limit < 1:
        limit = None            # "at most zero results" is never a real question

    conf = q.get("min_confidence")
    try:
        conf = float(conf) if conf is not None else None
    except (TypeError, ValueError):
        conf = None

    return {
        # An unknown intent behaves as `who` rather than raising. The planner is
        # a language model: a value outside the enum is a typo, and answering the
        # question plainly beats refusing it over a routing hint.
        "intent": intent if intent in INTENTS else "who",
        "start": q.get("start"),
        "end": q.get("end"),
        "zones": zones,
        "cameras": _as_list(q.get("cameras")),
        "entity_labels": labels,
        "exclude_entity_labels": _as_list(q.get("exclude_entity_labels")),
        "entity_type": etype if etype in ENTITY_TYPES else "any",
        "text": (q.get("text") or "").strip().lower(),
        "predicate_contains": (q.get("predicate_contains") or "").strip().lower(),
        "exclude_predicates": _as_list(q.get("exclude_predicates")),
        "min_confidence": conf,
        "order": order if order in ("asc", "desc") else "asc",
        "limit": limit,
    }


def _history_messages(history) -> list:
    """Prior turns as plain user/assistant messages. This is what makes the chat
    conversational: 'what about yesterday?' or 'and her?' can only be planned if the
    model sees what came before. Client-supplied, so clamp hard."""
    out = []
    for turn in (history or [])[-6:]:            # last 6 turns is plenty of context
        q = str(turn.get("q") or "").strip()[:2000]
        a = str(turn.get("a") or "").strip()[:2000]
        if q:
            out.append({"role": "user", "content": q})
            out.append({"role": "assistant", "content": a or "(no answer)"})
    return out


def plan_query(question: str, store: Store, now: Optional[float] = None,
               history=None) -> dict:
    """One LLM call: English -> structured query, in the context of the conversation
    so far. Raises RuntimeError with a plain message when no key is configured (the
    CLI/API surface it verbatim)."""
    if not CONFIG.vlm.enabled:
        raise RuntimeError(
            "Ask needs a model to understand the question. Set ANTHROPIC_API_KEY. "
            "(The raw memory is still queryable via /api/observations and the graph UI.)")
    import anthropic  # deferred, optional dep
    now = now or time.time()
    zones = [r["name"] for r in store.locations()]
    labels = [e["label"] for e in store.list_entities() if e["label"]]
    client = anthropic.Anthropic()
    cams = _known_cameras(store)
    msg = client.messages.create(
        model=CONFIG.vlm.model,
        max_tokens=300,
        system=PLAN_SYSTEM,
        tools=[QUERY_TOOL],
        tool_choice={"type": "tool", "name": "graph_query"},
        messages=[*_history_messages(history), {"role": "user", "content": (
            f"Current time: {time.strftime('%A %Y-%m-%d %H:%M:%S', time.localtime(now))} "
            f"(epoch {now:.0f})\nKnown zones: {zones}\nKnown cameras: {cams}\n"
            f"Known entity labels: {labels}\n\n"
            f"Question: {question}")}],
    )
    for block in msg.content:
        if getattr(block, "type", None) == "tool_use":
            return block.input
    raise RuntimeError("Could not parse the question into a query.")


def _known_cameras(store: Store) -> list[str]:
    """Camera names the planner is allowed to use.

    The memory's cameras (those with a zone drawn on them) unioned with the
    configured ones, because the two disagree in both directions and each gap
    is a real question the planner could otherwise not ask: a camera added this
    morning has no zone yet, and a camera removed from the config still owns
    every frame it ever recorded.
    """
    names = set(store.cameras())
    try:
        from .config import resolve_cameras
        names.update(c["name"] for c in resolve_cameras() if c.get("name"))
    except Exception:
        pass      # a malformed camera config must not take the whole query down
    return sorted(names)


MAX_PUSHED_IDS = 900        # SQLite's older bound-variable cap, less headroom


def _label_for(eid: str, row, loc_names: dict) -> Optional[str]:
    """The name a subject is answered under. Not a column — derived.

    An entity with no label is shown by its type and a short id; a fact nobody
    owns — an open gate, a spill — is recorded against the place
    (`store.scene_subject`) and shown as the zone's name. `None` means the
    subject cannot be named, and a subject that cannot be named cannot be an
    answer.

    There is one definition here because there are two callers: the loop below,
    and the id set handed to SQL when the question names someone. If those two
    ever disagreed, a question would filter on one meaning of "label" and
    display another.
    """
    if row is not None:
        return row["label"] or f"{row['type'].capitalize()} {eid[-6:]}"
    if eid.startswith(SCENE_PREFIX):
        place = eid[len(SCENE_PREFIX):]
        return loc_names.get(place, place)
    return None


class _SubjectFilter:
    """Which subjects may appear in the answer (search plan Phase 4).

    One rule with two faces. `accepts()` is the definition, and has to be: it
    runs on the *display* label and type, which are derived rather than stored,
    so nothing in SQL can be the authority on them. `ids` and `exclude_ids` are
    that same rule handed to SQLite, so rows that could never survive
    `accepts()` are never read off disk.

    The asymmetry is deliberate and load-bearing: SQL may only ever remove rows
    `accepts()` would have removed anyway. That is what makes the fallback safe
    — when a set is too big to bind, the pushdown is simply dropped, and the
    answer is identical, just slower.

    Resolution walks the entity table once. It is bounded by how many things
    have ever been seen rather than by how often they were seen, which is the
    distinction that makes this affordable and made the per-row version not.
    """

    def __init__(self, store: Store, plan: dict, loc_names: dict):
        self.labels = [s.strip().lower() for s in plan["entity_labels"] if s.strip()]
        self.excludes = [s.strip().lower()
                         for s in plan["exclude_entity_labels"] if s.strip()]
        self.etype = plan["entity_type"]
        self.ids: Optional[list[str]] = None
        self.exclude_ids: Optional[list[str]] = None
        # A positive constraint names the subjects that may answer; an exclusion
        # alone names only the ones that may not. Resolving the positive set for
        # an exclusion-only question would mean listing every subject that
        # exists in order to leave one out, which is the expensive way round.
        if self.labels or self.etype != "any":
            ids = [eid for eid, label, etype in self._subjects(store, loc_names)
                   if self.accepts(label, etype)]
            self.ids = ids if len(ids) <= MAX_PUSHED_IDS else None
        elif self.excludes:
            bad = [eid for eid, label, etype in self._subjects(store, loc_names)
                   if not self.accepts(label, etype)]
            self.exclude_ids = bad if len(bad) <= MAX_PUSHED_IDS else None

    def accepts(self, label: str, etype: str) -> bool:
        """Whether a subject, as it will be shown, answers this question."""
        low = label.lower()
        if self.etype != "any" and etype != self.etype:
            return False
        if self.labels and not any(q in low for q in self.labels):
            return False
        return not any(q in low for q in self.excludes)

    def _subjects(self, store: Store, loc_names: dict):
        """Every nameable subject, as (id, display label, type).

        Each half is skipped when the type filter has already ruled it out —
        `scene:` subjects come from a DISTINCT over the observations, so not
        asking for them when they cannot match is worth the two lines.
        """
        if self.etype != "scene":
            for e in store.list_entities(active_only=False):
                label = _label_for(e["entity_id"], e, loc_names)
                if label is not None:
                    yield e["entity_id"], label, e["type"]
        if self.etype in ("any", "scene"):
            for sid in store.scene_subjects():
                label = _label_for(sid, None, loc_names)
                if label is not None:
                    yield sid, label, "scene"


# How far back a limited question walks before giving up on the shortcut below.
# Quadrupling from here: "the last few" is normally answered by the first page,
# and the fallback is the aggregate that would have run anyway.
LIMIT_PROBE_START = 256
LIMIT_PROBE_MAX = 65536


def _limited_subjects(store: Store, plan: dict, filters: dict,
                      subjects: "_SubjectFilter", loc_names: dict) -> Optional[list[str]]:
    """The `limit` subjects at the asked-for end of the window, as ids.

    Reading the rows in the requested order and taking the first `limit`
    DISTINCT subjects gives exactly the right ones, and this is why: a subject's
    `last_seen` is where it first appears reading backwards, and its
    `first_seen` is where it first appears reading forwards. So the *shortlist*
    can be decided by a bounded walk even though each subject's aggregate spans
    all of its rows — and the aggregate is then computed for those subjects
    alone rather than for the whole memory.

    Without this, "the last two sightings" is the most expensive question in the
    system: no filter to push down, so every row ever recorded gets aggregated
    in order to throw all but two subjects away.

    Returns None when the walk cannot prove it found them, and the caller
    aggregates everything — the slow path, but the same answer.
    """
    want = plan["limit"]
    probe = LIMIT_PROBE_START
    while True:
        rows = store.observations(limit=probe, **filters)
        # dict.fromkeys keeps first-appearance order, which is the whole trick.
        candidates = list(dict.fromkeys(o["subject_entity_id"] for o in rows))
        ents = store.get_entities(candidates)
        keep: list[str] = []
        for eid in candidates:
            row = ents.get(eid)
            label = _label_for(eid, row, loc_names)
            if label is None:
                continue
            if subjects.accepts(label, row["type"] if row is not None else "scene"):
                keep.append(eid)
                if len(keep) == want:
                    return keep
        if len(rows) < probe:
            return keep          # the walk reached the end; this is all there is
        if probe >= LIMIT_PROBE_MAX:
            return None          # too far to be a shortcut — aggregate the lot
        probe *= 4


def _zone_ids(names, loc_names: dict) -> Optional[list[str]]:
    """The zones named, as location ids. `None` when the question named no place.

    A name matching no zone contributes no id, so a list where *nothing* matches
    comes back empty — a filter no row satisfies. That empty case is the point.
    The single-zone field it replaces turned an unrecognised place into no filter
    at all, so "who was in the car park", asked of a memory with no such zone,
    answered with everyone everywhere and looked authoritative doing it. A place
    that does not exist has no sightings in it.
    """
    if not names:
        return None
    wanted = {n.strip().lower() for n in names if n.strip()}
    return [lid for lid, name in loc_names.items() if (name or "").lower() in wanted]


def execute(store: Store, query: dict) -> dict:
    """Deterministic aggregation over observation rows. Every claim in the output
    is a row (or an aggregate of rows) the caller can inspect."""
    plan = normalize_plan(query)
    start, end = plan["start"], plan["end"]
    pred_q, text_q = plan["predicate_contains"], plan["text"]

    loc_names = {r["location_id"]: r["name"] for r in store.locations()}
    loc_ids = _zone_ids(plan["zones"], loc_names)
    cam_ids = plan["cameras"] or None
    subjects = _SubjectFilter(store, plan, loc_names)

    # Phase 2: the words go to the lexical index, which stems and ranks. The two
    # word fields differ in how strict they are, and deliberately so:
    #   `text`               — the search surface. Match or you are not a result.
    #   `predicate_contains` — the older field, still the machine contract that
    #                          'rule_fired' is matched on. It keeps its substring
    #                          behaviour AND gains the index, as a union: a phase
    #                          that widens recall must never take a hit away.
    # Both go through the same MATCH so ranking is comparable across them.
    #
    # Phase 4 hands the shortlist the same hard filters the row query gets. It is
    # capped, so a ranked list drawn from the whole memory and then narrowed to
    # one zone is shorter than one drawn from that zone — the narrow question
    # would have paid for the filter in recall.
    scores: dict[str, float] = {}
    if text_q or pred_q:
        for r in store.search_text(text_q or pred_q, since=start, until=end,
                                   location_ids=loc_ids, camera_ids=cam_ids,
                                   entity_ids=subjects.ids):
            scores[r["observation_id"]] = r["score"]

    # Phase 3: the window, the zone and the words are all decided by SQLite now.
    # This used to read the whole table and drop rows in a Python loop, which
    # meant a question about one hour of one camera paid for every hour of every
    # camera ever recorded. The union that Phase 2 expressed as two conditions in
    # the loop is the same union, handed to `match_any` — the ranked ids OR the
    # substring, and never wider than the hard filters above.
    filters = dict(since=start, until=end, location_ids=loc_ids,
                   camera_ids=cam_ids, entity_ids=subjects.ids,
                   exclude_entity_ids=subjects.exclude_ids,
                   exclude_predicates=plan["exclude_predicates"],
                   min_confidence=plan["min_confidence"],
                   # `order` reaches the rows, not just the answer: an entity
                   # keeps the first states it is seen in, capped, so reading the
                   # window backwards is what makes "the last few" describe the
                   # latest states rather than the earliest.
                   order=plan["order"])
    if text_q or pred_q:
        # No shortlist walk here: a word question is ordered by match score, not
        # by time, so the first rows read are not the first results.
        rows = store.observations(
            observation_ids=list(scores),
            predicate_contains=pred_q if pred_q and not text_q else None,
            match_any=True, **filters)
    else:
        if plan["limit"] is not None:
            short = _limited_subjects(store, plan, filters, subjects, loc_names)
            if short is not None:
                # Already screened by `accepts`, so this only ever narrows.
                filters = {**filters, "entity_ids": short}
        rows = store.observations(**filters)

    # One statement for every entity these rows are about, instead of one per
    # row. A busy hour is thousands of rows about a dozen people.
    ents = store.get_entities({o["subject_entity_id"] for o in rows})

    per_entity: dict[str, dict] = {}
    rejected: set[str] = set()      # ids already judged: unresolvable, or wrong label
    for o in rows:
        hit = scores.get(o["observation_id"])
        eid = o["subject_entity_id"]
        ent = per_entity.get(eid)
        if ent is None:
            if eid in rejected:
                continue
            row = ents.get(eid)
            label = _label_for(eid, row, loc_names)
            if label is None:
                rejected.add(eid)
                continue
            etype = row["type"] if row is not None else "scene"
            # Still checked here even when SQL already narrowed by it: the
            # subject filter is defined on the displayed name and type, and this
            # loop is where both are decided.
            if not subjects.accepts(label, etype):
                rejected.add(eid)
                continue
            ent = per_entity[eid] = {
                "entity_id": eid, "label": label, "type": etype,
                "first_seen": o["timestamp"], "last_seen": o["timestamp"],
                "n_observations": 0, "states": [], "rule_events": [], "keyframes": [],
                # None, not 0.0: a row found by substring was never scored, and
                # calling that a zero would rank it as the worst match instead of
                # the unranked one it is.
                "match_score": None,
            }
        ent["n_observations"] += 1
        if hit is not None:
            # An entity is as relevant as its best-matching row. Summing would
            # rank whoever was on camera longest, which is presence, not answer.
            prev = ent["match_score"]
            ent["match_score"] = hit if prev is None else max(prev, hit)
        ent["first_seen"] = min(ent["first_seen"], o["timestamp"])
        ent["last_seen"] = max(ent["last_seen"], o["timestamp"])
        pred = o["predicate"]
        if pred.startswith("rule_fired:"):
            ent["rule_events"].append({
                "rule": pred[len("rule_fired:"):], "timestamp": o["timestamp"],
                "keyframe": _kf(o["source_ref"]), "observation_id": o["observation_id"],
            })
        elif o["origin"] == "vlm" and pred not in ent["states"]:
            ent["states"].append(pred)
        if o["source_ref"]:
            url = _kf(o["source_ref"])
            if url not in ent["keyframes"]:
                ent["keyframes"].append(url)

    # A word question asks "who best fits these words", so it comes back ranked;
    # anything else is a window over a period, and reads as a timeline.
    if text_q or pred_q:
        # Scored first, then the unscored substring hits, then by time. bm25 is
        # relative, not absolute — its magnitude says nothing on its own, which
        # is why it orders results and never gates them.
        entities = sorted(per_entity.values(),
                          key=lambda e: (e["match_score"] is None,
                                         -(e["match_score"] or 0.0),
                                         e["first_seen"]))
    elif plan["order"] == "desc":
        # Most recently seen first. `last_seen`, not `first_seen`: "the latest"
        # means whoever was on camera most recently, not whoever turned up
        # latest — for someone who arrived early and stayed, those differ.
        entities = sorted(per_entity.values(), key=lambda e: e["last_seen"],
                          reverse=True)
    else:
        entities = sorted(per_entity.values(), key=lambda e: e["first_seen"])

    # `limit` caps subjects, not rows, and is applied here rather than as SQL's
    # LIMIT — which would be faster and wrong. An entity's first_seen, duration
    # and states are aggregates over all of its rows, so truncating rows would
    # not return the last two subjects, it would return two subjects with a
    # truncated history and no sign that anything was cut.
    if plan["limit"] is not None:
        entities = entities[:plan["limit"]]

    for e in entities:
        e["duration_s"] = round(e["last_seen"] - e["first_seen"], 1)
        e["keyframes"] = e["keyframes"][:4]
        e["states"] = e["states"][:8]
        if e["match_score"] is not None:
            e["match_score"] = float(f"{e['match_score']:.6g}")

    # Summed over what is actually being returned, so the count can never
    # describe rows the answer does not show — which is what it would do if the
    # loop's running total survived a `limit`.
    total = sum(e["n_observations"] for e in entities)
    result = {"query": query, "plan": plan, "total_observations": total,
              "entities": entities, "ranked": bool(text_q or pred_q)}
    if plan["intent"] == "count":
        # The same number the evidence already carried, promoted to the answer.
        # "How many" and "who" run the identical query — what differs is which
        # part is the reply, and a caller cannot infer that from the rows.
        result["count"] = total
    return result


def _kf(source_ref: Optional[str]) -> Optional[str]:
    """Keyframe URL for a source_ref. A pure transform on purpose: whether the
    file is still on disk is a serving concern (web.py drops pruned ones), and
    this runs once per observation row inside the query hot path."""
    return f"/keyframe/{os.path.basename(source_ref)}" if source_ref else None


NARRATE_SYSTEM = (
    "You are the voice of a camera-memory system answering an investigator. You are "
    "given their question and a JSON summary of EXACTLY what the memory returned for "
    "THIS question — who was seen, when, for how long, their states, and any rule "
    "events. Times are already formatted for you. Reply in 2-4 natural sentences, the "
    "way you'd tell a colleague what the footage shows, and pick up naturally from the "
    "earlier conversation (resolve 'them', 'that', 'the same person'). But every fact "
    "you state — every name, time, count, place, event — must come from THIS question's "
    "JSON, never from memory of earlier turns and never invented; prior turns are for "
    "tone and pronouns only. Never soften or inflate the numbers. If the summary is "
    "empty, say plainly that nothing matching was found. No preamble, no bullet points, "
    "no markdown — just the answer."
)


def _facts(result: dict) -> dict:
    """The sayable projection of the evidence: pre-formatted times, no ids or urls.
    Feeding the model formatted facts is what keeps the prose grounded — it phrases
    these, it cannot recompute them wrong."""
    t = lambda ts: time.strftime("%a %d %b %H:%M", time.localtime(ts))
    facts = {
        "total_observations": result["total_observations"],
        "entities": [{
            "who": e["label"],
            "first_seen": t(e["first_seen"]),
            "last_seen": t(e["last_seen"]),
            "duration_seconds": e["duration_s"],
            "times_observed": e["n_observations"],
            "states": [s.replace("state:", "").replace("_", " ") for s in e["states"]],
            "rule_events": [{"rule": ev["rule"].replace("_", " "), "at": t(ev["timestamp"])}
                            for ev in e["rule_events"]],
        } for e in result["entities"]],
    }
    if result.get("count") is not None:
        # Named separately from `total_observations` because the question asked
        # for it. Same number, but the model is being told which one is the reply.
        facts["count"] = result["count"]
    return facts


def narrate(question: str, result: dict, history=None) -> Optional[str]:
    """One LLM call: grounded evidence -> natural-language answer, continuing the
    conversation. Returns None when no model is configured (the UI falls back to the
    count summary). A narration failure is never fatal — the facts already rendered
    without it."""
    if not CONFIG.vlm.enabled:
        return None
    try:
        import anthropic
        msg = anthropic.Anthropic().messages.create(
            model=CONFIG.vlm.model, max_tokens=400, system=NARRATE_SYSTEM,
            messages=[*_history_messages(history), {"role": "user", "content":
                       f"Question: {question}\n\nWhat memory returned:\n"
                       f"{json.dumps(_facts(result))}"}],
        )
        text = "".join(b.text for b in msg.content
                       if getattr(b, "type", None) == "text").strip()
        return text or None
    except Exception:
        return None      # prose is a nicety; the grounded evidence stands alone


def ask(question: str, store: Optional[Store] = None, history=None) -> dict:
    """plan -> execute -> narrate, in the context of `history` (prior [{q,a},...]
    turns). The returned dict carries the query (the trace) and the aggregated
    evidence; `answer` is a grounded natural-language reply over that same evidence
    (None if no model)."""
    own = store is None
    store = store or Store()
    try:
        result = execute(store, plan_query(question, store, history=history))
        result["question"] = question
        result["answer"] = narrate(question, result, history=history)
        return result
    finally:
        if own:
            store.close()


def render_text(result: dict) -> str:
    """Plain-text rendering for the CLI. Same facts, no additions."""
    t = lambda ts: time.strftime("%a %H:%M:%S", time.localtime(ts))
    lines = []
    if result.get("answer"):
        lines += [result["answer"], ""]
    if result.get("count") is not None:
        lines.append(f"count: {result['count']} sighting(s).")
    lines.append(f"{len(result['entities'])} entit(y/ies), "
                 f"{result['total_observations']} observation(s) in window.")
    for e in result["entities"]:
        lines.append(f"\n{e['label']}  ({e['entity_id']})")
        lines.append(f"  seen {t(e['first_seen'])} -> {t(e['last_seen'])}"
                     f"  ({e['duration_s']}s, {e['n_observations']} obs)")
        for ev in e["rule_events"]:
            lines.append(f"  RULE FIRED {ev['rule']} @ {t(ev['timestamp'])}  {ev['keyframe']}")
        for s in e["states"]:
            lines.append(f"  state: {s}")
        if e["keyframes"]:
            lines.append(f"  evidence: {', '.join(k for k in e['keyframes'] if k)}")
    lines.append(f"\nquery trace: {json.dumps(result['query'])}")
    return "\n".join(lines)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="intelligence_os.ask")
    p.add_argument("question")
    args = p.parse_args(argv)
    try:
        print(render_text(ask(args.question)))
        return 0
    except RuntimeError as e:
        print(str(e), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
