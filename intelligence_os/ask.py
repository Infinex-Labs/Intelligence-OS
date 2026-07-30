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
from collections import Counter, defaultdict
from typing import Optional

from . import semantic
from .config import CONFIG
from .distill import WEEKDAY_ABBR, bucket_day, bucket_hour, bucket_weekday
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
# same reply. Phase 4 routed `count`; Phase 5 routes the three that reach a
# different table entirely — `how_often` to the mined habits, `who_with` to the
# scene snapshots, `relations` to the distilled edges. Those three also change
# *who the answer is about*: "who was with Priya" is a question whose subject is
# Priya and whose answer is somebody else, which is why they cannot be a filter.
INTENTS = ("who", "when", "count", "how_often", "who_with", "timeline",
           "last", "relations")

ENTITY_TYPES = ("person", "object", "scene", "any")

# How a recurrence answer is cut up. Only `how_often` reads it.
GROUP_BYS = ("weekday", "hour", "day")

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
            "group_by": {
                "type": ["string", "null"], "enum": [*GROUP_BYS, None],
                "description": "for 'how_often' only: which buckets the pattern is "
                "asked about. 'weekday' for 'mostly on Tuesdays?', 'hour' for "
                "'always in the afternoon?', 'day' for a per-date breakdown",
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

    group_by = str(q.get("group_by") or "").strip().lower()

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
        # Defaulted here rather than at the use site, so the trace shows which
        # buckets the answer was actually cut into. "Mostly Tuesdays?" and "how
        # often?" produce the same rows and different answers, and a reader of
        # the trace should not have to know which default applied.
        "group_by": group_by if group_by in GROUP_BYS else "weekday",
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

    def __init__(self, store: Store, plan: dict, loc_names: dict, *,
                 only: Optional[set] = None):
        # `only` is Phase 5's: `who_with` and `relations` ask about one subject
        # and answer with a different one, so once the companions or the related
        # things are known, they ARE the permitted set and the plan's labels have
        # already done their job as the anchor. Passing them back through the
        # label filter would ask "which of the people with Priya are called
        # Priya", which is nobody.
        self.only = None if only is None else set(only)
        named = self.only is None
        self.labels = ([s.strip().lower() for s in plan["entity_labels"] if s.strip()]
                       if named else [])
        # Exclusions survive, because "who was with her, apart from the courier"
        # is a constraint on the answer rather than on the anchor.
        self.excludes = [s.strip().lower()
                         for s in plan["exclude_entity_labels"] if s.strip()]
        self.etype = plan["entity_type"] if named else "any"
        self.ids: Optional[list[str]] = None
        self.exclude_ids: Optional[list[str]] = None
        # A positive constraint names the subjects that may answer; an exclusion
        # alone names only the ones that may not. Resolving the positive set for
        # an exclusion-only question would mean listing every subject that
        # exists in order to leave one out, which is the expensive way round.
        if self.only is not None:
            ids = sorted(self.only)
            self.ids = ids if len(ids) <= MAX_PUSHED_IDS else None
        elif self.labels or self.etype != "any":
            ids = [eid for eid, label, etype in self._subjects(store, loc_names)
                   if self.accepts(eid, label, etype)]
            self.ids = ids if len(ids) <= MAX_PUSHED_IDS else None
        elif self.excludes:
            bad = [eid for eid, label, etype in self._subjects(store, loc_names)
                   if not self.accepts(eid, label, etype)]
            self.exclude_ids = bad if len(bad) <= MAX_PUSHED_IDS else None

    def accepts(self, eid: str, label: str, etype: str) -> bool:
        """Whether a subject, as it will be shown, answers this question."""
        if self.only is not None and eid not in self.only:
            return False
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
            if subjects.accepts(eid, label,
                                row["type"] if row is not None else "scene"):
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


# --- retrieval and fusion (search plan Phase 6) ------------------------------
#
# Two indexes now answer a word question, and they fail in opposite directions.
# The lexical one is exact and literal: it finds the row that used the asker's
# word and nothing else, so it is silent whenever the memory happened to write
# the thing down differently. The semantic one is the reverse — it finds the row
# that MEANS the same thing and cannot tell you which word made it a match, so
# on its own it is confident about rows nobody would accept.
#
# Reciprocal Rank Fusion is what lets both be wrong without the answer being
# wrong. It scores by POSITION rather than by score, which matters because bm25
# and cosine are not on a common scale and never can be: bm25 is relative to a
# corpus, cosine is an angle. Ranks are comparable by construction, so no
# calibration step exists to drift.
#
# What fusion is not allowed to do: widen. Both lists are drawn from the same
# hard-filtered candidates, so a ranker can only reorder what the window, the
# zone, the camera and the subject already permitted.

MAX_HITS_SHOWN = 6              # per entity; collapsed first, then best-scored


def _fuse(*ranked_lists: tuple[str, list[str]]) -> tuple[dict, dict]:
    """RRF over `(name, [ids in rank order])`. Returns (scores, reasons).

    `score = Σ 1/(k + rank)`, the standard form. With one list this is
    order-preserving — it is 1/(k+rank), which is monotone in rank — so a
    deployment with no semantic index gets EXACTLY the ordering Phase 2 gave it.
    That property is why the fusion is unconditional rather than switched on
    when a second list exists: one code path, and no configuration under which
    the ranking silently becomes a different algorithm.

    `reasons` records which index found a row, and is carried through to the
    answer. A hit nobody can explain is the thing this layer is most at risk of
    producing, so every hit says whether a word matched, a meaning matched, or
    both agreed.
    """
    k = float(CONFIG.semantic.rrf_k)
    scores: dict[str, float] = {}
    reasons: dict[str, str] = {}
    for name, ids in ranked_lists:
        for rank, oid in enumerate(ids, start=1):
            scores[oid] = scores.get(oid, 0.0) + 1.0 / (k + rank)
            prev = reasons.get(oid)
            reasons[oid] = name if prev is None else ("both" if prev != name else prev)
    return scores, reasons


def _retrieve(store: Store, text_q: str, pred_q: str, *, plan: dict, loc_ids,
              cam_ids, entity_ids) -> tuple[dict, dict, dict]:
    """The ranked candidate ids for a word question, fused.

    Returns (scores, reasons, summary). The summary goes into the trace, and it
    earns its place: "no results" and "the semantic index has never been built"
    look identical from the outside, and the second one is fixed by running a
    backfill rather than by rephrasing the question.

    `text` goes to both indexes; `predicate_contains` goes only to the lexical
    one, and that is a deliberate asymmetry rather than an oversight. `text` is
    the search surface — the words a person typed, meant as prose. Phase 4
    removed `predicate_contains` from the planner's tool schema entirely; what
    is left of it is the machine contract that `rule_fired:` is matched on, an
    exact-substring promise about an identifier. Handing an identifier to a
    meaning ranker asks it which predicates FEEL like 'rule_fired', which is not
    a question with an answer.
    """
    lexical = [r["observation_id"] for r in store.search_text(
        text_q or pred_q, since=plan["start"], until=plan["end"],
        location_ids=loc_ids, camera_ids=cam_ids, entity_ids=entity_ids)]
    meaning: list[str] = []
    if text_q:
        meaning = [oid for oid, _score in semantic.search(
            store, text_q, since=plan["start"], until=plan["end"],
            location_ids=loc_ids, camera_ids=cam_ids, entity_ids=entity_ids)]
    scores, reasons = _fuse(("lexical", lexical), ("semantic", meaning))
    return scores, reasons, {
        "lexical": len(lexical),
        "semantic": len(meaning),
        "fused": len(scores),
        # Read off the index rather than off the result, so an empty answer can
        # still say whether meaning was consulted at all.
        "semantic_index": store.embedded_count(store.EMBED_OBSERVATION,
                                               semantic.model_name()),
        "searched_with": text_q or pred_q,
    }


def _diverse_keyframes(frames: list[tuple[float, str]], k: int) -> list[str]:
    """`k` keyframes spread across the time they cover, not the first `k`.

    Rows arrive in timestamp order, so taking the first four gives four pictures
    of the first minute of an hour on camera — four frames of somebody walking
    in, and nothing of what they then did. Endpoints are always kept, because
    "when did this start" and "what did it look like by the end" are the two
    questions a strip of evidence is actually read for.

    Deterministic: the same rows always yield the same strip, which a stored
    conversation turn depends on to replay as what it showed at the time.
    """
    if k < 1 or not frames:
        return []
    if len(frames) <= k:
        return [url for _ts, url in frames]
    last = len(frames) - 1
    picks = dict.fromkeys(round(i * last / (k - 1)) for i in range(k))
    return [frames[i][1] for i in picks]


def _collapse(hits: list[dict], window: float) -> list[dict]:
    """Runs of the same predicate inside `window` seconds, as one hit with a count.

    A settled scene is re-described every few seconds, so one open gate becomes
    thirty identical rows. Listed individually they read as thirty events, and
    they crowd every other fact out of an answer that has a limit on it. Merged,
    they read as what they are: one thing, seen thirty times, over this stretch.

    The merged hit keeps the BEST score and the strongest reason in the run, not
    the first — the run is one claim, so it is as well-evidenced as its best
    evidence.
    """
    out: list[dict] = []
    for hit in sorted(hits, key=lambda h: (h["predicate"], h["first_seen"])):
        prev = out[-1] if out else None
        if (prev is not None and prev["predicate"] == hit["predicate"]
                and hit["first_seen"] - prev["last_seen"] <= window):
            prev["n"] += 1
            prev["last_seen"] = max(prev["last_seen"], hit["last_seen"])
            if hit["score"] is not None and (prev["score"] is None
                                             or hit["score"] > prev["score"]):
                prev["score"] = hit["score"]
                prev["observation_id"] = hit["observation_id"]
            if hit["match_reason"] and prev["match_reason"] != hit["match_reason"]:
                prev["match_reason"] = ("both" if prev["match_reason"]
                                        else hit["match_reason"])
            continue
        out.append(dict(hit))
    return sorted(out, key=lambda h: (h["score"] is None, -(h["score"] or 0.0),
                                      h["first_seen"]))


# --- the three aggregates (search plan Phase 5) ------------------------------
#
# Everything above answers out of `observations`. These three do not, and that is
# the whole gap: two of the system's three memory kinds — the distilled edges and
# the scene snapshots — were written every night and never once read back. "How
# often does that van come by?" was answerable from data that already existed,
# and the answer was a list of rows.
#
# All three keep the same contract as the row path: no claim without its
# supporting observation ids, and no aggregate that the hard filters did not
# reach. What changes is which table the evidence starts in.

MAX_HABITS_SHOWN = 6            # per entity; they are ordered by weight


def _named(store: Store, ids, loc_names: dict) -> dict:
    """`{entity_id: (display label, type)}` for a set of ids, in one statement.

    Skips what cannot be named, because `_label_for` returning None is the rule
    that a subject with no name cannot be an answer — it holds here for exactly
    the same reason it holds in the row loop.
    """
    ids = set(ids)
    rows = store.get_entities(ids)
    out = {}
    for eid in ids:
        row = rows.get(eid)
        label = _label_for(eid, row, loc_names)
        if label is not None:
            out[eid] = (label, row["type"] if row is not None else "scene")
    return out


def _co_presence(store: Store, plan: dict, subjects: "_SubjectFilter",
                 loc_ids, loc_names: dict) -> list[dict]:
    """Who was in frame alongside the subjects the question named.

    The anchor set is re-checked with `accepts()` rather than trusted from
    `subjects.ids`, and it has to be: `ids` is None both when the question named
    nobody *and* when it named too many people to bind, and those two mean
    opposite things. In the row path the loop catches that; here there is no
    loop over rows to catch it, so the check is done on the snapshot's members
    directly. Getting this wrong would answer "who was with Priya" with everyone
    who was ever with anyone.
    """
    pairs = store.co_presence(entity_ids=subjects.ids, since=plan["start"],
                              until=plan["end"], location_ids=loc_ids)
    if not pairs:
        return []
    known = _named(store, [p["entity_id"] for p in pairs]
                   + [p["with_entity_id"] for p in pairs], loc_names)

    companions: dict[str, dict] = {}
    for p in pairs:
        anchor, other = p["entity_id"], p["with_entity_id"]
        if anchor not in known or other not in known:
            continue
        if not subjects.accepts(anchor, *known[anchor]):
            continue
        label, etype = known[other]
        rec = companions.get(other)
        if rec is None:
            rec = companions[other] = {
                "entity_id": other, "label": label, "type": etype,
                "with": [], "zones": [], "snapshot_ids": [],
                "first_seen": p["timestamp"], "last_seen": p["timestamp"],
            }
        if anchor not in rec["with"]:
            rec["with"].append(anchor)
        zone = loc_names.get(p["location_id"], p["location_id"])
        if zone and zone not in rec["zones"]:
            rec["zones"].append(zone)
        if p["snapshot_id"] not in rec["snapshot_ids"]:
            rec["snapshot_ids"].append(p["snapshot_id"])
        rec["first_seen"] = min(rec["first_seen"], p["timestamp"])
        rec["last_seen"] = max(rec["last_seen"], p["timestamp"])

    for rec in companions.values():
        rec["n_snapshots"] = len(rec["snapshot_ids"])
        rec["with"] = [{"entity_id": a, "label": known[a][0]} for a in rec["with"]]
    # Most shared frames first: how often two people are in the picture together
    # is the closest thing the evidence has to how much they were together.
    return sorted(companions.values(),
                  key=lambda r: (-r["n_snapshots"], r["first_seen"]))


def _bucket_order(group_by: str):
    """Tie-break within a bucket count, so equal counts do not order at random.

    Weekdays run Monday-first and hours and dates run in their natural order,
    which matters because the whole point of a recurrence answer is to be read.
    """
    if group_by == "weekday":
        return lambda b: WEEKDAY_ABBR.index(b)
    if group_by == "hour":
        return lambda b: int(b[:-1])
    return lambda b: b


def _recurrence(entities: list[dict], tallies: dict, plan: dict,
                store: Store, loc_ids, loc_names: dict) -> list[dict]:
    """How often, and when — computed from the rows, corroborated by the habits.

    Two sources, deliberately, because they fail in opposite directions. The
    counts come from the observations the question actually matched, so they are
    exact and available the moment something is recorded. The habits come from
    the nightly distillation, so they lag — but they are the corroborated,
    decaying claim that survives across windows, and they carry a weight saying
    how much repetition is behind them. A count of six with no habit means it
    happened six times; a count of six with a confirmed habit means it is what
    this subject does.

    Reporting only the counts would call any six coincidences a pattern.
    Reporting only the habits would answer "how often?" with nothing at all
    until the next nightly pass.
    """
    group_by = plan["group_by"]
    order = _bucket_order(group_by)
    out = []
    for e in entities:
        tally = tallies.get(e["entity_id"])
        if tally is None:
            continue
        counts = tally[group_by]
        groups = [{"bucket": b, "n": n} for b, n in
                  sorted(counts.items(), key=lambda kv: (-kv[1], order(kv[0])))]
        span_days = max((e["last_seen"] - e["first_seen"]) / 86400.0, 0.0)
        n = e["n_observations"]
        habits = [{
            "predicate": h["predicate"],
            "zone": loc_names.get(h["location_id"], h["location_id"]),
            "weight": round(h["weight"], 3),
            "status": h["status"],
            "n_supporting": len(json.loads(h["supporting_observation_ids"])),
            "last_reinforced_at": h["last_reinforced_at"],
        } for h in store.habits(entity_ids=[e["entity_id"]],
                                location_ids=loc_ids)[:MAX_HABITS_SHOWN]]
        hours = tally["hour"]
        top_hour = max(hours.items(), key=lambda kv: (kv[1], -int(kv[0][:-1]))) \
            if hours else None
        out.append({
            "entity_id": e["entity_id"], "label": e["label"],
            "n_sightings": n,
            # Distinct days, not rows. Ten frames of one visit is one visit, and
            # "how often" is a question about visits.
            "n_occasions": len(tally["days"]),
            "first_seen": e["first_seen"], "last_seen": e["last_seen"],
            "span_days": round(span_days, 2),
            "per_week": round(n / max(span_days / 7.0, 1.0), 2),
            "group_by": group_by,
            "groups": groups,
            "top": groups[0] if groups else None,
            "top_hour": {"bucket": top_hour[0], "n": top_hour[1]} if top_hour else None,
            "habits": habits,
        })
    return out


def _relations(store: Store, plan: dict, subjects: "_SubjectFilter",
               loc_ids, cam_ids, loc_names: dict) -> tuple[list[dict], dict]:
    """The distilled edges these subjects sit on, and the evidence under them.

    Returns (edges, evidence-by-other-end). The second half is why this is not
    just a table read: the thing at the far end of a 'uses' edge is usually an
    object that has no observations *of its own* — a forklift is never the
    subject of a row, it is only ever what somebody was near. So its entry is
    built from the edge's own `supporting_observation_ids`, which is the same
    provenance the graph UI walks when an operator asks why a belief exists.

    The window and the zone filter reach the edge through that evidence rather
    than through the edge's own columns, and that is the right end to filter
    from: an edge has one location and a lifetime of reinforcement behind it,
    so asking whether it falls in a window is only answerable by asking where
    its evidence does.
    """
    edges = store.relation_edges(subject_entity_ids=subjects.ids,
                                 kinds=["relation", "event"])
    if not edges:
        return [], {}

    anchors = _named(store, [e["subject_entity_id"] for e in edges], loc_names)
    kept = [e for e in edges
            if e["subject_entity_id"] in anchors
            and subjects.accepts(e["subject_entity_id"],
                                 *anchors[e["subject_entity_id"]])]
    if not kept:
        return [], {}

    supporting = {e["relation_id"]: json.loads(e["supporting_observation_ids"])
                  for e in kept}
    rows = _rows_by_id(store, {oid for ids in supporting.values() for oid in ids},
                       plan, loc_ids, cam_ids)

    # The far end: a thing, or failing that a place. A 'frequents' edge has no
    # object at all, and the place IS the answer to "where does she go".
    far = {e["relation_id"]: (e["object_entity_id"] or
                              (SCENE_PREFIX + e["location_id"] if e["location_id"]
                               else None)) for e in kept}
    known = _named(store, [f for f in far.values() if f], loc_names)

    out, evidence = [], {}
    for e in kept:
        other = far[e["relation_id"]]
        if other not in known:
            continue
        ev = [rows[oid] for oid in supporting[e["relation_id"]] if oid in rows]
        if not ev:
            continue        # every trace of it falls outside the window asked about
        out.append({
            "subject_entity_id": e["subject_entity_id"],
            "subject_label": anchors[e["subject_entity_id"]][0],
            "predicate": e["predicate"], "kind": e["kind"],
            "object_entity_id": other, "object_label": known[other][0],
            "zone": loc_names.get(e["location_id"], e["location_id"]),
            "weight": round(e["weight"], 3), "status": e["status"],
            "n_supporting": len(ev),
            "observation_ids": [o["observation_id"] for o in ev][:8],
        })
        evidence.setdefault(other, []).extend(ev)
    return out, evidence


def _rows_by_id(store: Store, ids, plan: dict, loc_ids, cam_ids) -> dict:
    """Observation rows for these ids that also satisfy the plan's hard filters.

    Chunked, because the ids come from however much evidence a belief has
    accumulated and SQLite binds one variable per id. Chunking is safe where
    narrowing a ranked list would not be: this is a set membership test, so the
    union of the chunks is the answer the single query would have given.
    """
    ids = list(ids)
    out = {}
    for i in range(0, len(ids), MAX_PUSHED_IDS):
        for row in store.observations(
                observation_ids=ids[i:i + MAX_PUSHED_IDS],
                since=plan["start"], until=plan["end"], location_ids=loc_ids,
                camera_ids=cam_ids, min_confidence=plan["min_confidence"],
                exclude_predicates=plan["exclude_predicates"]):
            out[row["observation_id"]] = row
    return out


def _entities_from_evidence(evidence: dict, loc_names: dict,
                            store: Store) -> list[dict]:
    """Entity entries built from supporting rows rather than from own sightings.

    Same shape the row loop produces, so a caller cannot tell which path an
    answer came down — and should not have to. What differs is the meaning of
    `n_observations`: here it counts the rows that *evidence* the claim, not
    the times the subject was seen.
    """
    known = _named(store, evidence, loc_names)
    out = []
    for eid, rows in evidence.items():
        if eid not in known:
            continue
        label, etype = known[eid]
        times = [r["timestamp"] for r in rows]
        frames, seen = [], set()
        for r in rows:
            url = _kf(r["source_ref"])
            if url and url not in seen:
                seen.add(url)
                frames.append((r["timestamp"], url))
        out.append({
            "entity_id": eid, "label": label, "type": etype,
            "first_seen": min(times), "last_seen": max(times),
            "duration_s": round(max(times) - min(times), 1),
            "n_observations": len(rows), "states": [], "rule_events": [],
            # Picked for time coverage, like the row path's — an edge's evidence
            # spans every reinforcement it ever had, so the first four frames of
            # it are four pictures of the day the belief started.
            "keyframes": _diverse_keyframes(sorted(frames),
                                            CONFIG.semantic.max_keyframes),
            "match_score": None, "match_reason": None, "hits": [],
        })
    return out


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

    # Phase 5's two re-aimings. Both run before the row query, because both
    # change *whose* rows the answer is about: the labels in the plan named the
    # anchor, and the answer is whatever the anchor turned out to be connected
    # to. Everything downstream is unchanged — the same aggregation, over a
    # different set of subjects.
    co_presence = relation_edges = None
    if plan["intent"] == "who_with":
        co_presence = _co_presence(store, plan, subjects, loc_ids, loc_names)
        subjects = _SubjectFilter(store, plan, loc_names,
                                  only={c["entity_id"] for c in co_presence})
    elif plan["intent"] == "relations":
        relation_edges, evidence = _relations(store, plan, subjects, loc_ids,
                                              cam_ids, loc_names)
        # The row path cannot answer this one. The far end of a 'uses' edge is
        # an object, and an object is never the *subject* of an observation —
        # a forklift has no sightings of its own, only rows about people near
        # it. So the entities are built from the edge's evidence instead.
        entities = _entities_from_evidence(evidence, loc_names, store)
        entities.sort(key=lambda e: (-e["n_observations"], e["first_seen"]))
        if plan["limit"] is not None:
            entities = entities[:plan["limit"]]
        return {"query": query, "plan": plan,
                "total_observations": sum(e["n_observations"] for e in entities),
                "entities": entities, "ranked": False,
                "relations": relation_edges}

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
    #
    # Phase 6 made this two shortlists fused into one. `scores` is no longer bm25
    # — it is the RRF score — and `reasons` says which index put each row there.
    # The union handed to SQL below is unchanged in shape and strictly wider in
    # content, which is the only direction a retrieval phase may move it.
    scores: dict[str, float] = {}
    reasons: dict[str, str] = {}
    retrieval = None
    if text_q or pred_q:
        scores, reasons, retrieval = _retrieve(
            store, text_q, pred_q, plan=plan, loc_ids=loc_ids, cam_ids=cam_ids,
            entity_ids=subjects.ids)

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
    seen_frames: dict[str, set] = defaultdict(set)   # dedupe before the ordering
    # Kept beside the entities rather than on them, so a question that is not
    # about recurrence carries none of this in its payload.
    want_recurrence = plan["intent"] == "how_often"
    tallies: dict[str, dict] = defaultdict(
        lambda: {"days": set(), "weekday": Counter(), "hour": Counter(),
                 "day": Counter()})
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
            if not subjects.accepts(eid, label, etype):
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
                # Which index reached this entity — 'lexical', 'semantic',
                # 'both', or None for a row nothing ranked. Phase 6 can surface
                # a row on meaning alone, so an answer has to be able to say
                # that is what happened rather than implying a word matched.
                "match_reason": None,
                "hits": [],
            }
        ent["n_observations"] += 1
        if hit is not None:
            # An entity is as relevant as its best-matching row. Summing would
            # rank whoever was on camera longest, which is presence, not answer.
            prev = ent["match_score"]
            ent["match_score"] = hit if prev is None else max(prev, hit)
            why = reasons.get(o["observation_id"])
            was = ent["match_reason"]
            ent["match_reason"] = why if was is None else (
                was if was == why else "both")
            ent["hits"].append({
                "predicate": o["predicate"], "text": o["text"] or o["predicate"],
                "first_seen": o["timestamp"], "last_seen": o["timestamp"], "n": 1,
                "score": hit, "match_reason": why,
                "observation_id": o["observation_id"],
            })
        ent["first_seen"] = min(ent["first_seen"], o["timestamp"])
        ent["last_seen"] = max(ent["last_seen"], o["timestamp"])
        if want_recurrence:
            # Bucketed on the deployment's clock, the same one `distill` cuts
            # habits with and `_facts` renders times in. A pattern reported in a
            # timezone nobody works in is not a pattern anybody recognises.
            ts, tally = o["timestamp"], tallies[eid]
            day = bucket_day(ts)
            tally["days"].add(day)
            tally["day"][day] += 1
            tally["weekday"][bucket_weekday(ts)] += 1
            tally["hour"][f"{bucket_hour(ts):02d}h"] += 1
        pred = o["predicate"]
        if pred.startswith("rule_fired:"):
            ent["rule_events"].append({
                "rule": pred[len("rule_fired:"):], "timestamp": o["timestamp"],
                "keyframe": _kf(o["source_ref"]), "observation_id": o["observation_id"],
            })
        elif o["origin"] == "vlm" and pred not in ent["states"]:
            ent["states"].append(pred)
        if o["source_ref"]:
            # Carried with its timestamp, because Phase 6 picks the strip for
            # time coverage rather than taking whatever came first. Deduped by
            # URL, so one keyframe shared by six rows of the same report is one
            # picture, and it is dated by the first row that cited it.
            url = _kf(o["source_ref"])
            if url not in seen_frames[eid]:
                seen_frames[eid].add(url)
                ent["keyframes"].append((o["timestamp"], url))

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

    cfg = CONFIG.semantic
    for e in entities:
        e["duration_s"] = round(e["last_seen"] - e["first_seen"], 1)
        # Sorted by time here rather than at collection: rows arrive in the
        # order the question asked for, and a `desc` question would otherwise
        # hand the picker a reversed list and get its endpoints backwards.
        e["keyframes"] = _diverse_keyframes(sorted(e["keyframes"]),
                                            cfg.max_keyframes)
        e["states"] = e["states"][:8]
        e["hits"] = _collapse(e["hits"], cfg.collapse_seconds)[:MAX_HITS_SHOWN]
        for h in e["hits"]:
            h["score"] = None if h["score"] is None else float(f"{h['score']:.6g}")
        if e["match_score"] is not None:
            e["match_score"] = float(f"{e['match_score']:.6g}")

    # Summed over what is actually being returned, so the count can never
    # describe rows the answer does not show — which is what it would do if the
    # loop's running total survived a `limit`.
    total = sum(e["n_observations"] for e in entities)
    result = {"query": query, "plan": plan, "total_observations": total,
              "entities": entities, "ranked": bool(text_q or pred_q)}
    if retrieval is not None:
        # Part of the trace, not part of the answer. It says how the candidates
        # were found — which is the difference between "memory does not contain
        # this" and "the index that would have found it was never built".
        result["retrieval"] = retrieval
    if plan["intent"] == "count":
        # The same number the evidence already carried, promoted to the answer.
        # "How many" and "who" run the identical query — what differs is which
        # part is the reply, and a caller cannot infer that from the rows.
        result["count"] = total
    if want_recurrence:
        result["recurrence"] = _recurrence(entities, tallies, plan, store,
                                           loc_ids, loc_names)
    if co_presence is not None:
        # Carried whole, not narrowed to the entities above. A companion whose
        # own sightings fall outside the window still shared the frame, and the
        # snapshot is the evidence of that — dropping them here would answer
        # "nobody" to a question the snapshot can answer.
        result["co_presence"] = co_presence
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
    "A row marked matched_by 'semantic' was found by MEANING, not by the words asked: "
    "say what was actually recorded ('the closest thing recorded is \"standing around, "
    "waiting\"') rather than repeating the question's wording back as if the memory used "
    "it. Every fact "
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
            # Phase 6. A row reached by meaning alone used different words from
            # the ones asked, so the prose must be able to say "the closest
            # thing recorded was..." rather than implying the memory used the
            # asker's phrasing. Omitted entirely for an unranked question, where
            # there is nothing to explain.
            **({"matched_by": e["match_reason"]} if e.get("match_reason") else {}),
            **({"what_matched": [
                f"{h['text']}" + (f" (x{h['n']})" if h["n"] > 1 else "")
                for h in e["hits"]]} if e.get("hits") else {}),
        } for e in result["entities"]],
    }
    if result.get("count") is not None:
        # Named separately from `total_observations` because the question asked
        # for it. Same number, but the model is being told which one is the reply.
        facts["count"] = result["count"]
    # Phase 5's aggregates, projected the same way: pre-formatted, no ids. The
    # narrator is given the pattern already computed rather than the rows to
    # count, for the same reason it is given formatted times — it phrases these,
    # it cannot recompute them wrong.
    for r in result.get("recurrence") or []:
        facts.setdefault("recurrence", []).append({
            "who": r["label"],
            "times_seen": r["n_sightings"],
            "separate_days": r["n_occasions"],
            "over_days": r["span_days"],
            "roughly_per_week": r["per_week"],
            "grouped_by": r["group_by"],
            "most_often": (f"{r['top']['bucket']} ({r['top']['n']} of "
                           f"{r['n_sightings']})") if r["top"] else None,
            "usual_hour": r["top_hour"]["bucket"] if r["top_hour"] else None,
            # Weight and status included on purpose: a candidate habit is a
            # weaker claim than a confirmed one, and the prose should be able
            # to say so instead of stating both as fact.
            "mined_habits": [f"{h['predicate']} ({h['status']}, weight "
                             f"{h['weight']}, {h['n_supporting']} observations)"
                             for h in r["habits"]],
        })
    for c in result.get("co_presence") or []:
        facts.setdefault("seen_together", []).append({
            "who": c["label"],
            "with": [w["label"] for w in c["with"]],
            "times_in_frame_together": c["n_snapshots"],
            "where": c["zones"],
            "first_seen": t(c["first_seen"]), "last_seen": t(c["last_seen"]),
        })
    for e in result.get("relations") or []:
        facts.setdefault("connections", []).append({
            "subject": e["subject_label"],
            "connection": e["predicate"].replace("_", " "),
            "to": e["object_label"],
            "confidence": f"{e['status']}, weight {e['weight']}",
            "supported_by": e["n_supporting"],
        })
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
    for r in result.get("recurrence") or []:
        top = f", most often {r['top']['bucket']} ({r['top']['n']})" if r["top"] else ""
        hour = f" around {r['top_hour']['bucket']}" if r["top_hour"] else ""
        lines.append(f"how often: {r['label']} — {r['n_sightings']} sighting(s) on "
                     f"{r['n_occasions']} day(s) over {r['span_days']}d"
                     f"{top}{hour}.")
        for h in r["habits"]:
            lines.append(f"  habit: {h['predicate']}  weight {h['weight']} "
                         f"({h['status']}, {h['n_supporting']} obs)")
    for c in result.get("co_presence") or []:
        with_who = ", ".join(w["label"] for w in c["with"])
        lines.append(f"with: {c['label']} — alongside {with_who} in "
                     f"{c['n_snapshots']} frame(s) @ {', '.join(c['zones'])}")
    for e in result.get("relations") or []:
        lines.append(f"edge: {e['subject_label']} {e['predicate']} "
                     f"{e['object_label']}  weight {e['weight']} "
                     f"({e['status']}, {e['n_supporting']} obs)")
    lines.append(f"{len(result['entities'])} entit(y/ies), "
                 f"{result['total_observations']} observation(s) in window.")
    for e in result["entities"]:
        why = f"  [{e['match_reason']}]" if e.get("match_reason") else ""
        lines.append(f"\n{e['label']}  ({e['entity_id']}){why}")
        lines.append(f"  seen {t(e['first_seen'])} -> {t(e['last_seen'])}"
                     f"  ({e['duration_s']}s, {e['n_observations']} obs)")
        for h in e.get("hits") or []:
            times = f" x{h['n']}" if h["n"] > 1 else ""
            lines.append(f"  match: {h['text']}{times}  "
                         f"({h['match_reason']}, {h['score']})")
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
