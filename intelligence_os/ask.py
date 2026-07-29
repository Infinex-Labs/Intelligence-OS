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

QUERY_TOOL = {
    "name": "graph_query",
    "description": "Translate the user's question into a memory query.",
    "input_schema": {
        "type": "object",
        "properties": {
            "start": {"type": ["number", "null"],
                      "description": "window start, epoch seconds; null = no lower bound"},
            "end": {"type": ["number", "null"],
                    "description": "window end, epoch seconds; null = now"},
            "zone": {"type": ["string", "null"],
                     "description": "one of the known zone names, or null for anywhere"},
            "entity_label": {"type": ["string", "null"],
                             "description": "one of the known entity labels if the question "
                             "names one, else null"},
            # Phase 2 put a stemming index behind this field, which inverts the
            # old advice: 'smok' used to be the safe way to reach "smoking", and
            # now it is the way to miss it — porter stems 'smoking' to 'smoke'
            # and 'smok' to itself. Whole words, as the user said them.
            "predicate_contains": {"type": ["string", "null"],
                                   "description": "the words to search for, as whole words "
                                   "(e.g. 'smoking', 'cigarettes', 'gate open', "
                                   "'rule_fired'); do not truncate them. null if the "
                                   "question names no particular thing to look for"},
        },
        "required": ["start", "end", "zone", "entity_label", "predicate_contains"],
    },
}

PLAN_SYSTEM = (
    "You translate a question about what a camera-memory system saw into a "
    "structured query. You will be given the current time, the known zone names, "
    "and the known entity labels. Resolve relative times ('yesterday', 'around 3', "
    "'last Tuesday 2-4pm') into epoch seconds using the current time. Map any "
    "mentioned place to the closest known zone name (or null). Only use zone names "
    "and entity labels from the provided lists. Earlier turns of the conversation may "
    "precede the question — use them to resolve follow-ups ('what about yesterday?', "
    "'and her?', 'only the smoking ones') into a complete query on their own. Answer "
    "by calling graph_query."
)


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
    msg = client.messages.create(
        model=CONFIG.vlm.model,
        max_tokens=300,
        system=PLAN_SYSTEM,
        tools=[QUERY_TOOL],
        tool_choice={"type": "tool", "name": "graph_query"},
        messages=[*_history_messages(history), {"role": "user", "content": (
            f"Current time: {time.strftime('%A %Y-%m-%d %H:%M:%S', time.localtime(now))} "
            f"(epoch {now:.0f})\nKnown zones: {zones}\nKnown entity labels: {labels}\n\n"
            f"Question: {question}")}],
    )
    for block in msg.content:
        if getattr(block, "type", None) == "tool_use":
            return block.input
    raise RuntimeError("Could not parse the question into a query.")


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


def _subjects_labelled(store: Store, label_q: str, loc_names: dict) -> Optional[list[str]]:
    """Subject ids whose display label contains `label_q`, for the SQL filter.

    One pass over the entity table, which is bounded by how many things have
    ever been seen rather than by how often they were seen — the distinction
    that makes this affordable and made the per-row version not.

    `None` means "too many to push down": above the bound-variable cap the ids
    go back to being filtered in the loop. A label matching that many subjects
    was not narrowing anything anyway, so the fallback gives up nothing.
    """
    ids = [e["entity_id"] for e in store.list_entities(active_only=False)
           if label_q in (_label_for(e["entity_id"], e, loc_names) or "").lower()]
    ids += [sid for sid in store.scene_subjects()
            if label_q in (_label_for(sid, None, loc_names) or "").lower()]
    return None if len(ids) > MAX_PUSHED_IDS else ids


def execute(store: Store, query: dict) -> dict:
    """Deterministic aggregation over observation rows. Every claim in the output
    is a row (or an aggregate of rows) the caller can inspect."""
    start, end = query.get("start"), query.get("end")
    zone_name = query.get("zone")
    label_q = (query.get("entity_label") or "").strip().lower()
    pred_q = (query.get("predicate_contains") or "").strip().lower()
    text_q = (query.get("text") or "").strip().lower()

    loc_names = {r["location_id"]: r["name"] for r in store.locations()}
    loc_id = None
    if zone_name:
        for lid, name in loc_names.items():
            if name == zone_name:
                loc_id = lid
                break

    # Phase 2: the words go to the lexical index, which stems and ranks. The two
    # word fields differ in how strict they are, and deliberately so:
    #   `text`               — the search surface. Match or you are not a result.
    #   `predicate_contains` — the older field, still the machine contract that
    #                          'rule_fired' is matched on. It keeps its substring
    #                          behaviour AND gains the index, as a union: a phase
    #                          that widens recall must never take a hit away.
    # Both go through the same MATCH so ranking is comparable across them.
    loc_ids = [loc_id] if loc_id else None
    scores: dict[str, float] = {}
    if text_q or pred_q:
        for r in store.search_text(text_q or pred_q, since=start, until=end,
                                   location_ids=loc_ids):
            scores[r["observation_id"]] = r["score"]

    # Phase 3: the window, the zone and the words are all decided by SQLite now.
    # This used to read the whole table and drop rows in a Python loop, which
    # meant a question about one hour of one camera paid for every hour of every
    # camera ever recorded. The union that Phase 2 expressed as two conditions in
    # the loop is the same union, handed to `match_any` — the ranked ids OR the
    # substring, and never wider than the hard filters above.
    ent_ids = _subjects_labelled(store, label_q, loc_names) if label_q else None
    if text_q or pred_q:
        rows = store.observations(
            since=start, until=end, location_ids=loc_ids, entity_ids=ent_ids,
            observation_ids=list(scores),
            predicate_contains=pred_q if pred_q and not text_q else None,
            match_any=True)
    else:
        rows = store.observations(since=start, until=end, location_ids=loc_ids,
                                  entity_ids=ent_ids)

    # One statement for every entity these rows are about, instead of one per
    # row. A busy hour is thousands of rows about a dozen people.
    ents = store.get_entities({o["subject_entity_id"] for o in rows})

    per_entity: dict[str, dict] = {}
    rejected: set[str] = set()      # ids already judged: unresolvable, or wrong label
    total = 0
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
            # Still checked here even when SQL already narrowed by it: the label
            # filter is defined on the displayed name, and this loop is where
            # that name is decided.
            if label_q and label_q not in label.lower():
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
        total += 1
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
    else:
        entities = sorted(per_entity.values(), key=lambda e: e["first_seen"])
    for e in entities:
        e["duration_s"] = round(e["last_seen"] - e["first_seen"], 1)
        e["keyframes"] = e["keyframes"][:4]
        e["states"] = e["states"][:8]
        if e["match_score"] is not None:
            e["match_score"] = float(f"{e['match_score']:.6g}")
    return {"query": query, "total_observations": total, "entities": entities,
            "ranked": bool(text_q or pred_q)}


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
    return {
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
