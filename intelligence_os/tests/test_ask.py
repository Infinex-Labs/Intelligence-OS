"""Acceptance gate for Ask's deterministic aggregation (M1 step 7).

plan_query needs a key and isn't tested here; execute() is where the §8.2
guarantee lives (facts = rows), so that's what gets the offline gate. Run:
    .venv/bin/python -m intelligence_os.tests.test_ask
"""
import os
import tempfile

from intelligence_os.store import Store
from intelligence_os.ask import execute, _facts, _history_messages


def main():
    # explicit temp path, like test_foundation (env redirect is too late: the
    # package imports config at import time)
    store = Store(os.path.join(tempfile.mkdtemp(), "test.db"))
    bay = store.upsert_location("loading_bay", {"polygon": [[0, 0], [1, 0], [1, 1]]})
    aisle = store.upsert_location("aisle_3", {"polygon": [[2, 2], [3, 2], [3, 3]]})
    p1 = store.create_entity("person", label="Person 47")
    p2 = store.create_entity("person", label="Bob")  # labelled: its random id-fallback
    #   ("Person <hex>") could otherwise contain "person 47" and break the filter below

    # p1: present in bay 1000-1600, rule fired at 1500 with keyframe, vlm state
    store.add_observation(p1, "present", location_id=bay, timestamp=1000, source_ref="/x/kf_a.jpg")
    store.add_observation(p1, "present", location_id=bay, timestamp=1600)
    store.add_observation(p1, "rule_fired:linger", location_id=bay, timestamp=1500,
                          origin="rule", source_ref="/x/rule_linger_1.jpg")
    store.add_observation(p1, "state:smoking", location_id=bay, timestamp=1550, origin="vlm")
    # p2: only in the aisle, and outside the window
    store.add_observation(p2, "present", location_id=aisle, timestamp=5000)

    # zone + window scoping
    r = execute(store, {"start": 900, "end": 2000, "zone": "loading_bay"})
    assert len(r["entities"]) == 1, r
    e = r["entities"][0]
    assert e["label"] == "Person 47" and e["n_observations"] == 4, e
    assert e["first_seen"] == 1000 and e["last_seen"] == 1600 and e["duration_s"] == 600.0, e
    assert e["rule_events"] == [{"rule": "linger", "timestamp": 1500,
                                 "keyframe": "/keyframe/rule_linger_1.jpg",
                                 "observation_id": e["rule_events"][0]["observation_id"]}], e
    assert e["states"] == ["state:smoking"], e
    assert "/keyframe/kf_a.jpg" in e["keyframes"], e
    print("  zone+window scoping, rule events, states, keyframes OK")

    # window excludes everything -> empty, no invention
    r = execute(store, {"start": 10_000, "end": None, "zone": None})
    assert r["entities"] == [] and r["total_observations"] == 0, r
    print("  empty window -> empty answer OK")

    # predicate filter finds the smoking observation only
    r = execute(store, {"start": None, "end": None, "zone": None,
                        "predicate_contains": "smok"})
    assert r["total_observations"] == 1 and r["entities"][0]["states"] == ["state:smoking"], r

    # entity label filter
    r = execute(store, {"start": None, "end": None, "zone": None, "entity_label": "person 47"})
    assert len(r["entities"]) == 1 and r["entities"][0]["entity_id"] == p1, r
    print("  predicate + label filters OK")

    # the narration facts carry every countable claim and no invented ones:
    # what the model is allowed to say == what execute() found, nothing more.
    r = execute(store, {"start": 900, "end": 2000, "zone": "loading_bay"})
    f = _facts(r)
    assert f["total_observations"] == r["total_observations"], f
    assert len(f["entities"]) == 1, f
    fe = f["entities"][0]
    assert fe["who"] == "Person 47" and fe["times_observed"] == 4, fe
    assert fe["states"] == ["smoking"], fe                       # state:/underscores stripped
    assert fe["rule_events"] == [{"rule": "linger", "at": fe["rule_events"][0]["at"]}], fe
    assert "entity_id" not in fe and "keyframes" not in fe, "no ids/urls leak into prose"
    print("  narration facts grounded to the evidence OK")

    # conversation history is client-supplied: clamp depth, tolerate junk shapes.
    assert _history_messages(None) == [] and _history_messages([]) == []
    big = [{"q": f"q{i}", "a": f"a{i}"} for i in range(20)]
    m = _history_messages(big)
    assert len(m) == 12, m                       # last 6 turns -> 6 user + 6 assistant
    assert m[0] == {"role": "user", "content": "q14"}, m[0]
    assert m[-1]["role"] == "assistant" and m[-1]["content"] == "a19", m[-1]
    # missing answer still pairs (Anthropic requires alternating roles); junk skipped
    assert _history_messages([{"q": "hi"}]) == \
        [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "(no answer)"}]
    assert _history_messages([{"a": "orphan"}, {}]) == [], "no question -> no turn"
    print("  conversation history clamped + shaped OK")

    store.close()
    print("test_ask: ALL PASS")


if __name__ == "__main__":
    main()
