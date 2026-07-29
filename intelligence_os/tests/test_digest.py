"""Acceptance gate for the digest's banding + triage (M2).

Seeded store, no models. Explicit temp Store path (test_foundation pattern). Run:
    .venv/bin/python -m intelligence_os.tests.test_digest
"""
import os
import tempfile

from intelligence_os.store import Store
from intelligence_os.digest import build, feedback
from intelligence_os.distill import bucket_hour

DAY = 86400.0


def main():
    store = Store(os.path.join(tempfile.mkdtemp(), "test.db"))
    bay = store.upsert_location("loading_bay", {"polygon": [[0, 0], [1, 0], [1, 1]]})

    # p1 is OLD (created before the window) with a mined habit at the bay,
    # covering the hour its 10h-past-midnight-UTC sighting falls in LOCALLY.
    #
    # Hours used to be written here as literals because both sides of the
    # comparison were UTC. Since Phase 5 they are the deployment's local hours
    # (`distill.bucket_hour`), so the literal would only be right on a machine
    # in UTC — the test would pass in CI and fail on the developer's laptop
    # while the code was correct on both. Asking for the bucket is what makes
    # this assert the behaviour rather than the timezone.
    p1 = store.create_entity("person", label="Regular")
    store.conn.execute("UPDATE entities SET created_at=? WHERE entity_id=?", (0.0, p1))
    store.conn.commit()
    base = 30 * DAY  # windows land mid-epoch
    o1 = store.add_observation(p1, "present", location_id=bay, timestamp=base + 10 * 3600)
    store.reinforce_relation("habit", p1,
                             f"present_around_{bucket_hour(base + 10 * 3600):02d}h",
                             location_id=bay, supporting_observation_ids=[o1])
    # relation created_at is 'now' (wall clock) — pin event-band inputs instead:
    store.conn.execute("UPDATE relations SET created_at=? WHERE kind='habit'", (base,))
    store.conn.commit()

    since = base + 40 * DAY
    now = since + DAY

    # in-window: p1 present 3h past midnight UTC (7h off the habit, whatever
    # local hour that lands on) and at the habitual hour
    store.add_observation(p1, "present", location_id=bay, timestamp=since + 3 * 3600,
                          source_ref="/x/kf1.jpg")
    store.add_observation(p1, "present", location_id=bay, timestamp=since + 10 * 3600)
    # in-window: a rule fires on p1
    store.add_observation(p1, "rule_fired:linger", location_id=bay, origin="rule",
                          timestamp=since + 4 * 3600, source_ref="/x/rule1.jpg")
    # in-window: a brand-new entity appears
    p2 = store.create_entity("person")
    store.conn.execute("UPDATE entities SET created_at=? WHERE entity_id=?",
                       (since + 5 * 3600, p2))
    store.conn.commit()
    store.add_observation(p2, "present", location_id=bay, timestamp=since + 5 * 3600,
                          source_ref="/x/kf2.jpg")

    d = build(store, since=since, now=now)

    # band 1: the rule fire, with keyframe
    assert len(d["rule_fired"]) == 1, d["rule_fired"]
    r = d["rule_fired"][0]
    assert r["title"] == "Rule fired: linger" and r["keyframe"] == "/keyframe/rule1.jpg", r

    # band 2: exactly two unusual items — the off-hour deviation + first-seen p2.
    # The habitual-hour presence matches the habit -> NOT flagged. p2 has no
    # habit baseline -> no unusual_hour item for it (value before baseline).
    odd = bucket_hour(since + 3 * 3600)
    sigs = {i["signature"] for i in d["unusual"]}
    assert sigs == {f"unusual_hour:{p1}:{bay}:{odd:02d}", f"first_seen:{p2}"}, sigs
    uh = next(i for i in d["unusual"] if i["signature"].startswith("unusual_hour"))
    assert uh["label"] == "Regular" and uh["keyframe"] == "/keyframe/kf1.jpg", uh

    # band 3: routine counts exclude nothing silently (3 non-rule obs in window)
    assert d["routine"]["observations"] == 3, d["routine"]
    print("  three bands OK")

    # triage: dismiss the unusual-hour item -> gone next build; confirm rule -> gone
    feedback(store, uh["signature"], p1, "dismiss")
    feedback(store, r["signature"], p1, "confirm")
    d2 = build(store, since=since, now=now)
    assert d2["rule_fired"] == [], d2["rule_fired"]
    assert {i["signature"] for i in d2["unusual"]} == {f"first_seen:{p2}"}, d2["unusual"]
    print("  dismiss/confirm triage OK")

    # bad action refuses
    try:
        feedback(store, "x", p1, "meh")
        raise AssertionError("should have raised")
    except ValueError:
        pass
    store.close()
    print("test_digest: ALL PASS")


if __name__ == "__main__":
    main()
