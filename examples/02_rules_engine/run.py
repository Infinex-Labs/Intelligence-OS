#!/usr/bin/env python3
"""Example 2 — the rule cascade, and why it refuses things.

A rule in Intelligence OS is not an `if` statement over model output. It is a
cascade: zone match → dwell gate → optional VLM verification → per-subject
cooldown. Cheap checks gate expensive ones, and a rule that cannot be reliably
observed is refused at compile time rather than firing noise forever.

Rules are normally written in English and compiled by one LLM call. That needs
an API key, so this example skips the call and feeds the compiler's *output*
straight into the same trust boundary the real compiler goes through — which is
exactly how the acceptance tests do it.

    python examples/02_rules_engine/run.py
"""
import os
import sys
import tempfile
import time
import types
from pathlib import Path

_TMP = Path(tempfile.mkdtemp(prefix="ios-example-"))
os.environ["INTELLIGENCE_OS_DATA"] = str(_TMP)
os.environ["INTELLIGENCE_OS_DB"] = str(_TMP / "example.db")
os.environ.pop("IDENTITY_ENABLED", None)      # the default: face identity is off
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np                            # noqa: E402
from intelligence_os.store import Store       # noqa: E402
from intelligence_os import rules as R        # noqa: E402

FRAME = np.zeros((240, 320, 3), np.uint8)     # stands in for a real keyframe


def rule(title):
    print(f"\n\033[1m{title}\033[0m\n" + "─" * 68)


def detection(entity_id, cls, bbox, location_id):
    """What run.py hands the engine: a detection already resolved to an entity."""
    return types.SimpleNamespace(entity_id=entity_id,
                                 det=types.SimpleNamespace(cls_name=cls, bbox=bbox),
                                 location_id=location_id)


def main():
    store = Store(os.environ["INTELLIGENCE_OS_DB"])
    bay = store.upsert_location("loading_bay", region={"points": [[0, 0], [320, 0], [320, 240], [0, 240]]},
                                camera_id="front_door")
    person = store.create_entity("person")

    rule("1. The trust boundary — some rules are refused, loudly")
    # _apply_gates is the compiler's output check, split out so it is testable
    # without a key. Each dict below is what the LLM would have produced.
    refused_impossible = R._apply_gates(
        {"feasible": False, "reason": "stress is not something a camera can see",
         "name": "stressed", "targets_individual": False}, "alert me if someone looks stressed")
    print(f"  'if someone looks stressed'  → {type(refused_impossible).__name__}: "
          f"{refused_impossible.reason} — {refused_impossible.message}")

    refused_identity = R._apply_gates(
        {"feasible": True, "targets_individual": True, "name": "raj_leaves",
         "identity_note": "This rule tracks one specific person.",
         "trigger_class": "person", "needs_verify": False}, "tell me when Raj leaves")
    print(f"  'when Raj leaves'            → {type(refused_identity).__name__}: "
          f"{refused_identity.reason}")
    print("     (face identity is opt-in. A rule about a named individual will not")
    print("      compile until someone turns it on deliberately.)")

    spec = R._apply_gates(
        {"feasible": True, "targets_individual": False, "name": "loitering_at_the_bay",
         "trigger_class": "person", "trigger_zone": "loading_bay",
         "dwell_seconds": 30, "cooldown_seconds": 300, "needs_verify": False},
        "tell me if someone hangs around the loading bay")
    print(f"  'if someone hangs around'    → compiled: {spec['name']}")
    print(f"     trigger={spec['trigger']}  gate={spec['gate']}  cooldown={spec['cooldown']}")

    rule("2. The dwell gate — presence is not an alert")
    engine = R.RuleEngine(store, rules=[spec], frames_dir=_TMP / "frames")
    t0 = time.time()
    seen = detection(person, "person", (10, 10, 90, 200), bay)

    for elapsed in (0, 10, 29, 31, 45):
        fired = engine.feed([seen], FRAME, t0 + elapsed)
        state = "FIRED  ⚑" if fired else "quiet"
        print(f"  t+{elapsed:>3}s  person in loading_bay   {state}")
    print("     30s of continuous presence is the gate. Walking through does not")
    print("     trip it; standing there does.")

    rule("3. Cooldown — one alert per subject, not one per frame")
    fired_again = engine.feed([seen], FRAME, t0 + 60)
    print(f"  t+ 60s  still there            {'FIRED ⚑' if fired_again else 'quiet (cooling down)'}")
    print(f"     cooldown is {spec['cooldown']['per_track_seconds']:.0f}s per subject.")

    rule("4. What lands in memory")
    # A fired rule writes an observation with origin='rule' — so an alert is a
    # first-class part of the graph, with a keyframe, not a log line.
    for o in store.observations():
        if o["origin"] == "rule":
            print(f"  {time.strftime('%H:%M:%S', time.localtime(o['timestamp']))}  "
                  f"predicate={o['predicate']}  status={o['status']}  "
                  f"keyframe={o['source_ref']}")
    print("\n     Because it is an observation, it shows up in the timeline, it can be")
    print("     dismissed by an operator, and a dismissal becomes a negative example")
    print("     that sharpens the rule next time the pipeline starts.")

    store.close()
    print(f"\nScratch database was {_TMP} — your real memory graph was not touched.")


if __name__ == "__main__":
    main()
