#!/usr/bin/env python3
"""Example 1 — the memory graph, without a camera.

The vision stages are glue. *This* is the product: observations accumulate,
distillation turns repetition into durable memory, every claim keeps its
evidence, and an operator can correct the graph when it is wrong.

This script fabricates three days of observations for two people and a laptop,
then does everything the real pipeline would do downstream of the camera. No
webcam, no API key, no model weights — it runs on the three packages in
requirements-dev.txt.

    python examples/01_memory_graph/run.py
"""
import os
import sys
import tempfile
import time
from pathlib import Path

# Point the whole package at a throwaway data directory BEFORE importing it:
# config.py reads these at import time, so setting them afterwards is too late
# and you would write into your real memory graph.
_TMP = Path(tempfile.mkdtemp(prefix="ios-example-"))
os.environ["INTELLIGENCE_OS_DATA"] = str(_TMP)
os.environ["INTELLIGENCE_OS_DB"] = str(_TMP / "example.db")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from intelligence_os.store import Store          # noqa: E402
from intelligence_os.distill import Distiller    # noqa: E402

DAY = 86400
NOW = time.time()


def rule(title):
    print(f"\n\033[1m{title}\033[0m\n" + "─" * 64)


def seed(store):
    """Three mornings in a workshop, as the pipeline would have recorded them."""
    bay = store.upsert_location("loading_bay", region={"points": [[0, 0], [640, 0], [640, 480], [0, 480]]},
                                camera_id="front_door")
    desk = store.upsert_location("desk", region={"points": [[0, 0], [320, 0], [320, 240], [0, 240]]},
                                 camera_id="office")

    raj = store.create_entity("person", label="Raj")
    # Anonymous by default. A person only gets a name when an operator gives
    # them one — the graph works perfectly well with entity_2.
    stranger = store.create_entity("person")
    laptop = store.create_entity("object", label="laptop")

    for day in (3, 2, 1):
        # 09:xx each morning: Raj in the loading bay, then at his desk near the
        # laptop. Times below are UTC, because that is the bucket `mine_habits`
        # groups by — hence `present_around_09h` in the output.
        t = NOW - day * DAY
        morning = t - (t % DAY) + 9 * 3600
        for i in range(4):
            store.add_observation(raj, "present", location_id=bay,
                                  timestamp=morning + i * 300, confidence=0.9,
                                  source_ref=f"frames/{int(morning)}_{i}.jpg",
                                  camera_id="front_door", origin="detector")
        for i in range(4):
            store.add_observation(raj, "present", location_id=desk,
                                  timestamp=morning + 1800 + i * 300, confidence=0.9,
                                  source_ref=f"frames/{int(morning)}_desk_{i}.jpg",
                                  camera_id="office", origin="detector")
            store.add_observation(raj, "near", object_entity_id=laptop, location_id=desk,
                                  timestamp=morning + 1800 + i * 300, confidence=0.8,
                                  source_ref=f"frames/{int(morning)}_desk_{i}.jpg",
                                  camera_id="office", origin="detector")

    # ...and one visitor, once, yesterday afternoon
    store.add_observation(stranger, "present", location_id=bay,
                          timestamp=NOW - DAY + 15 * 3600, confidence=0.7,
                          source_ref="frames/visitor.jpg", camera_id="front_door",
                          origin="detector")
    return {"raj": raj, "stranger": stranger, "laptop": laptop, "bay": bay, "desk": desk}


def main():
    store = Store(os.environ["INTELLIGENCE_OS_DB"])
    ids = seed(store)

    rule("1. The transcript — every observation is subject → predicate → object")
    for o in store.observations()[:4]:
        subject = store.get_entity(o["subject_entity_id"])
        print(f"  {time.strftime('%a %H:%M', time.gmtime(o['timestamp']))}Z  "
              f"{subject['label'] or subject['entity_id']:<10} {o['predicate']:<8} "
              f"cam={o['camera_id']:<11} keyframe={o['source_ref']}")
    print(f"  ... {len(store.observations())} observations in total")

    rule("2. Distillation — repetition becomes durable memory")
    # The same pass `python -m intelligence_os.distill` runs.
    print("  ", Distiller(store).run())

    for r in store.relations():
        subject = store.get_entity(r["subject_entity_id"])
        target = ""
        if r["object_entity_id"]:
            target = f"→ {store.get_entity(r['object_entity_id'])['label']}"
        elif r["location_id"]:
            target = f"@ {store.conn.execute('SELECT name FROM locations WHERE location_id=?', (r['location_id'],)).fetchone()['name']}"
        print(f"  [{r['kind']:<8}] {subject['label'] or subject['entity_id']:<10} "
              f"{r['predicate']:<22} {target:<22} weight={r['weight']:.2f} {r['status']}")

    rule("3. Provenance — every claim walks back down to its evidence")
    # This is the guarantee the whole system exists to keep. A habit is not an
    # opinion the model formed; it is a pointer to the frames that produced it.
    habit = next(r for r in store.relations() if r["kind"] == "habit")
    import json
    supporting = json.loads(habit["supporting_observation_ids"])
    print(f"  claim:    {habit['predicate']} (weight {habit['weight']:.2f})")
    print(f"  based on: {len(supporting)} observations")
    for oid in supporting[:3]:
        o = store.get_observation(oid)
        print(f"    {time.strftime('%a %H:%M', time.gmtime(o['timestamp']))}Z  "
              f"keyframe {o['source_ref']}")

    rule("4. Correction — the graph is wrong sometimes, so it is editable")
    print(f"  entities before merge: {len(store.list_entities('person'))}")
    # Suppose the visitor turns out to have been Raj all along, badly lit.
    store.merge(ids["stranger"], ids["raj"])
    print(f"  after merge(stranger → Raj): {len(store.list_entities('person'))}")
    print(f"  Raj's observations now: {len(store.observations(ids['raj']))}"
          "   (the visitor's evidence moved, it wasn't discarded)")

    rule("5. Deletion — the privacy path, and it really cascades")
    removed = store.cascade_delete(ids["raj"])
    print(f"  cascade_delete(Raj) removed: {removed}")
    print(f"  observations left in the graph: {len(store.observations())}")

    store.close()
    print(f"\nScratch database was {_TMP} — nothing here touched your real memory graph.")


if __name__ == "__main__":
    main()
