"""End-to-end memory pipeline test for Phases 3-6 (scene-state -> distillation).

Drives the memory layer with synthetic, scripted detections (no camera/models) so
the higher-layer logic is validated deterministically:
  - scene-state inventories accumulate over time and are queryable per-location
  - change detection emits new/gone correctly
  - an 'acquired' event is produced for an object that appears later (candidate)
  - relations (uses/frequents) and habits are mined and gated candidate->confirmed
  - a single VLM-style observation does NOT become a confirmed fact (corroboration)

Run: .venv/bin/python -m intelligence_os.tests.test_pipeline
"""
from __future__ import annotations

import tempfile
from pathlib import Path

from intelligence_os.detect import Detection
from intelligence_os.distill import Distiller
from intelligence_os.observe import Observer, ResolvedDetection
from intelligence_os.scene_state import SceneState, Zone
from intelligence_os.store import Store

DAY = 86400.0


def check(name: str, cond: bool):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    assert cond, name


def _person_det(bbox):
    return Detection("person", bbox, conf=0.9, track_id=1)


def _obj_det(name, bbox, tid):
    d = Detection(name, bbox, conf=0.8, track_id=tid)
    return d


def test_pipeline():
    print("test_pipeline (scene-state -> change detection -> distillation)")
    path = Path(tempfile.mkdtemp()) / "p.db"
    store = Store(path)

    # one location: the desk
    desk = store.upsert_location("desk", {"polygon": [[0, 0], [100, 0], [100, 100], [0, 100]]})
    scene = SceneState(store, [Zone(desk, "desk", [[0, 0], [100, 0], [100, 100], [0, 100]])])
    observer = Observer(store)
    # disable cooldown for the scripted test so each scripted sighting writes
    observer.cooldown = 0.0

    person = store.create_entity("person", label="raj")
    laptop = store.create_entity("object", label="laptop")
    chair = store.create_entity("object", label="chair")

    # Day 0..4: raj present at the desk near the laptop every "day" at ~09:00.
    # chair only appears from day 2 onward (-> should distill as 'acquired').
    base = 0.0  # epoch 0 -> gmtime hour 0; we just need distinct days + a stable hour
    for day in range(5):
        t = base + day * DAY + 9 * 3600
        resolved = [
            ResolvedDetection(person, _person_det((10, 10, 30, 90)), desk),
            ResolvedDetection(laptop, _obj_det("laptop", (35, 40, 60, 70), 2), desk),
        ]
        present = [person, laptop]
        if day >= 2:
            resolved.append(ResolvedDetection(chair, _obj_det("chair", (70, 20, 95, 95), 3), desk))
            present.append(chair)
        observer.observe_frame(resolved, t, source_ref=f"frame_{day}")
        scene.snapshot({desk: present}, t)

    # scene-state queryable per-location/time
    check("desk inventory on day 0 has no chair",
          chair not in scene.inventory_at(desk, base + 0 * DAY + 9 * 3600 + 1))
    check("desk inventory on day 4 has chair",
          chair in scene.inventory_at(desk, base + 4 * DAY + 9 * 3600 + 1))

    # distill
    result = Distiller(store).run()
    print(f"    distill result: {result}")

    # change detection saw the chair appear
    events = store.relations(kind="event")
    acquired = [e for e in events if e["predicate"] == "acquired"
                and e["subject_entity_id"] == chair]
    check("chair distilled as an 'acquired' event", len(acquired) == 1)
    check("acquired event is a CANDIDATE, not auto-confirmed (corroboration gate)",
          acquired[0]["status"] == "candidate")
    check("acquired event links supporting evidence",
          len(__import__("json").loads(acquired[0]["supporting_observation_ids"])) >= 1)

    # relation mining: raj uses laptop (repeated 'near' -> confirmed)
    uses = [r for r in store.relations(kind="relation")
            if r["predicate"] == "uses" and r["object_entity_id"] == laptop]
    check("raj 'uses' laptop relation mined", len(uses) == 1)
    check("'uses' confirmed after repeated evidence", uses[0]["status"] == "confirmed")

    # frequents the desk
    frequents = [r for r in store.relations(kind="relation") if r["predicate"] == "frequents"]
    check("raj 'frequents' desk mined", any(r["subject_entity_id"] == person for r in frequents))

    # habit: present around 09h
    habits = store.relations(kind="habit")
    check("daily 09h presence mined as a habit",
          any("09h" in h["predicate"] for h in habits))

    store.close()


def test_single_vlm_claim_not_confirmed():
    print("test_single_vlm_claim_not_confirmed (NFR-4 corroboration)")
    path = Path(tempfile.mkdtemp()) / "v.db"
    store = Store(path)
    p = store.create_entity("person")
    # one overconfident VLM observation
    o = store.add_observation(p, "state:sleeping", confidence=0.95, origin="vlm")
    # distillation would only confirm with repeated evidence; a single obs stays weak
    rid = store.reinforce_relation("habit", p, "sleeping",
                                   supporting_observation_ids=[o], obs_confidence=0.95)
    rel = store.relations(kind="habit")[0]
    check("single VLM-derived claim stays candidate", rel["status"] == "candidate")
    store.close()


def test_scene_description_context_analysis():
    print("test_scene_description_context_analysis")
    from intelligence_os.vlm import SceneDescription
    desc = SceneDescription(raw={
        "people": [
            {"ref": "ent_raj_1", "state": "sleeping", "context_analysis": "sleeping instead of working at desk"}
        ]
    }, model="mock")
    states = desc.states()
    check("context_analysis is flattened to state list", 
          ("ent_raj_1", "context_analysis:sleeping instead of working at desk") in states)
    check("state is also flattened to state list",
          ("ent_raj_1", "state:sleeping") in states)


def main():
    tests = [test_pipeline, test_single_vlm_claim_not_confirmed, test_scene_description_context_analysis]
    for t in tests:
        t()
    print(f"\nAll {len(tests)} pipeline tests passed.")


if __name__ == "__main__":
    main()
