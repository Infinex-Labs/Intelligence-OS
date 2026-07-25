"""Acceptance gate for the rule compiler trust-boundary + cascade engine (M1).

No key, no YOLO, no network: the compiler gates are tested via _apply_gates on
fabricated compiler output, and the engine via fake resolved-detections. Isolation
follows the test_foundation pattern: explicit temp Store path + frames_dir (the
package imports config at import time, so env-var redirection is too late). Run:
    .venv/bin/python -m intelligence_os.tests.test_rules
"""
import os
import tempfile
import types
from pathlib import Path

import numpy as np
from intelligence_os.store import Store
from intelligence_os import rules as R

os.environ.pop("IDENTITY_ENABLED", None)
_TMP = Path(tempfile.mkdtemp())
FRAME = np.zeros((120, 120, 3), np.uint8)


def _store() -> Store:
    return Store(_TMP / f"test_{len(os.listdir(_TMP))}.db")


def _engine(store, rules, verifier=None) -> R.RuleEngine:
    return R.RuleEngine(store, rules=rules, verifier=verifier, frames_dir=_TMP / "frames")


def _rd(entity_id, cls, bbox, loc):
    det = types.SimpleNamespace(cls_name=cls, bbox=bbox)
    return types.SimpleNamespace(entity_id=entity_id, det=det, location_id=loc)


def test_compiler_gates():
    poly = "person"
    # infeasible -> loud refusal (§9.2)
    r = R._apply_gates({"feasible": False, "reason": "stress isn't visible",
                        "name": "x", "targets_individual": False,
                        "trigger_class": poly, "needs_verify": False}, "if stressed")
    assert isinstance(r, R.Refusal) and r.reason == "infeasible", r

    # identity-scoped + identity OFF -> refuse (§11)
    r = R._apply_gates({"feasible": True, "targets_individual": True,
                        "identity_note": "track one person leaving", "name": "wife",
                        "trigger_class": poly, "needs_verify": False}, "when my wife leaves")
    assert isinstance(r, R.Refusal) and r.reason == "needs_identity", r

    # identity-scoped + identity ON -> compiles
    os.environ["IDENTITY_ENABLED"] = "true"
    try:
        r = R._apply_gates({"feasible": True, "targets_individual": True,
                            "identity_note": "track one person", "name": "wife",
                            "trigger_class": poly, "needs_verify": False}, "when my wife leaves")
        assert isinstance(r, dict) and r["needs_identity"] is True, r
    finally:
        os.environ.pop("IDENTITY_ENABLED", None)

    # geometry-only feasible rule -> no verify block
    r = R._apply_gates({"feasible": True, "targets_individual": False, "name": "zone_dwell",
                        "trigger_class": poly, "trigger_zone": "loading_bay",
                        "dwell_seconds": 10, "needs_verify": False}, "person lingers")
    assert r["trigger"]["zone"] == "loading_bay" and r["verify"] is None, r

    # visual rule -> verify prompt carried through
    r = R._apply_gates({"feasible": True, "targets_individual": False, "name": "smoke",
                        "trigger_class": poly, "needs_verify": True,
                        "verify_prompt": "smoking? yes/no/unclear."}, "smoking")
    assert r["verify"]["prompt"].startswith("smoking"), r
    print("  compiler gates OK")


def test_engine_dwell_cooldown():
    store = _store()
    lid = store.upsert_location("loading_bay",
                                {"polygon": [[0, 0], [100, 0], [100, 100], [0, 100]]})
    eid = store.create_entity("person")
    walker = store.create_entity("person")

    rule = {"name": "linger", "trigger": {"class": "person", "zone": "loading_bay"},
            "gate": {"dwell_seconds": 10}, "verify": None,
            "cooldown": {"per_track_seconds": 300}}
    eng = _engine(store, [rule])
    rd = _rd(eid, "person", (10, 10, 40, 90), lid)

    assert eng.feed([rd], FRAME, 1000.0) == []      # 0s dwell
    assert eng.feed([rd], FRAME, 1005.0) == []      # 5s dwell
    fired = eng.feed([rd], FRAME, 1011.0)           # 11s -> fires
    assert len(fired) == 1 and fired[0].rule == "linger", fired
    assert store.get_entity(eid) is not None
    # event was written with a keyframe
    obs = [o for o in store.observations(eid) if o["origin"] == "rule"]
    assert obs and obs[0]["source_ref"], obs

    assert eng.feed([rd], FRAME, 1012.0) == []      # cooldown suppresses

    # a passer-by (2s) never fires
    rw = _rd(walker, "person", (10, 10, 40, 90), lid)
    assert eng.feed([rw], FRAME, 1011.0) == []
    assert eng.feed([rw], FRAME, 1013.0) == []
    store.close()
    print("  engine dwell + cooldown + passer-by OK")


def test_engine_zone_and_verify():
    store = _store()
    lid = store.upsert_location("bay", {"polygon": [[0, 0], [50, 0], [50, 50], [0, 50]]})
    eid = store.create_entity("person")

    # leaving the zone restarts the dwell clock
    rule = {"name": "z", "trigger": {"class": "person", "zone": "bay"},
            "gate": {"dwell_seconds": 10}, "verify": None,
            "cooldown": {"per_track_seconds": 300}}
    eng = _engine(store, [rule])
    rd_in = _rd(eid, "person", (5, 5, 20, 40), lid)
    rd_out = _rd(eid, "person", (5, 5, 20, 40), "elsewhere")
    assert eng.feed([rd_in], FRAME, 0.0) == []
    assert eng.feed([rd_in], FRAME, 8.0) == []
    assert eng.feed([rd_out], FRAME, 9.0) == []     # left zone -> reset
    assert eng.feed([rd_in], FRAME, 12.0) == []     # only 3s since re-entry

    # verify rule: fires only on 'yes'; never without a verifier
    vrule = {"name": "smoke", "trigger": {"class": "person", "zone": None},
             "gate": {"dwell_seconds": 0}, "verify": {"prompt": "smoking?"},
             "cooldown": {"per_track_seconds": 300}}
    rd = _rd(eid, "person", (5, 5, 20, 40), None)
    assert _engine(store, [vrule], verifier=lambda f, b, p, n=None: "no").feed([rd], FRAME, 500) == []
    assert _engine(store, [vrule]).feed([rd], FRAME, 600) == []
    fired = _engine(store, [vrule], verifier=lambda f, b, p, n=None: "yes").feed([rd], FRAME, 700)
    assert len(fired) == 1, fired

    # a 'no' verify is NOT re-asked every frame (VLM cost throttle)
    calls = []
    eng = _engine(store, [vrule], verifier=lambda f, b, p, n=None: calls.append(p) or "no")
    for t in (800, 801, 802, 810):
        assert eng.feed([rd], FRAME, t) == []
    assert len(calls) == 1, calls                    # throttled within 30s
    assert eng.feed([rd], FRAME, 800 + 31) == []
    assert len(calls) == 2, calls                    # re-asked after retry window
    store.close()
    print("  engine zone-reset + verify gating + retry throttle OK")


def test_correction_loop():
    """§9.3 (M5): a dismissed fire's crop is fed back to the verifier as a negative."""
    import cv2
    store = _store()
    frames = _TMP / "frames"
    frames.mkdir(parents=True, exist_ok=True)
    eid = store.create_entity("person")

    # a past fire of rule 'smoke' with its saved crop, then an operator dismissal of it
    ts = 1000
    kf = frames / f"rule_smoke_{int(ts * 1000)}.jpg"
    cv2.imwrite(str(kf), FRAME)
    cv2.imwrite(R._crop_path(kf), FRAME[:40, :20])          # the negative crop
    store.add_observation(eid, "rule_fired:smoke", source_ref=str(kf),
                          origin="rule", timestamp=ts)
    store.add_observation(eid, f"digest:dismissed:rule:smoke:{eid}:{int(ts)}",
                          origin="operator")

    # the linkage resolves to the saved crop
    crops = R.dismissed_crops(store, "smoke")
    assert crops == [R._crop_path(kf)], crops

    # the engine loads it and hands it to the verifier as `negatives`
    vrule = {"name": "smoke", "trigger": {"class": "person", "zone": None},
             "gate": {"dwell_seconds": 0}, "verify": {"prompt": "smoking?"},
             "cooldown": {"per_track_seconds": 300}}
    seen = {}
    def vf(f, b, p, n=None):
        seen["n"] = n
        return "no"
    eng = _engine(store, [vrule], verifier=vf)
    eng.feed([_rd(eid, "person", (5, 5, 20, 40), None)], FRAME, 2000)
    assert seen.get("n") and len(seen["n"]) == 1, seen
    store.close()
    print("  correction loop: dismissed crop fed back as negative OK")


if __name__ == "__main__":
    test_compiler_gates()
    test_engine_dwell_cooldown()
    test_engine_zone_and_verify()
    test_correction_loop()
    print("test_rules: ALL PASS")
