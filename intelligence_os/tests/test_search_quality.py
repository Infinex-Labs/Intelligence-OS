"""Search quality gate (search plan Phase 0).

This suite exists to make "search got better" a diff instead of an opinion. It
scores a fixed corpus against a fixed case list and asserts the result matches
the **recorded baseline** in `search_cases.yaml`.

That means it is green today *while documenting failure* — most cases are
`baseline: fail`, and it asserts they still fail. Two ways it goes red:

  a case regresses   something that worked stopped working
  a case improves    a phase fixed it -> flip `baseline: fail` to `pass`

The second is a deliberate forcing function: nothing improves silently either,
so the scorecard in docs/search-baseline.md cannot drift from reality.

Run:  python -m intelligence_os.tests.test_search_quality
Score: python scripts/search_eval.py
"""
from __future__ import annotations

import json
import os
import tempfile
import time
import unittest
from pathlib import Path

from intelligence_os.tests import _stubs  # noqa: F401  (headless dep stubs)
from intelligence_os.store import Store
from intelligence_os.ask import execute
from intelligence_os.tests.fixtures import search_corpus

CASES_PATH = Path(__file__).parent / "fixtures" / "search_cases.yaml"

# --- what ask.execute() actually honours today -------------------------------
# A plan field outside this set is silently ignored by the engine, which does not
# mean "no filter" — it means the caller asked a narrower question and got the
# whole table back. That is the exact silent-failure mode this plan is about, so
# the runner treats an unsupported field as a hard fail rather than letting the
# unfiltered result accidentally satisfy the expectations.
#
# Each phase that lands extends this set. Keep the comments: they are the record
# of what the engine could do when.
SUPPORTED_PLAN_FIELDS = {
    "start", "end", "zone", "entity_label", "predicate_contains",  # M1 five-slot form
    "text",                                          # Phase 2 (FTS5)
    # Phase 4 widened the form. `zone`/`entity_label` stay in the set above
    # because `normalize_plan` still up-converts them — the old shape executes,
    # it is just no longer what the planner emits.
    "intent", "zones", "cameras", "entity_labels", "exclude_entity_labels",
    "entity_type", "exclude_predicates", "min_confidence", "order", "limit",
    # Phase 5. `group_by` joins the set now that it changes the answer rather
    # than being ignored: it chooses which buckets a recurrence is cut into, so
    # "mostly on Tuesdays?" and "how often?" no longer return the same thing.
    "group_by",
    # Phase 6 added no plan field. It changed how `text` is answered — two
    # indexes fused instead of one — which is a retrieval change, not a query
    # form change, and the case list is unchanged as a result.
}

RECALL_KS = (1, 5, 20)


def expected_baseline(case: dict, semantic_on: bool) -> str:
    """Which baseline this case is held to, given what the machine has.

    Phase 6's paraphrase cases can only pass where a local embedding model
    exists, and the plan is explicit that the suite must stay green with the
    dependency uninstalled. So those cases carry two recorded baselines and the
    runner picks by what the corpus actually managed to embed — measured, not
    declared, and not a config flag either.

    The ratchet survives intact in both modes: each is asserted in both
    directions, so a regression with the model installed is as loud as a
    regression without it, and neither can improve silently.
    """
    if semantic_on and case.get("baseline_semantic"):
        return case["baseline_semantic"]
    return case.get("baseline", "fail")


def load_cases() -> list[dict]:
    import yaml
    with open(CASES_PATH) as fh:
        cases = yaml.safe_load(fh)
    ids = [c["id"] for c in cases]
    assert len(ids) == len(set(ids)), "duplicate case ids"
    return cases


# --- plan resolution ---------------------------------------------------------
def _ts(value, anchor: float):
    """'-6.2d' -> anchor - 6.2 days. None -> None. A number is an absolute epoch."""
    if value is None:
        return None
    if isinstance(value, str) and value.endswith("d"):
        return anchor + float(value[:-1]) * search_corpus.DAY
    return float(value)


def resolve_plan(case: dict, corpus: dict) -> dict:
    """Turn the YAML plan into the dict ask.execute() consumes.

    Corpus keys are swapped for the real labels they were created with, so a case
    reads as `priya` rather than hard-coding "Priya" and drifting from the
    fixture. A value that is not a corpus key passes through untouched — that is
    how the true-negative cases name someone who does not exist.
    """
    store: Store = corpus["_store"]
    labels = {}
    for key, eid in corpus["entities"].items():
        row = store.get_entity(eid)
        if row and row["label"]:
            labels[key] = row["label"]

    plan = dict(case.get("plan") or {})
    for field in ("start", "end"):
        if field in plan:
            plan[field] = _ts(plan[field], corpus["anchor"])
    if plan.get("entity_label") is not None:
        plan["entity_label"] = labels.get(plan["entity_label"], plan["entity_label"])
    # Phase 4's plural forms get the same treatment, one element at a time. A
    # value that is not a corpus key still passes through untouched, which is how
    # `exclude_entity_labels: ["delivery courier"]` names someone the way a user
    # would type it rather than by the fixture's internal key.
    for field in ("entity_labels", "exclude_entity_labels"):
        if plan.get(field):
            plan[field] = [labels.get(v, v) for v in plan[field]]
    return plan


def unsupported_fields(plan: dict) -> list[str]:
    return sorted(k for k in plan if k not in SUPPORTED_PLAN_FIELDS)


# --- scoring -----------------------------------------------------------------
def _state_blob(result: dict) -> str:
    parts = []
    for e in result.get("entities", []):
        parts.extend(e.get("states") or [])
        parts.extend(ev.get("rule") or "" for ev in (e.get("rule_events") or []))
    return " || ".join(parts).lower()


def _keyframe_blob(result: dict) -> str:
    parts = []
    for e in result.get("entities", []):
        parts.extend(k or "" for k in (e.get("keyframes") or []))
        parts.extend(ev.get("keyframe") or "" for ev in (e.get("rule_events") or []))
    return " || ".join(parts)


def evaluate(case: dict, result: dict, corpus: dict) -> tuple[bool, list[str]]:
    """Every expectation present must hold. Returns (passed, reasons_it_failed)."""
    ids = corpus["entities"]
    got = [e["entity_id"] for e in result.get("entities", [])]
    why: list[str] = []

    if case.get("expect_empty"):
        if got:
            why.append(f"expected nothing, got {len(got)} entities")

    for key in case.get("expect_entities") or []:
        if ids[key] not in got:
            why.append(f"missing expected entity {key!r}")

    for key in case.get("forbid_entities") or []:
        if ids[key] in got:
            why.append(f"forbidden entity {key!r} present")

    want_text = case.get("expect_text")
    if want_text and want_text.lower() not in _state_blob(result):
        why.append(f"text {want_text!r} absent from results")

    want_kf = case.get("expect_keyframe")
    if want_kf and want_kf not in _keyframe_blob(result):
        why.append(f"keyframe {want_kf!r} not attached")

    want_key = case.get("expect_result_key")
    if want_key and not result.get(want_key):
        why.append(f"result carries no {want_key!r} — raw rows are not an aggregate answer")

    min_count = case.get("expect_min_count")
    if min_count is not None:
        count = result.get("count")
        if count is None:
            count = result.get("total_observations", 0)
        if count < min_count:
            why.append(f"count {count} < expected {min_count}")

    max_ents = case.get("expect_max_entities")
    if max_ents is not None and len(got) > max_ents:
        why.append(f"{len(got)} entities exceeds limit of {max_ents}")

    return (not why), why


def rank_metrics(case: dict, result: dict, corpus: dict) -> dict | None:
    """recall@k and reciprocal rank over the returned entity order.

    Only meaningful for cases that expect specific entities, so true-negative and
    aggregate-only cases return None rather than polluting the average.
    """
    wanted = [corpus["entities"][k] for k in (case.get("expect_entities") or [])]
    if not wanted:
        return None
    got = [e["entity_id"] for e in result.get("entities", [])]
    out = {f"recall@{k}": sum(1 for w in wanted if w in got[:k]) / len(wanted)
           for k in RECALL_KS}
    rr = 0.0
    for i, eid in enumerate(got, start=1):
        if eid in wanted:
            rr = 1.0 / i
            break
    out["rr"] = rr
    return out


def run_case(case: dict, corpus: dict) -> dict:
    """Execute one case. Never raises: a crash is a result, and a scored one."""
    store: Store = corpus["_store"]
    semantic_on = bool(corpus.get("semantic"))
    plan = resolve_plan(case, corpus)
    missing = unsupported_fields(plan)

    started = time.perf_counter()
    if missing:
        # Do not run it. An ignored filter returns the unfiltered table, which
        # could satisfy the expectations for entirely the wrong reason.
        result, passed = {}, False
        why = [f"engine ignores plan field(s) {missing} — would return unfiltered rows"]
    else:
        try:
            result = execute(store, plan)
            passed, why = evaluate(case, result, corpus)
        except Exception as exc:                      # noqa: BLE001 — a crash is a score
            result, passed, why = {}, False, [f"raised {type(exc).__name__}: {exc}"]
    elapsed_ms = (time.perf_counter() - started) * 1000

    expectations = bool(case.get("expect_entities") or case.get("expect_text")
                        or case.get("expect_min_count"))
    return {
        "id": case["id"],
        "question": case["question"],
        "gap": case.get("gap", "none"),
        "fixed_by": case.get("fixed_by", 0),
        "baseline": expected_baseline(case, semantic_on),
        "actual": "pass" if passed else "fail",
        "why": why,
        "unsupported": missing,
        # The headline metric: an answer existed, and we returned nothing at all
        # without saying so. Indistinguishable to the user from "it never happened".
        "silent_failure": bool(expectations and not result.get("entities")),
        "metrics": rank_metrics(case, result, corpus) if not missing else None,
        "elapsed_ms": elapsed_ms,
    }


def build_corpus(scale: int = 0) -> dict:
    store = Store(os.path.join(tempfile.mkdtemp(), "search_eval.db"))
    corpus = search_corpus.build(store, scale=scale)
    corpus["_store"] = store
    return corpus


def run_all(scale: int = 0) -> dict:
    """Score every case. Returns the scorecard the report and the gate both read."""
    corpus = build_corpus(scale=scale)
    try:
        cases = load_cases()
        rows = [run_case(c, corpus) for c in cases]
        scored = [r for r in rows if r["metrics"]]
        n_scored = len(scored) or 1
        agg = {f"recall@{k}": sum(r["metrics"][f"recall@{k}"] for r in scored) / n_scored
               for k in RECALL_KS}
        agg["mrr"] = sum(r["metrics"]["rr"] for r in scored) / n_scored
        latencies = sorted(r["elapsed_ms"] for r in rows if not r["unsupported"])
        return {
            "cases": rows,
            "passing": sum(1 for r in rows if r["actual"] == "pass"),
            "total": len(rows),
            "silent_failures": sum(1 for r in rows if r["silent_failure"]),
            "ranking": agg,
            "latency_ms": {
                "p50": _pct(latencies, 0.50),
                "p95": _pct(latencies, 0.95),
            },
            "retention": search_corpus.retention(corpus["_store"]),
            "scale": scale,
            # How many vectors the corpus managed to build. 0 means this machine
            # has no local embedding model, and every number above was measured
            # against Phase 2's lexical retrieval — which the scorecard has to
            # say out loud, or two runs on two laptops read as a regression.
            "semantic": corpus.get("semantic", 0),
        }
    finally:
        corpus["_store"].close()


def _pct(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    idx = min(len(sorted_values) - 1, max(0, int(round(q * (len(sorted_values) - 1)))))
    return round(sorted_values[idx], 2)


# --- the gate ----------------------------------------------------------------
class SearchQualityBaseline(unittest.TestCase):
    """Asserts reality still matches the recorded baseline, in both directions."""

    @classmethod
    def setUpClass(cls):
        cls.report = run_all()

    def test_every_case_matches_its_recorded_baseline(self):
        drifted = [(r["id"], r["baseline"], r["actual"], r["why"])
                   for r in self.report["cases"] if r["baseline"] != r["actual"]]
        regressions = [d for d in drifted if d[1] == "pass"]
        improvements = [d for d in drifted if d[1] == "fail"]

        msgs = []
        for cid, _, _, why in regressions:
            msgs.append(f"REGRESSION {cid}: was passing, now fails — {'; '.join(why)}")
        for cid, _, _, _ in improvements:
            msgs.append(f"IMPROVED {cid}: now passes. Flip `baseline: fail` -> `pass` "
                        f"in search_cases.yaml and re-run scripts/search_eval.py")
        self.assertEqual([], msgs, "\n" + "\n".join(msgs))

    def test_true_negatives_stay_empty(self):
        """The guard against Phase 7 relaxing its way into inventing an answer."""
        rows = {r["id"]: r for r in self.report["cases"]}
        negatives = [c["id"] for c in load_cases() if c.get("expect_empty")]
        self.assertTrue(negatives, "no true-negative cases — the guard is vacuous")
        for cid in negatives:
            self.assertEqual("pass", rows[cid]["actual"],
                             f"{cid} must return nothing: {rows[cid]['why']}")

    def test_corpus_is_deterministic(self):
        """Two builds must agree, or every metric here is noise."""
        a, b = run_all(), run_all()
        self.assertEqual([r["actual"] for r in a["cases"]],
                         [r["actual"] for r in b["cases"]])
        self.assertEqual(a["retention"]["rate"], b["retention"]["rate"])

    def test_nothing_perceived_is_discarded(self):
        """Phase 1's target, and now a ratchet.

        Before Phase 1 this measured 46.2%: `objects[]` was lost entirely and
        `locations[].contents` almost so. `SceneDescription.rows()` keeps all
        four sections, so the only acceptable number here is 1.0 — a section
        that stops being written shows up as a failure naming itself.
        """
        r = self.report["retention"]
        self.assertGreater(r["perceived"], 0, "nothing perceived — fixture is empty")
        lost = {name: f"{s['retained']}/{s['perceived']}"
                for name, s in r["sections"].items() if s["retained"] < s["perceived"]}
        self.assertEqual({}, lost, f"perceived but not searchable: {lost}")

    def test_full_report_round_trips(self):
        """The audit copy is the whole report, not the flattened rows.

        Re-flattening is how a later phase can improve wording over history
        instead of only going forward, so `raw` has to survive verbatim.
        """
        store = build_corpus()["_store"]
        try:
            descs = store.scene_descriptions()
            self.assertEqual(len(search_corpus.PERCEIVED), len(descs))
            for (_off, _cam, _zone, _kf, report), row in zip(
                    search_corpus.PERCEIVED, descs):
                raw = json.loads(row["raw"])
                self.assertEqual(sorted(report), sorted(raw),
                                 "a report section went missing on the way in")
                self.assertEqual([o["label"] for o in report["objects"]],
                                 [o["label"] for o in raw["objects"]])
                self.assertTrue(row["text"], "flattened text is empty")
            # Every VLM row points back at the report it was flattened out of.
            vlm = [o for o in store.observations() if o["origin"] == "vlm"]
            self.assertTrue(vlm)
            orphans = [o["observation_id"] for o in vlm if not o["description_id"]]
            self.assertEqual([], orphans, "VLM rows with no provenance")
        finally:
            store.close()


def main() -> int:
    """Script entry point, matching the acceptance-gate convention."""
    unittest.main(module=__name__, argv=["search-quality"], exit=False, verbosity=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
