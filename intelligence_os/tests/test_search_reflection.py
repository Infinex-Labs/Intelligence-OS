"""The relaxation ladder (search plan Phase 7).

The quality gate scores whether the ladder finds the right answers. This file
is about the other half, which matters more: whether it can be trusted not to
find the WRONG ones.

A widening search is one of the easier things to build and one of the easier
things to build badly. The bad version is not subtly bad — it relaxes until
something comes back, and then reports that something as the answer, which is
strictly worse than the empty result it replaced because it looks like
knowledge. So most of what is here is about the refusals:

  * a question that was answered is never touched at all
  * a filter naming something memory has never heard of is never dropped
  * an exclusion is never overruled
  * a widening always says it widened

The three properties that make those hold are structural, not statistical, so
they are asserted directly rather than measured. No model is needed for any of
this: the ladder is a policy over SQL.

Run: python -m unittest intelligence_os.tests.test_search_reflection
"""
from __future__ import annotations

import os
import tempfile
import unittest

from intelligence_os.tests import _stubs  # noqa: F401  (headless dep stubs)
from intelligence_os import ask
from intelligence_os.config import CONFIG
from intelligence_os.store import Store

HOUR = 3600.0
DAY = 86400.0
T0 = 1780408800.0          # the corpus anchor; a fixed UTC Tuesday


class _LadderBase(unittest.TestCase):
    """A small memory with one obvious near-miss in it.

    Deliberately not the eval corpus. That one is shared, and a fixture whose
    shape several suites depend on is a fixture nobody can add an awkward row
    to. This one exists to be awkward.
    """

    def setUp(self):
        # The ladder is a config flag rather than a dependency, and these tests
        # are what proves it works when on — so they turn it on regardless of
        # how the process was launched, and put it back afterwards. The OFF
        # behaviour gets its own class below.
        self._was = CONFIG.reflect.enabled
        CONFIG.reflect.enabled = True
        self.store = Store(os.path.join(tempfile.mkdtemp(), "reflect.db"))
        self.addCleanup(self.store.close)
        self.addCleanup(lambda: setattr(CONFIG.reflect, "enabled", self._was))

        self.bay = self.store.upsert_location(
            "loading_bay", {"polygon": [[0, 0], [1, 0], [1, 1]]}, camera_id="cam_dock")
        self.gate = self.store.upsert_location(
            "side_gate", {"polygon": [[2, 0], [3, 0], [3, 1]]}, camera_id="cam_yard")

        self.dave = self.store.create_entity("person", label="Dave")
        self.priya = self.store.create_entity("person", label="Priya")

        # Dave: at the BAY, at T0. The near-miss target.
        self.store.add_observation(self.dave, "present", location_id=self.bay,
                                   timestamp=T0, confidence=0.9, origin="detector",
                                   camera_id="cam_dock", source_ref="/kf/dave.jpg")
        # Priya: at the GATE, two days later.
        self.store.add_observation(self.priya, "present", location_id=self.gate,
                                   timestamp=T0 + 2 * DAY, confidence=0.9,
                                   origin="detector", camera_id="cam_yard")

    def run_plan(self, **plan) -> dict:
        return ask.execute(self.store, plan)

    def labels(self, result: dict) -> list[str]:
        return sorted(e["label"] for e in result["entities"])

    def step_names(self, result: dict) -> list[str]:
        rx = result.get("relaxation") or {}
        return [s["step"] for s in rx.get("attempted", [])]


class TestAnAnsweredQuestionIsUntouched(_LadderBase):
    """The containment property, and the reason this phase is low-risk.

    If the ladder can only run on an empty result, then nothing that worked
    before can change — no regression is possible in the space of questions
    that already had answers, which is most of them.
    """

    def test_a_question_with_an_answer_carries_no_relaxation_at_all(self):
        got = self.run_plan(start=T0 - HOUR, end=T0 + HOUR, zone="loading_bay")
        self.assertEqual(["Dave"], self.labels(got))
        self.assertIsNone(got.get("relaxation"),
                          "the ladder ran on a question that was already answered")

    def test_the_plan_returned_is_the_plan_asked_for(self):
        got = self.run_plan(start=T0 - HOUR, end=T0 + HOUR, zone="loading_bay")
        self.assertEqual(["loading_bay"], got["plan"]["zones"])

    def test_a_partial_answer_is_still_an_answer(self):
        """One entity out of two is not a dead end. Widening from here would be
        the ladder deciding the answer was too small, which is not its job."""
        got = self.run_plan(start=T0 - HOUR, end=T0 + HOUR)
        self.assertEqual(["Dave"], self.labels(got))
        self.assertIsNone(got.get("relaxation"))


class TestTheRungs(_LadderBase):

    def test_the_zone_is_dropped_when_nothing_is_there(self):
        got = self.run_plan(start=T0 - HOUR, end=T0 + HOUR, zone="side_gate")
        self.assertEqual(["Dave"], self.labels(got))
        self.assertEqual("drop_zone", got["relaxation"]["answered_by"])

    def test_the_window_is_widened_when_the_question_just_missed(self):
        """The case the whole phase is named for: right place, wrong hour."""
        got = self.run_plan(start=T0 + 2 * HOUR, end=T0 + 4 * HOUR, zone="loading_bay")
        self.assertEqual(["Dave"], self.labels(got))
        self.assertEqual("widen_window", got["relaxation"]["answered_by"])

    def test_the_subject_is_dropped_only_after_time_and_place(self):
        """Order is load-bearing, because stop-at-first-hit means the order IS
        the answer. Dropping the name changes who the answer is about, so it
        has to be the last thing tried, not the first that happens to work."""
        got = self.run_plan(start=T0 - HOUR, end=T0 + HOUR, zone="loading_bay",
                            entity_label="Priya")
        self.assertEqual(["widen_window", "drop_zone", "drop_entity_label"],
                         self.step_names(got))
        self.assertEqual("drop_entity_label", got["relaxation"]["answered_by"])
        self.assertEqual(["Dave"], self.labels(got))

    def test_each_rung_derives_from_the_original_plan_not_the_previous_rung(self):
        """A ladder, not a slide. If rung 2 inherited rung 1's dropped zone, the
        window would be widened across every zone at once and the answer would
        come back from somewhere nobody asked about."""
        got = self.run_plan(start=T0 + 2 * HOUR, end=T0 + 3 * HOUR, zone="side_gate")
        # Rung 1 widened *within the gate* and found nothing (Dave is at the
        # bay). Rung 2 dropped the gate but kept the ORIGINAL window, which
        # Dave is outside. If the rungs compounded, rung 2 would be "anywhere,
        # wider" and would return Dave — an answer from the wrong place AND the
        # wrong hour, assembled out of two separate concessions.
        self.assertEqual([], self.labels(got))
        self.assertIsNone(got["relaxation"]["answered_by"])
        self.assertEqual(["widen_window", "drop_zone"], self.step_names(got))

    def test_it_stops_at_the_first_rung_that_works(self):
        got = self.run_plan(start=T0 - HOUR, end=T0 + HOUR, zone="side_gate",
                            entity_label="Dave")
        # The window widens first and still finds nothing (Dave is never at the
        # gate); the zone then goes and finds him. Nothing is tried after that.
        self.assertEqual(["widen_window", "drop_zone"], self.step_names(got))

    def test_the_gentlest_rung_wins_even_when_a_harsher_one_would_also_hit(self):
        """The ordering rule, asserted as a preference rather than a sequence.

        Both rungs can answer this: widening the window finds Dave at the bay
        that was asked about, and dropping the zone finds Priya somewhere else.
        The right answer is the one that gave up less.
        """
        got = self.run_plan(start=T0 + 2 * HOUR, end=T0 + 4 * HOUR, zone="loading_bay")
        self.assertEqual("widen_window", got["relaxation"]["answered_by"])
        self.assertEqual(["Dave"], self.labels(got))


class TestItRefusesToDropWhatMemoryNeverKnew(_LadderBase):
    """The property that makes the true negatives hold by construction.

    Relaxing a filter that named a real thing widens the search. Relaxing one
    that named nothing abandons it — and answers a question about somebody
    else, confidently, with a photograph attached.
    """

    def test_a_name_nobody_has_is_never_dropped(self):
        got = self.run_plan(entity_label="Mallory")
        self.assertEqual([], self.labels(got))
        self.assertNotIn("drop_entity_label", self.step_names(got))

    def test_a_place_that_is_not_a_place_is_never_dropped(self):
        got = self.run_plan(zone="car_park")
        self.assertEqual([], self.labels(got))
        self.assertNotIn("drop_zone", self.step_names(got))

    def test_a_name_that_does_exist_is_dropped(self):
        """The other direction, so the rule above cannot be satisfied by a rung
        that never fires."""
        got = self.run_plan(start=T0 - HOUR, end=T0 + HOUR, zone="loading_bay",
                            entity_label="Priya")
        self.assertIn("drop_entity_label", self.step_names(got))
        self.assertEqual(["Dave"], self.labels(got))


class TestTheWindowWideningIsBounded(_LadderBase):
    """Why there is an absolute cap and not just a multiplier.

    x4 of a six-hour question is a day, which is a fuzz around what was asked.
    x4 of a six-week question is half a year, which is a different question. The
    cap is what stops "was anyone there next week?" reaching back through
    everything ever recorded and calling it an answer.
    """

    def test_a_huge_empty_window_does_not_swallow_the_whole_memory(self):
        got = self.run_plan(start=T0 + 7 * DAY, end=T0 + 14 * DAY)
        self.assertIn("widen_window", self.step_names(got))
        self.assertEqual([], self.labels(got),
                         "a week-wide future window widened its way into the past")

    def test_the_pad_never_exceeds_the_cap(self):
        plan = ask.normalize_plan({"start": T0, "end": T0 + 30 * DAY})
        widened = ask._widen(plan)
        self.assertAlmostEqual(CONFIG.reflect.widen_cap_s, plan["start"] - widened["start"])
        self.assertAlmostEqual(CONFIG.reflect.widen_cap_s, widened["end"] - plan["end"])

    def test_a_short_window_widens_proportionally(self):
        plan = ask.normalize_plan({"start": T0, "end": T0 + 600})
        widened = ask._widen(plan)
        self.assertAlmostEqual(600 * CONFIG.reflect.widen_factor,
                               plan["start"] - widened["start"])

    def test_an_unbounded_window_has_nothing_to_widen(self):
        self.assertIsNone(ask._widen(ask.normalize_plan({})))

    def test_a_half_open_window_widens_the_bounded_side(self):
        plan = ask.normalize_plan({"start": T0})
        widened = ask._widen(plan)
        self.assertAlmostEqual(CONFIG.reflect.widen_cap_s, plan["start"] - widened["start"])
        self.assertIsNone(widened["end"])


class TestExclusionsSurviveEveryRung(_LadderBase):
    """Widening may add candidates. It may never overrule what was ruled out.

    "Anyone except the courier" is a constraint on the ANSWER, and a ladder that
    treats it as one more thing to loosen would hand back precisely the person
    the question was written to avoid.
    """

    def test_an_excluded_name_never_comes_back(self):
        got = self.run_plan(start=T0 - HOUR, end=T0 + HOUR, zone="side_gate",
                            exclude_entity_labels=["Dave"])
        self.assertNotIn("Dave", self.labels(got))
        self.assertTrue(self.step_names(got), "the ladder did not run at all")

    def test_an_excluded_predicate_never_comes_back(self):
        got = self.run_plan(start=T0 - HOUR, end=T0 + HOUR, zone="side_gate",
                            exclude_predicates=["present"])
        self.assertEqual([], self.labels(got))

    def test_the_entity_type_survives(self):
        """Not in the plan's list of relaxable things, and it should not be:
        "which vehicle" answered with a person is not a wider answer."""
        got = self.run_plan(start=T0 - HOUR, end=T0 + HOUR, zone="side_gate",
                            entity_type="object")
        self.assertEqual([], self.labels(got))

    def test_the_confidence_floor_survives(self):
        got = self.run_plan(start=T0 - HOUR, end=T0 + HOUR, zone="side_gate",
                            min_confidence=0.99)
        self.assertEqual([], self.labels(got))


class TestItAlwaysSaysWhatItDid(_LadderBase):
    """Disclosure is not decoration. A loosened answer that does not say it was
    loosened is a correct answer to a question nobody asked, presented as the
    answer to the one they did ask — worse than the empty result it replaced.
    """

    def test_the_trace_names_the_rung_and_what_it_cost(self):
        got = self.run_plan(start=T0 - HOUR, end=T0 + HOUR, zone="side_gate")
        rx = got["relaxation"]
        self.assertEqual("drop_zone", rx["answered_by"])
        self.assertIn("side_gate", rx["loosened"])
        # Every rung's cost is recorded, not just the one that worked — the
        # rungs that found nothing are what distinguish "we looked" from "it is
        # not there", and they are the reason a reported empty means something.
        self.assertEqual([("widen_window", 0), ("drop_zone", 1)],
                         [(s["step"], s["found"]) for s in rx["attempted"]])

    def test_the_trace_keeps_the_question_and_the_query_apart(self):
        """`query` is what was asked, `plan` is what ran. On a relaxed answer
        those differ, and a trace that showed only one of them would be hiding
        exactly the thing worth showing."""
        got = self.run_plan(start=T0 - HOUR, end=T0 + HOUR, zone="side_gate")
        self.assertEqual("side_gate", got["query"]["zone"])
        self.assertEqual([], got["plan"]["zones"])

    def test_the_narration_facts_carry_the_disclosure(self):
        got = self.run_plan(start=T0 - HOUR, end=T0 + HOUR, zone="side_gate")
        facts = ask._facts(got)
        self.assertIn("relaxed_query", facts)
        self.assertIn("side_gate", facts["relaxed_query"]["loosened_instead"])

    def test_the_narrator_is_told_it_must_disclose(self):
        """The obligation lives in the system prompt, not in the data — facts
        are facts, and an instruction smuggled into a JSON field is neither."""
        self.assertIn("relaxed_query", ask.NARRATE_SYSTEM)
        self.assertIn("FIRST sentence", ask.NARRATE_SYSTEM)

    def test_a_failed_ladder_still_reports_what_it_tried(self):
        """The honest empty, and the actual headline of this phase: 'nothing
        found' and 'nothing found, and here is everywhere else we looked' are
        different answers to the person reading them."""
        got = self.run_plan(start=T0 + 7 * DAY, end=T0 + 14 * DAY, zone="loading_bay")
        rx = got["relaxation"]
        self.assertIsNone(rx["answered_by"])
        self.assertTrue(rx["attempted"])
        facts = ask._facts(got)
        self.assertTrue(facts["relaxed_query"]["also_tried_and_still_nothing"])

    def test_the_cli_prints_the_disclosure(self):
        got = self.run_plan(start=T0 - HOUR, end=T0 + HOUR, zone="side_gate")
        text = ask.render_text(got)
        self.assertIn("relaxed:", text)
        self.assertIn("side_gate", text)


class TestDeadEndMeansAllThreeAnswerShapes(_LadderBase):
    """`who_with` and `relations` answer out of different tables, and their
    answer can be non-empty while the entity list is not. Judging "did we find
    anything?" on the entity list alone would widen a question that had already
    been answered — the exact regression the containment property forbids.
    """

    def setUp(self):
        super().setUp()
        # Priya and a visitor share a frame at the gate, two days after Dave.
        self.visitor = self.store.create_entity("person", label="Visitor")
        self.store.add_observation(self.visitor, "present", location_id=self.gate,
                                   timestamp=T0 + 2 * DAY, confidence=0.9,
                                   origin="detector", camera_id="cam_yard")
        self.store.add_snapshot(self.gate, [self.priya, self.visitor],
                                timestamp=T0 + 2 * DAY)

    def test_a_co_presence_answer_is_not_a_dead_end(self):
        got = self.run_plan(intent="who_with", entity_label="Priya", zone="side_gate")
        self.assertTrue(got["co_presence"])
        self.assertIsNone(got.get("relaxation"),
                          "widened a question the snapshots had already answered")

    def test_a_co_presence_question_with_nothing_in_it_does_widen(self):
        got = self.run_plan(intent="who_with", entity_label="Priya",
                            zone="loading_bay")
        self.assertEqual("drop_zone", got["relaxation"]["answered_by"])
        self.assertTrue(got["co_presence"])


class TestTheLadderIsCapped(_LadderBase):

    def test_no_more_than_max_steps_are_ever_tried(self):
        got = self.run_plan(start=T0 + 7 * DAY, end=T0 + 14 * DAY, zone="loading_bay",
                            entity_label="Priya", text="something not recorded")
        self.assertLessEqual(len(self.step_names(got)), CONFIG.reflect.max_steps)

    def test_a_cap_of_zero_disables_the_rungs_without_disabling_the_report(self):
        CONFIG.reflect.max_steps = 0
        self.addCleanup(lambda: setattr(CONFIG.reflect, "max_steps", 3))
        got = self.run_plan(start=T0 - HOUR, end=T0 + HOUR, zone="side_gate")
        self.assertEqual([], self.labels(got))
        self.assertEqual([], got["relaxation"]["attempted"])


class TestItCanBeTurnedOff(_LadderBase):
    """With the switch off, a dead end is a dead end again — Phase 6 behaviour,
    exactly, and not approximately."""

    def setUp(self):
        super().setUp()
        CONFIG.reflect.enabled = False

    def test_an_empty_result_stays_empty_and_says_nothing(self):
        got = self.run_plan(start=T0 - HOUR, end=T0 + HOUR, zone="side_gate")
        self.assertEqual([], self.labels(got))
        self.assertIsNone(got.get("relaxation"))

    def test_answers_are_unaffected(self):
        got = self.run_plan(start=T0 - HOUR, end=T0 + HOUR, zone="loading_bay")
        self.assertEqual(["Dave"], self.labels(got))

    def test_the_facts_carry_no_disclosure_because_there_is_nothing_to_disclose(self):
        got = self.run_plan(start=T0 - HOUR, end=T0 + HOUR, zone="side_gate")
        self.assertNotIn("relaxed_query", ask._facts(got))


if __name__ == "__main__":
    unittest.main()
