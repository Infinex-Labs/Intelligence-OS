"""Habits, co-presence and distilled edges, queried (search plan Phase 5).

Phases 1-4 made the *observation log* searchable. This one is about the two
memory kinds that were being written every night and never read: the distilled
edges in `relations`, and the settled inventories in `scene_snapshots`. "How
often does that van come by?" and "who was she with?" were unanswerable from
data that already existed, and — worse — they were answered anyway, with a list
of rows, because raw sightings are what the engine had.

Four things are under test, and only the first is "the feature works":

  1. each intent reaches its own table and computes the right aggregate
  2. an aggregate answer is about somebody *else*. `who_with` and `relations`
     take the plan's labels as the ANCHOR and answer with what the anchor turned
     out to be connected to, so the label filter must not also be applied to the
     answer — and the anchor set must stay authoritative even when the SQL
     pushdown behind it is dropped, which is where this could silently widen
     into "everyone who was ever with anyone"
  3. the hard filters still reach the aggregates. A window or a zone that
     narrows the rows has to narrow the pattern, or "how often at the bay" is
     really "how often anywhere" wearing a filter
  4. the buckets are cut on the deployment's clock, in one place, and every
     reader of them cuts them the same way

The mining half is here too — weekday buckets (G8) and the collapse of the
per-observation write loop (G9) — because a habit that is mined wrong cannot be
queried right.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from intelligence_os.tests import _stubs   # noqa: F401  (stubs the deps we lack)

import intelligence_os.ask as ask                          # noqa: E402
from intelligence_os.ask import execute                     # noqa: E402
from intelligence_os.config import CONFIG                   # noqa: E402
from intelligence_os.distill import (                       # noqa: E402
    Distiller, bucket_day, bucket_hour, bucket_weekday)
from intelligence_os.store import Store, scene_subject      # noqa: E402

DAY = 86400.0
WEEK = 7 * DAY
# A Tuesday, 14:00 UTC. Which local hour that lands on is the machine's
# business — nothing below asserts on the number, only on agreement.
T0 = 1_780_408_800.0


class _Base(unittest.TestCase):
    """A gate and a bay, a regular visitor, a van with a Tuesday habit.

    Small enough that every row asserted on below is written here by name. The
    distiller is run explicitly by the tests that need mined edges, so a test
    that fails says whether the miner or the reader broke.
    """

    def setUp(self):
        # Off for the same reason as in test_search_query_form: several tests
        # here assert that a filter narrows an aggregate to nothing, and with
        # the ladder on those would be asserting the much weaker "...and the
        # ladder also came up empty". The ladder's own behaviour, including
        # what it does to these aggregates, lives in test_search_reflection.py.
        self._relax = CONFIG.reflect.enabled
        CONFIG.reflect.enabled = False
        self.addCleanup(lambda: setattr(CONFIG.reflect, "enabled", self._relax))

        self._dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._dir.cleanup)
        self.store = s = Store(db_path=Path(self._dir.name) / "t.db")
        self.addCleanup(s.close)

        self.gate = s.upsert_location("gate", {"polygon": [[0, 0]]}, camera_id="cam_a")
        self.bay = s.upsert_location("bay", {"polygon": [[5, 5]]}, camera_id="cam_b")

        self.priya = s.create_entity("person", label="Priya")
        self.visitor = s.create_entity("person", label="Visitor")
        self.courier = s.create_entity("person", label="Delivery Courier")
        self.van = s.create_entity("object", label="White Van")
        self.forklift = s.create_entity("object", label="Forklift")

        # The van, six Tuesdays at the bay. One visit per day, so sightings and
        # occasions differ only where a test makes them differ.
        self.tuesdays = [T0 - i * WEEK for i in range(6)]
        for t in self.tuesdays:
            s.add_observation(self.van, "present", location_id=self.bay,
                              timestamp=t, confidence=0.85, origin="detector",
                              camera_id="cam_b", source_ref="/kf/van.jpg")
        # ...and twice at the gate, so a zone filter has something to remove.
        for t in (T0 - 3 * DAY, T0 - 10 * DAY):
            s.add_observation(self.van, "present", location_id=self.gate,
                              timestamp=t, confidence=0.85, origin="detector",
                              camera_id="cam_a")

        # Priya lets a visitor in at the gate; the courier is there alone later.
        s.add_observation(self.priya, "present", location_id=self.gate,
                          timestamp=T0 - 2 * DAY, origin="detector", camera_id="cam_a")
        s.add_observation(self.visitor, "present", location_id=self.gate,
                          timestamp=T0 - 2 * DAY + 10, origin="detector",
                          camera_id="cam_a", source_ref="/kf/visitor.jpg")
        s.add_observation(self.courier, "present", location_id=self.gate,
                          timestamp=T0 - DAY, origin="detector", camera_id="cam_a")

        self.snap_pair = s.add_snapshot(self.gate, [self.priya, self.visitor],
                                        timestamp=T0 - 2 * DAY + 5)
        self.snap_alone = s.add_snapshot(self.gate, [self.courier],
                                         timestamp=T0 - DAY)

        # Repeated proximity, which is what a 'uses' edge is mined from.
        self.near_ids = [
            s.add_observation(self.priya, "near", object_entity_id=self.forklift,
                              location_id=self.bay, timestamp=T0 - 2 * DAY + 60 * i,
                              origin="detector", camera_id="cam_b")
            for i in range(4)]

    def mine(self):
        d = Distiller(self.store)
        d.mine_habits()
        d.mine_relations()

    def labels(self, **plan) -> list[str]:
        return sorted(e["label"] for e in execute(self.store, plan)["entities"])


# --- 1. mining: the buckets a habit is cut into ------------------------------
class TestHabitsAreMinedAtTwoGranularities(_Base):
    """G8. An hour bucket alone cannot express "most Tuesdays"."""

    def test_both_an_hour_habit_and_a_weekday_one_are_mined(self):
        self.mine()
        preds = {h["predicate"] for h in self.store.habits()}
        hour = f"{bucket_hour(T0):02d}h"
        self.assertIn(f"present_around_{hour}", preds)
        self.assertIn(f"present_{bucket_weekday(T0)}_around_{hour}", preds)

    def test_the_coarse_habit_is_not_a_summary_of_the_fine_ones(self):
        """A daily pattern is a strong hour habit and no weekday habit at all.

        This is why both are mined rather than one derived from the other:
        summing five weekday buckets of one day each would report five weak
        weekly patterns where the truth is one strong daily one.
        """
        s = self.store
        person = s.create_entity("person", label="Daily Raj")
        for day in range(5):                    # five consecutive days, same hour
            s.add_observation(person, "present", location_id=self.bay,
                              timestamp=T0 + day * DAY, origin="detector")
        self.mine()
        mine_preds = {h["predicate"] for h in s.habits(entity_ids=[person])}
        hour = f"{bucket_hour(T0):02d}h"
        self.assertIn(f"present_around_{hour}", mine_preds,
                      "five days at one hour is an hour habit")
        weekday = [p for p in mine_preds if p != f"present_around_{hour}"]
        self.assertEqual([], weekday,
                         "each weekday was seen once — no weekday recurs yet")

    def test_a_habit_needs_more_than_one_day(self):
        s = self.store
        once = s.create_entity("person", label="Passing Through")
        for i in range(20):                     # twenty rows, one day
            s.add_observation(once, "present", location_id=self.bay,
                              timestamp=T0 + i * 30, origin="detector")
        self.mine()
        self.assertEqual([], s.habits(entity_ids=[once]),
                         "twenty sightings in one afternoon is not a routine")

    def test_the_buckets_are_the_same_ones_every_reader_cuts(self):
        """One definition, three callers — the bug this replaces was two.

        Habits were bucketed with `time.gmtime` and rendered with
        `time.localtime`, so a deployment east of Greenwich mined "present
        around 09h" from footage everyone there remembers as half past two. Any
        reader computing its own hour is how that comes back.
        """
        self.mine()
        hour = bucket_hour(T0)
        habit = next(h for h in self.store.habits(entity_ids=[self.van])
                     if h["predicate"].startswith("present_around_"))
        self.assertEqual(f"present_around_{hour:02d}h", habit["predicate"])

        r = execute(self.store, {"intent": "how_often", "entity_labels": ["van"],
                                 "group_by": "hour"})
        self.assertEqual(f"{hour:02d}h", r["recurrence"][0]["top_hour"]["bucket"],
                         "ask must bucket the way distill mined")

    def test_a_day_is_a_local_calendar_day_not_an_epoch_division(self):
        """`int(ts // 86400)` counts UTC days, and splits local evenings in two."""
        self.assertEqual(bucket_day(T0), bucket_day(T0 + 60))
        self.assertNotEqual(bucket_day(T0), bucket_day(T0 + DAY))


class TestMiningIsBounded(_Base):
    """G9. A nightly job must not get slower every night, forever."""

    def test_the_window_excludes_what_falls_before_it(self):
        self.mine()
        self.assertTrue(self.store.habits(entity_ids=[self.van]))

        fresh = Store(db_path=Path(self._dir.name) / "w.db")
        self.addCleanup(fresh.close)
        van = fresh.create_entity("object", label="White Van")
        bay = fresh.upsert_location("bay", {"polygon": [[0, 0]]})
        for t in self.tuesdays:
            fresh.add_observation(van, "present", location_id=bay, timestamp=t,
                                  origin="detector")
        # Three days back from the newest row: only the last Tuesday survives,
        # and one day is not a habit.
        Distiller(fresh).mine_habits(since=T0 - 3 * DAY)
        self.assertEqual([], fresh.habits(),
                         "a bounded scan must not mine what it did not read")

    def test_the_window_is_measured_from_the_memory_not_the_wall_clock(self):
        """An outage is not a routine ending.

        The corpus is anchored years from `now()`, so a window measured from the
        wall clock would contain nothing and quietly un-reinforce every habit
        the deployment has.
        """
        d = Distiller(self.store)
        since = d.mine_window()
        self.assertIsNotNone(since)
        self.assertAlmostEqual(
            self.store.newest_observation_at() - CONFIG.distill.mine_window_days * DAY,
            since, places=3)
        d.mine_habits(since=since)
        self.assertTrue(self.store.habits(entity_ids=[self.van]))

    def test_an_empty_memory_has_no_window_rather_than_a_wrong_one(self):
        empty = Store(db_path=Path(self._dir.name) / "e.db")
        self.addCleanup(empty.close)
        self.assertIsNone(empty.newest_observation_at())
        self.assertIsNone(Distiller(empty).mine_window())

    def test_one_weighted_write_equals_n_repeated_ones(self):
        """The collapse is arithmetic, not an approximation.

        The increment is additive and the cap is a `min`, so n applications land
        exactly where one scaled by n does. If that ever stops being true, every
        confirmed edge in every existing memory changes status.
        """
        s = self.store
        a, b = s.create_entity("object"), s.create_entity("object")
        for _ in range(7):
            s.reinforce_relation("relation", a, "uses", object_entity_id=b,
                                 obs_confidence=0.8)
        loop_weight = s.relations(subject_entity_id=a)[0]["weight"]

        c, d = s.create_entity("object"), s.create_entity("object")
        s.reinforce_relation("relation", c, "uses", object_entity_id=d,
                             obs_confidence=0.8, times=7)
        self.assertAlmostEqual(loop_weight, s.relations(subject_entity_id=c)[0]["weight"],
                               places=9)

    def test_the_cap_still_holds_when_the_evidence_is_enormous(self):
        s = self.store
        a, b = s.create_entity("object"), s.create_entity("object")
        s.reinforce_relation("relation", a, "uses", object_entity_id=b, times=100_000)
        self.assertEqual(CONFIG.distill.weight_cap,
                         s.relations(subject_entity_id=a)[0]["weight"])


# --- 2. the store's three reads ----------------------------------------------
class TestTheStoreReads(_Base):

    def test_relation_edges_follows_the_none_and_empty_convention(self):
        self.mine()
        s = self.store
        self.assertTrue(s.relation_edges(), "None everywhere is no filter")
        self.assertEqual([], s.relation_edges(kinds=[]),
                         "an empty list is a filter nothing satisfies")
        self.assertEqual([], s.relation_edges(subject_entity_ids=[]))

    def test_a_suppressed_belief_does_not_come_back_as_an_answer(self):
        """`suppress_relation` is an operator saying a belief is wrong."""
        self.mine()
        edge = next(e for e in self.store.relation_edges(kinds=["relation"])
                    if e["predicate"] == "uses")
        self.store.suppress_relation(edge["relation_id"])
        self.assertNotIn(edge["relation_id"],
                         [e["relation_id"] for e in self.store.relation_edges()])
        self.assertIn(edge["relation_id"],
                      [e["relation_id"] for e in
                       self.store.relation_edges(include_suppressed=True)],
                      "suppression is reversible, so the row must survive")

    def test_habits_is_relation_edges_asked_for_one_kind(self):
        self.mine()
        self.assertTrue(self.store.habits())
        self.assertEqual({"habit"}, {h["kind"] for h in self.store.habits()})
        self.assertEqual([], self.store.habits(entity_ids=[self.courier]))

    def test_co_presence_pairs_who_shared_a_frame(self):
        pairs = self.store.co_presence()
        self.assertEqual(
            {(self.priya, self.visitor), (self.visitor, self.priya)},
            {(p["entity_id"], p["with_entity_id"]) for p in pairs})

    def test_alone_in_the_frame_is_not_company(self):
        """The courier has a snapshot of his own; a snapshot is not a companion."""
        pairs = self.store.co_presence(entity_ids=[self.courier])
        self.assertEqual([], pairs)

    def test_co_presence_honours_the_window_and_the_place(self):
        self.assertTrue(self.store.co_presence(since=T0 - 3 * DAY, until=T0))
        self.assertEqual([], self.store.co_presence(since=T0 - DAY))
        self.assertEqual([], self.store.co_presence(location_ids=[self.bay]))
        self.assertEqual([], self.store.co_presence(location_ids=[]),
                         "an empty zone list is a filter, not a pass")


# --- 3. how_often ------------------------------------------------------------
class TestHowOften(_Base):

    def ask_van(self, **extra):
        plan = {"intent": "how_often", "entity_labels": ["van"], **extra}
        return execute(self.store, plan)

    def test_the_answer_carries_a_recurrence_not_just_rows(self):
        r = self.ask_van()
        self.assertTrue(r["recurrence"], "raw rows are not a recurrence answer")
        rec = r["recurrence"][0]
        self.assertEqual("White Van", rec["label"])
        self.assertEqual(8, rec["n_sightings"])

    def test_occasions_count_days_not_rows(self):
        """Ten frames of one visit is one visit, and "how often" asks about visits."""
        for i in range(1, 10):
            self.store.add_observation(self.van, "present", location_id=self.bay,
                                       timestamp=T0 + i * 30, origin="detector")
        rec = self.ask_van()["recurrence"][0]
        self.assertEqual(17, rec["n_sightings"])
        self.assertEqual(8, rec["n_occasions"], "the extra frames are one more visit")

    def test_the_weekday_bucket_is_what_makes_most_tuesdays_a_question(self):
        rec = self.ask_van(group_by="weekday")["recurrence"][0]
        top = rec["top"]
        self.assertEqual(bucket_weekday(T0), top["bucket"])
        self.assertEqual(6, top["n"], "six of the eight sightings are Tuesdays")

    def test_group_by_changes_the_answer_rather_than_being_ignored(self):
        """A plan field the engine accepts and does not act on is the silent
        widening this whole plan is about, so this is the load-bearing test for
        `group_by` being in SUPPORTED_PLAN_FIELDS at all."""
        by_weekday = self.ask_van(group_by="weekday")["recurrence"][0]
        by_hour = self.ask_van(group_by="hour")["recurrence"][0]
        by_day = self.ask_van(group_by="day")["recurrence"][0]
        self.assertEqual("weekday", by_weekday["group_by"])
        self.assertEqual(f"{bucket_hour(T0):02d}h", by_hour["top"]["bucket"])
        self.assertEqual(8, len(by_day["groups"]), "eight distinct days")
        self.assertNotEqual(by_weekday["groups"], by_hour["groups"])

    def test_an_unknown_grouping_falls_back_visibly_in_the_trace(self):
        r = self.ask_van(group_by="phase_of_the_moon")
        self.assertEqual("weekday", r["plan"]["group_by"],
                         "the trace must show which buckets actually ran")

    def test_the_mined_habit_is_attached_with_its_provenance(self):
        self.mine()
        habits = self.ask_van()["recurrence"][0]["habits"]
        self.assertTrue(habits)
        h = habits[0]
        self.assertIn(h["status"], ("candidate", "confirmed"))
        self.assertGreater(h["n_supporting"], 0)
        self.assertGreater(h["weight"], 0.0)

    def test_the_counts_answer_before_the_nightly_pass_has_run(self):
        """Habits lag by a distillation pass; the counts do not.

        Answering "how often" only out of `relations` would return nothing at
        all for anything observed since the last nightly run — which is exactly
        the window someone asking is most likely to mean.
        """
        rec = self.ask_van()["recurrence"][0]          # no mine() above
        self.assertEqual([], rec["habits"])
        self.assertEqual(8, rec["n_sightings"], "the counts stand on their own")

    def test_a_zone_narrows_the_pattern_and_not_just_the_rows(self):
        rec = self.ask_van(zones=["bay"])["recurrence"][0]
        self.assertEqual(6, rec["n_sightings"], "the two gate visits are elsewhere")
        self.assertEqual(6, rec["top"]["n"])

    def test_a_window_narrows_the_pattern(self):
        rec = self.ask_van(start=T0 - 2 * WEEK, end=T0)["recurrence"][0]
        self.assertEqual(5, rec["n_sightings"],   # 3 Tuesdays + 2 gate visits
                         "the three older Tuesdays are outside the window")
        self.assertLessEqual(rec["span_days"], 14.01)

    def test_a_question_about_nobody_gets_no_pattern_rather_than_a_made_up_one(self):
        r = execute(self.store, {"intent": "how_often",
                                 "entity_labels": ["nobody by that name"]})
        self.assertEqual([], r["entities"])
        self.assertEqual([], r["recurrence"])

    def test_recurrence_is_absent_when_it_was_not_asked_for(self):
        r = execute(self.store, {"entity_labels": ["van"]})
        self.assertNotIn("recurrence", r,
                         "a question that is not about recurrence carries none")


# --- 4. who_with -------------------------------------------------------------
class TestWhoWith(_Base):

    def test_the_answer_is_the_companion_not_the_person_asked_about(self):
        """The plan's label is the anchor. Applying it to the answer as well
        would ask which of the people with Priya are called Priya."""
        r = execute(self.store, {"intent": "who_with", "entity_labels": ["priya"]})
        self.assertEqual(["Visitor"], [e["label"] for e in r["entities"]])
        self.assertEqual(["Visitor"], [c["label"] for c in r["co_presence"]])
        self.assertEqual(["Priya"], [w["label"] for w in r["co_presence"][0]["with"]])

    def test_with_no_anchor_everyone_who_shared_a_frame_is_an_answer(self):
        r = execute(self.store, {"intent": "who_with", "zones": ["gate"]})
        self.assertEqual(["Priya", "Visitor"],
                         sorted(e["label"] for e in r["entities"]))
        self.assertNotIn("Delivery Courier",
                         [e["label"] for e in r["entities"]],
                         "he was at the gate, but never with anyone")

    def test_an_anchor_nobody_matches_answers_with_nobody(self):
        r = execute(self.store, {"intent": "who_with",
                                 "entity_labels": ["the phantom"]})
        self.assertEqual([], r["entities"])
        self.assertEqual([], r["co_presence"])

    def test_the_anchor_holds_even_when_the_pushdown_is_dropped(self):
        """`ids is None` means two opposite things — no filter, or too many to
        bind — and only the row path has a loop that re-checks. This is the one
        place a dropped pushdown could widen "who was with Priya" into "everyone
        who was ever with anyone", so the anchor is re-checked against the
        snapshot's own members.
        """
        self.addCleanup(setattr, ask, "MAX_PUSHED_IDS", ask.MAX_PUSHED_IDS)
        ask.MAX_PUSHED_IDS = 0                  # nothing is ever pushed down
        r = execute(self.store, {"intent": "who_with", "entity_labels": ["priya"]})
        self.assertEqual(["Visitor"], [c["label"] for c in r["co_presence"]],
                         "the anchor is a rule, not an id list")

    def test_the_zone_reaches_the_snapshots(self):
        r = execute(self.store, {"intent": "who_with", "zones": ["bay"]})
        self.assertEqual([], r["co_presence"], "the pair was at the gate")

    def test_an_exclusion_still_applies_to_the_answer(self):
        """'who was with her, apart from X' constrains the answer, not the anchor."""
        r = execute(self.store, {"intent": "who_with", "entity_labels": ["priya"],
                                 "exclude_entity_labels": ["visitor"]})
        self.assertEqual([], r["entities"])

    def test_the_companion_carries_where_and_when_and_which_snapshot(self):
        c = execute(self.store, {"intent": "who_with",
                                 "entity_labels": ["priya"]})["co_presence"][0]
        self.assertEqual(["gate"], c["zones"])
        self.assertEqual([self.snap_pair], c["snapshot_ids"])
        self.assertEqual(1, c["n_snapshots"])


# --- 5. relations ------------------------------------------------------------
class TestRelations(_Base):

    def ask_priya(self, **extra):
        return execute(self.store, {"intent": "relations",
                                    "entity_labels": ["priya"], **extra})

    def test_a_thing_with_no_sightings_of_its_own_is_still_an_answer(self):
        """The row path cannot reach this at all.

        A forklift is never the *subject* of an observation — it is only ever
        what somebody was near — so a question answered out of `observations`
        returns nothing for it however the filters are set. Its entry is built
        from the edge's own supporting rows.
        """
        self.mine()
        r = self.ask_priya()
        self.assertEqual(["Forklift"], [e["label"] for e in r["entities"]])
        self.assertEqual([], self.store.observations(subject_entity_id=self.forklift),
                         "the fixture's premise: it has no sightings of its own")

    def test_the_edge_carries_its_weight_status_and_evidence(self):
        self.mine()
        edge = next(e for e in self.ask_priya()["relations"]
                    if e["predicate"] == "uses")
        self.assertEqual("Priya", edge["subject_label"])
        self.assertEqual("Forklift", edge["object_label"])
        self.assertEqual(4, edge["n_supporting"])
        self.assertEqual(sorted(self.near_ids), sorted(edge["observation_ids"]))
        self.assertIn(edge["status"], ("candidate", "confirmed"))

    def test_a_window_with_no_evidence_in_it_has_no_edge_in_it(self):
        """The window reaches the belief through its evidence, which is the only
        end it can: an edge has one lifetime of reinforcement behind it, so
        whether it falls inside a window is a question about where its rows do.
        """
        self.mine()
        self.assertEqual([], self.ask_priya(start=T0 - DAY, end=T0)["relations"])
        self.assertTrue(self.ask_priya(start=T0 - 3 * DAY, end=T0)["relations"])

    def test_a_zone_reaches_the_belief_the_same_way(self):
        self.mine()
        self.assertTrue(self.ask_priya(zones=["bay"])["relations"])
        self.assertEqual([], self.ask_priya(zones=["gate"])["relations"])

    def test_a_suppressed_belief_is_not_an_answer(self):
        self.mine()
        for e in self.store.relation_edges(subject_entity_ids=[self.priya]):
            self.store.suppress_relation(e["relation_id"])
        self.assertEqual([], self.ask_priya()["relations"])
        self.assertEqual([], self.ask_priya()["entities"])

    def test_a_place_answers_where_does_she_go(self):
        """A 'frequents' edge has no object at all — the place is the far end."""
        s = self.store
        for i in range(5):
            s.add_observation(self.priya, "present", location_id=self.gate,
                              timestamp=T0 - 20 * DAY + i * DAY, origin="detector")
        self.mine()
        r = self.ask_priya()
        self.assertIn("gate", [e["label"] for e in r["entities"]])
        self.assertIn(scene_subject(self.gate),
                      [e["object_entity_id"] for e in r["relations"]])

    def test_nothing_mined_yet_is_an_empty_answer_not_a_crash(self):
        r = self.ask_priya()                    # no mine() above
        self.assertEqual([], r["relations"])
        self.assertEqual([], r["entities"])


# --- 6. the contract the three share -----------------------------------------
class TestTheGroundedEvidenceContract(_Base):
    """Whatever table an answer starts in, it ends up the same shape."""

    def setUp(self):
        super().setUp()
        self.mine()

    def test_every_aggregate_answer_still_returns_entities_and_a_trace(self):
        for plan in ({"intent": "how_often", "entity_labels": ["van"]},
                     {"intent": "who_with", "entity_labels": ["priya"]},
                     {"intent": "relations", "entity_labels": ["priya"]}):
            with self.subTest(intent=plan["intent"]):
                r = execute(self.store, plan)
                self.assertEqual(plan, r["query"], "the trace is what arrived")
                self.assertEqual(plan["intent"], r["plan"]["intent"])
                self.assertEqual(r["total_observations"],
                                 sum(e["n_observations"] for e in r["entities"]),
                                 "the count must describe the rows shown")
                for e in r["entities"]:
                    self.assertLessEqual(e["first_seen"], e["last_seen"])
                    self.assertLessEqual(len(e["keyframes"]), 4)

    def test_every_claim_keeps_something_to_point_at(self):
        rec = execute(self.store, {"intent": "how_often",
                                   "entity_labels": ["van"]})["recurrence"][0]
        self.assertTrue(all(h["n_supporting"] > 0 for h in rec["habits"]))

        co = execute(self.store, {"intent": "who_with",
                                  "entity_labels": ["priya"]})["co_presence"][0]
        self.assertTrue(co["snapshot_ids"])

        rel = execute(self.store, {"intent": "relations",
                                   "entity_labels": ["priya"]})["relations"][0]
        self.assertTrue(rel["observation_ids"])

    def test_the_narrator_is_handed_the_pattern_not_the_rows_to_count(self):
        """`_facts` is the only thing the prose model sees. An aggregate missing
        from it is an aggregate the answer cannot mention."""
        facts = ask._facts(execute(self.store, {"intent": "how_often",
                                                "entity_labels": ["van"]}))
        self.assertIn("recurrence", facts)
        self.assertEqual(8, facts["recurrence"][0]["times_seen"])
        self.assertIn("of 8", facts["recurrence"][0]["most_often"])
        # Status travels with the claim: a candidate habit is a weaker statement
        # than a confirmed one, and prose that cannot see the difference will
        # state both as fact.
        self.assertTrue(any("candidate" in h or "confirmed" in h
                            for h in facts["recurrence"][0]["mined_habits"]))

        facts = ask._facts(execute(self.store, {"intent": "who_with",
                                                "entity_labels": ["priya"]}))
        self.assertEqual(["Priya"], facts["seen_together"][0]["with"])

        facts = ask._facts(execute(self.store, {"intent": "relations",
                                                "entity_labels": ["priya"]}))
        self.assertEqual("Forklift", facts["connections"][0]["to"])

    def test_the_tool_schema_and_the_normalizer_still_agree(self):
        emitted = set(ask.QUERY_TOOL["input_schema"]["properties"])
        understood = set(ask.normalize_plan({}))
        self.assertEqual(set(), emitted - understood,
                         "the planner can emit a field execute() never reads")
        self.assertIn("group_by", emitted)

    def test_the_supported_field_set_lists_everything_the_engine_honours(self):
        """The eval runner refuses to score a plan carrying a field the engine
        ignores. That list drifting from reality is how an ignored filter passes
        for a working one."""
        from intelligence_os.tests.test_search_quality import SUPPORTED_PLAN_FIELDS
        emitted = set(ask.QUERY_TOOL["input_schema"]["properties"])
        self.assertEqual(set(), emitted - SUPPORTED_PLAN_FIELDS)


if __name__ == "__main__":
    unittest.main(verbosity=2)
