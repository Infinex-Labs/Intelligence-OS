"""The widened query form (search plan Phase 4).

Phase 3 made the engine fast at answering the questions it could already hear.
Phase 4 is about the ones it could not: several places, several people, "anyone
except", "what did that camera see", "the last two", "how many". The old form
had one zone, one name and one substring, and — the part that actually hurt — an
unrepresentable question came back as the nearest representable one, with an
answer that looked just as confident.

So most of what is under test here is not "the new filter works". It is:

  1. each new field narrows to exactly what it names, alone and combined
  2. a filter that matches nothing returns nothing, rather than everything —
     the failure mode the single-`zone` field had, and the reason a widened
     form is not automatically an improvement
  3. the old five-slot shape still executes unchanged (`normalize_plan`)
  4. the subject filter's two faces — the id set pushed into SQL and the
     `accepts()` check in the loop — cannot disagree, whichever one runs
  5. `intent` routes without filtering, and `limit` caps subjects, not rows

`test_search_quality.py` holds the other half: that the twenty-six cases Phase 4
does not target answer byte-identically to before it.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from intelligence_os.tests import _stubs   # noqa: F401  (stubs the deps we lack)

import intelligence_os.ask as ask          # noqa: E402
from intelligence_os.ask import execute, normalize_plan   # noqa: E402
from intelligence_os.store import Store, scene_subject    # noqa: E402

T0 = 1_700_000_000.0


class _Base(unittest.TestCase):
    """Three zones on two cameras; people, an object, and a place-owned fact.

    Deliberately not the eval corpus: this suite needs to name every row it
    asserts on, and shares the eval fixture's shape only where that matters.
    """

    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._dir.cleanup)
        self.store = s = Store(db_path=Path(self._dir.name) / "t.db")
        self.addCleanup(s.close)

        self.bay = s.upsert_location("bay", {"polygon": [[0, 0]]}, camera_id="cam_a")
        self.yard = s.upsert_location("yard", {"polygon": [[5, 5]]}, camera_id="cam_b")
        self.aisle = s.upsert_location("aisle", {"polygon": [[9, 9]]}, camera_id="cam_a")

        self.ann = s.create_entity("person", label="Ann")
        self.bob = s.create_entity("person", label="Bob")
        self.courier = s.create_entity("person", label="Delivery Courier")
        self.van = s.create_entity("object", label="White Van")
        self.scene_bay = scene_subject(self.bay)

        # (subject, predicate, zone, camera, offset, confidence, origin)
        self.spec = [
            (self.ann,       "present",         self.bay,   "cam_a",   0, 0.9, "detector"),
            (self.ann,       "state:smoking",   self.bay,   "cam_a",   0, 0.8, "vlm"),
            (self.bob,       "present",         self.yard,  "cam_b",  60, 0.4, "detector"),
            (self.courier,   "present",         self.yard,  "cam_b", 120, 0.9, "detector"),
            (self.van,       "present",         self.aisle, "cam_a", 180, 0.7, "detector"),
            (self.scene_bay, "object:gate — open", self.bay, "cam_a", 240, 0.9, "vlm"),
        ]
        for subj, pred, zone, cam, off, conf, origin in self.spec:
            s.add_observation(subj, pred, location_id=zone, camera_id=cam,
                              timestamp=T0 + off, confidence=conf, origin=origin,
                              text=pred)

    def who(self, **plan) -> list[str]:
        """The labels a plan answers with, sorted so order is asserted separately."""
        return sorted(e["label"] for e in execute(self.store, plan)["entities"])

    def order_of(self, **plan) -> list[str]:
        return [e["label"] for e in execute(self.store, plan)["entities"]]


# --- 1. each field narrows to what it names ----------------------------------
class TestEachNewField(_Base):

    def test_zones_takes_several(self):
        self.assertEqual(self.who(zones=["bay", "aisle"]),
                         ["Ann", "White Van", "bay"])

    def test_one_zone_in_a_list_is_the_old_single_zone(self):
        self.assertEqual(self.who(zones=["yard"]), ["Bob", "Delivery Courier"])

    def test_cameras_reaches_a_column_recorded_since_m6_and_never_searchable(self):
        self.assertEqual(self.who(cameras=["cam_b"]), ["Bob", "Delivery Courier"])

    def test_entity_labels_takes_several_people(self):
        self.assertEqual(self.who(entity_labels=["ann", "bob"]), ["Ann", "Bob"])

    def test_exclude_entity_labels_is_anyone_except(self):
        self.assertEqual(self.who(zones=["yard"],
                                  exclude_entity_labels=["delivery courier"]),
                         ["Bob"])

    def test_entity_type_person(self):
        self.assertEqual(self.who(entity_type="person"),
                         ["Ann", "Bob", "Delivery Courier"])

    def test_entity_type_object_excludes_people_and_places(self):
        self.assertEqual(self.who(entity_type="object"), ["White Van"])

    def test_entity_type_scene_is_the_facts_nobody_owns(self):
        # A gate is not an entity anyone can point at — the fact is recorded
        # against the place, and `scene` is how a question reaches those.
        self.assertEqual(self.who(entity_type="scene"), ["bay"])

    def test_min_confidence_is_a_floor_not_a_ranking(self):
        self.assertEqual(self.who(min_confidence=0.85), ["Ann", "Delivery Courier", "bay"])

    def test_exclude_predicates_drops_exact_predicates(self):
        self.assertEqual(self.who(zones=["bay"], exclude_predicates=["present"]), ["Ann", "bay"])

    def test_limit_caps_the_subjects_returned(self):
        self.assertEqual(len(execute(self.store, {"limit": 2})["entities"]), 2)

    def test_order_desc_puts_the_most_recent_first(self):
        self.assertEqual(self.order_of(order="desc")[0], "bay")
        self.assertEqual(self.order_of(order="asc")[0], "Ann")

    def test_filters_compose_by_and_not_by_or(self):
        # cam_a covers bay and aisle; the zone narrows it further. If these
        # OR-ed, the yard would come back too.
        self.assertEqual(self.who(cameras=["cam_a"], zones=["aisle"]), ["White Van"])

    def test_a_filter_can_narrow_to_nothing_without_erroring(self):
        self.assertEqual(self.who(zones=["bay"], entity_labels=["bob"]), [])


# --- 2. narrowing must never widen -------------------------------------------
class TestNarrowingNeverWidens(_Base):
    """The failure the old form actually had, in every place it could recur.

    A filter the engine cannot satisfy has exactly one honest answer — nothing.
    The tempting bug in all four of these is to treat "no match" as "no filter",
    which turns the narrowest question in the system into the widest.
    """

    def test_an_unknown_zone_answers_with_nothing_not_everything(self):
        self.assertEqual(self.who(zones=["car_park"]), [])

    def test_one_real_zone_and_one_imaginary_keeps_only_the_real_one(self):
        self.assertEqual(self.who(zones=["bay", "car_park"]), ["Ann", "bay"])

    def test_an_unknown_camera_answers_with_nothing(self):
        self.assertEqual(self.who(cameras=["cam_roof"]), [])

    def test_an_unknown_name_answers_with_nothing(self):
        self.assertEqual(self.who(entity_labels=["mallory"]), [])

    def test_excluding_everyone_leaves_nobody(self):
        # Between them these letters appear in every label in the fixture.
        self.assertTrue(self.who())
        self.assertEqual(self.who(exclude_entity_labels=["a", "b", "e"]), [])

    def test_an_empty_list_is_no_filter_not_an_impossible_one(self):
        # The other direction, and the reason the two cannot share a code path:
        # a field the planner simply left empty must not mean "nowhere".
        everyone = self.who()
        self.assertEqual(self.who(zones=[], cameras=[], entity_labels=[]), everyone)
        self.assertEqual(len(everyone), 5)


# --- 3. the old shape still executes -----------------------------------------
class TestTheOldShapeStillExecutes(_Base):
    """The planner is one tool-call boundary, so the previous form is a dict a
    caller may still send. It up-converts; it does not error and does not widen."""

    def test_zone_becomes_zones(self):
        self.assertEqual(self.who(zone="yard"), self.who(zones=["yard"]))

    def test_entity_label_becomes_entity_labels(self):
        self.assertEqual(self.who(entity_label="ann"), self.who(entity_labels=["ann"]))

    def test_the_whole_five_slot_plan_still_answers(self):
        self.assertEqual(
            self.who(start=None, end=None, zone="bay", entity_label="ann",
                     predicate_contains="smoking"),
            ["Ann"])

    def test_predicate_contains_is_kept_not_folded_into_text(self):
        # They are not synonyms. `text` is the lexical index alone; this also
        # keeps a literal substring test, which is the machine contract that
        # 'rule_fired' is matched on. Folding it in would drop that half.
        p = normalize_plan({"predicate_contains": "rule_fired"})
        self.assertEqual(p["predicate_contains"], "rule_fired")
        self.assertEqual(p["text"], "")

    def test_the_new_shape_wins_when_both_are_present(self):
        p = normalize_plan({"zone": "bay", "zones": ["yard"]})
        self.assertEqual(p["zones"], ["yard"])

    def test_a_bare_string_where_a_list_was_documented(self):
        # The planner is a language model; a documented array arrives as a
        # string often enough that dropping it would silently widen the answer.
        self.assertEqual(normalize_plan({"zones": "bay"})["zones"], ["bay"])

    def test_an_unknown_intent_answers_the_question_rather_than_refusing(self):
        self.assertEqual(normalize_plan({"intent": "vibes"})["intent"], "who")

    def test_junk_in_the_numeric_fields_is_dropped_not_raised(self):
        p = normalize_plan({"limit": "two", "min_confidence": "high"})
        self.assertIsNone(p["limit"])
        self.assertIsNone(p["min_confidence"])

    def test_a_limit_of_zero_is_not_a_real_question(self):
        self.assertIsNone(normalize_plan({"limit": 0})["limit"])

    def test_an_empty_plan_is_a_complete_plan(self):
        p = normalize_plan({})
        self.assertEqual(p["intent"], "who")
        self.assertEqual(p["entity_type"], "any")
        self.assertEqual(p["order"], "asc")
        self.assertEqual(p["zones"], [])


# --- 4. the subject filter's two faces cannot disagree -----------------------
class TestPushdownAndLoopAgree(_Base):
    """`ids` narrows the SQL; `accepts()` decides the answer. SQL may only
    remove rows `accepts()` would have removed anyway — which is what makes the
    bound-variable fallback safe rather than a second, weaker filter."""

    PLANS = [
        {"entity_labels": ["ann"]},
        {"entity_labels": ["ann", "bob"]},
        {"exclude_entity_labels": ["delivery courier"]},
        {"entity_type": "object"},
        {"entity_type": "person", "exclude_entity_labels": ["ann"]},
        {"entity_type": "scene"},
        {"entity_labels": ["a"], "exclude_entity_labels": ["van"]},
    ]

    def test_dropping_the_pushdown_never_changes_the_answer(self):
        self.addCleanup(setattr, ask, "MAX_PUSHED_IDS", ask.MAX_PUSHED_IDS)
        for plan in self.PLANS:
            with self.subTest(plan=plan):
                ask.MAX_PUSHED_IDS = 900
                pushed = self.who(**plan)
                ask.MAX_PUSHED_IDS = 0        # nothing fits; every set falls back
                self.assertEqual(pushed, self.who(**plan))

    def test_the_fallback_really_is_a_fallback(self):
        self.addCleanup(setattr, ask, "MAX_PUSHED_IDS", ask.MAX_PUSHED_IDS)
        ask.MAX_PUSHED_IDS = 0
        f = ask._SubjectFilter(self.store, normalize_plan({"entity_labels": ["ann"]}), {})
        self.assertIsNone(f.ids, "the id set must be dropped, not truncated")

    def test_an_exclusion_alone_does_not_enumerate_every_subject(self):
        # Resolving a positive set in order to leave one person out would mean
        # binding every subject that exists. The exclusion is its own small set.
        f = ask._SubjectFilter(
            self.store, normalize_plan({"exclude_entity_labels": ["ann"]}), {})
        self.assertIsNone(f.ids)
        self.assertEqual(f.exclude_ids, [self.ann])

    def test_a_type_filter_skips_the_half_it_cannot_match(self):
        # `scene:` subjects come from a DISTINCT over the observations, so a
        # person-only question must not pay for it — and vice versa.
        seen = []
        orig = Store.scene_subjects
        self.addCleanup(setattr, Store, "scene_subjects", orig)
        Store.scene_subjects = lambda s: seen.append(1) or orig(s)
        ask._SubjectFilter(self.store, normalize_plan({"entity_type": "person"}), {})
        self.assertEqual(seen, [])
        ask._SubjectFilter(self.store, normalize_plan({"entity_type": "scene"}), {})
        self.assertEqual(seen, [1])

    def test_a_name_matches_the_displayed_label_not_the_stored_one(self):
        # An unlabelled person is shown as "Person <hex>", and that synthesized
        # name is what a follow-up question would quote back.
        ghost = self.store.create_entity("person")
        self.store.add_observation(ghost, "present", location_id=self.bay,
                                   timestamp=T0 + 300, origin="detector")
        self.assertEqual(self.who(entity_labels=[f"person {ghost[-6:]}"]),
                         [f"Person {ghost[-6:]}"])

    def test_a_place_owned_fact_is_named_by_its_zone_in_both_faces(self):
        self.assertEqual(self.who(entity_labels=["bay"]), ["bay"])


# --- 5. intent routes; limit caps subjects -----------------------------------
class TestIntent(_Base):

    def test_count_promotes_the_number_to_the_answer(self):
        r = execute(self.store, {"intent": "count", "zones": ["bay"]})
        self.assertEqual(r["count"], 3)

    def test_a_count_never_disagrees_with_the_rows_behind_it(self):
        r = execute(self.store, {"intent": "count"})
        self.assertEqual(r["count"], r["total_observations"])
        self.assertEqual(r["count"],
                         sum(e["n_observations"] for e in r["entities"]))

    def test_counting_still_returns_the_evidence(self):
        # "How many" with no rows attached is a number the user cannot check,
        # which is the one thing this system is built not to produce.
        self.assertTrue(execute(self.store, {"intent": "count"})["entities"])

    def test_a_non_counting_intent_carries_no_count_key(self):
        self.assertNotIn("count", execute(self.store, {"intent": "who"}))

    def test_intent_does_not_filter(self):
        rows = self.who()
        for intent in ("who", "when", "count", "timeline", "last", "how_often"):
            with self.subTest(intent=intent):
                self.assertEqual(self.who(intent=intent), rows)

    def test_a_limit_caps_subjects_rather_than_rows(self):
        # Ann has two rows. A row limit of 2 would return Ann alone; a subject
        # limit of 2 returns two subjects, each with its whole history.
        r = execute(self.store, {"limit": 2, "order": "asc"})
        self.assertEqual([e["label"] for e in r["entities"]], ["Ann", "Bob"])
        self.assertEqual(r["entities"][0]["n_observations"], 2)

    def test_the_total_describes_only_what_was_returned(self):
        capped = execute(self.store, {"limit": 1})
        self.assertEqual(capped["total_observations"],
                         capped["entities"][0]["n_observations"])

    def test_a_limited_question_does_not_read_the_whole_memory(self):
        # "The last two sightings" has no filter to push down, so without the
        # shortlist walk it aggregates every row ever recorded in order to throw
        # all but two subjects away.
        for i in range(400):
            e = self.store.create_entity("person", label=f"Extra {i}")
            self.store.add_observation(e, "present", location_id=self.bay,
                                       timestamp=T0 - 10_000 - i, origin="detector")
        rows = []
        orig = Store.observations
        self.addCleanup(setattr, Store, "observations", orig)

        def spy(s, *a, **kw):
            out = orig(s, *a, **kw)
            rows.append(len(out))
            return out
        Store.observations = spy
        r = execute(self.store, {"limit": 2, "order": "desc"})
        self.assertEqual([e["label"] for e in r["entities"]], ["bay", "White Van"])
        self.assertLess(max(rows), 406, "the walk should be bounded, not a full read")

    def test_the_walk_finds_the_same_subjects_a_full_read_would(self):
        # The shortcut is only sound because a subject's last_seen is where it
        # first appears reading backwards. Checked against the slow path, over
        # every combination of direction and depth the fixture can express.
        self.addCleanup(setattr, ask, "LIMIT_PROBE_START", ask.LIMIT_PROBE_START)
        for order in ("asc", "desc"):
            for n in range(1, 7):
                with self.subTest(order=order, limit=n):
                    ask.LIMIT_PROBE_START = 256
                    fast = self.order_of(order=order, limit=n)
                    ask.LIMIT_PROBE_START = 1     # forces the walk to grow
                    self.assertEqual(fast, self.order_of(order=order, limit=n))

    def test_giving_up_on_the_walk_still_answers_correctly(self):
        self.addCleanup(setattr, ask, "LIMIT_PROBE_MAX", ask.LIMIT_PROBE_MAX)
        self.addCleanup(setattr, ask, "LIMIT_PROBE_START", ask.LIMIT_PROBE_START)
        expected = self.order_of(order="desc", limit=3)
        ask.LIMIT_PROBE_START = ask.LIMIT_PROBE_MAX = 1
        self.assertEqual(self.order_of(order="desc", limit=3), expected)

    def test_the_walk_respects_the_other_filters(self):
        self.assertEqual(self.order_of(order="desc", limit=2, entity_type="person"),
                         ["Delivery Courier", "Bob"])
        self.assertEqual(self.order_of(order="desc", limit=2, zones=["bay"]),
                         ["bay", "Ann"])

    def test_a_limited_word_question_skips_the_walk(self):
        # A ranked answer is ordered by score, so the first rows read are not
        # the first results and the walk would pick the wrong subjects.
        r = execute(self.store, {"text": "present", "limit": 1, "order": "desc"})
        self.assertEqual(len(r["entities"]), 1)
        self.assertTrue(r["ranked"])

    def test_the_trace_records_the_plan_that_ran_not_the_one_that_arrived(self):
        # The old shape executes, so the trace has to show what it became —
        # otherwise the inspectable query and the executed query are different
        # things and only one of them is shown.
        r = execute(self.store, {"zone": "bay"})
        self.assertEqual(r["query"], {"zone": "bay"})
        self.assertEqual(r["plan"]["zones"], ["bay"])


# --- the shortlist is drawn from the filtered pool ---------------------------
class TestWordQuestionsRespectTheHardFilters(_Base):

    def test_a_word_question_still_obeys_the_zone(self):
        self.assertEqual(self.who(text="smoking", zones=["yard"]), [])
        self.assertEqual(self.who(text="smoking", zones=["bay"]), ["Ann"])

    def test_the_ranked_shortlist_is_drawn_from_the_narrowed_pool(self):
        # search_text caps its ranked list, so filtering after it would cost
        # recall on exactly the narrow questions the filters were added for.
        seen = {}
        orig = Store.search_text
        self.addCleanup(setattr, Store, "search_text", orig)

        def spy(s, q, **kw):
            seen.update(kw)
            return orig(s, q, **kw)
        Store.search_text = spy
        execute(self.store, {"text": "smoking", "zones": ["bay"],
                             "cameras": ["cam_a"], "entity_labels": ["ann"]})
        self.assertEqual(seen["location_ids"], [self.bay])
        self.assertEqual(seen["camera_ids"], ["cam_a"])
        self.assertEqual(seen["entity_ids"], [self.ann])

    def test_a_word_question_is_still_ranked(self):
        self.assertTrue(execute(self.store, {"text": "smoking"})["ranked"])
        self.assertFalse(execute(self.store, {"zones": ["bay"]})["ranked"])


# --- the planner's vocabulary ------------------------------------------------
class TestThePlannerVocabulary(_Base):

    def test_cameras_are_read_off_the_zones_not_the_observations(self):
        self.assertEqual(self.store.cameras(), ["cam_a", "cam_b"])

    def test_a_camera_with_no_zone_is_still_offered_via_the_config(self):
        self.addCleanup(setattr, ask, "_known_cameras", ask._known_cameras)
        import intelligence_os.config as config
        orig = config.resolve_cameras
        self.addCleanup(setattr, config, "resolve_cameras", orig)
        config.resolve_cameras = lambda *a, **k: [{"name": "cam_roof", "source": 0}]
        self.assertEqual(ask._known_cameras(self.store),
                         ["cam_a", "cam_b", "cam_roof"])

    def test_a_broken_camera_config_does_not_take_the_query_down(self):
        import intelligence_os.config as config
        orig = config.resolve_cameras
        self.addCleanup(setattr, config, "resolve_cameras", orig)

        def boom(*a, **k):
            raise ValueError("Duplicate camera names in config")
        config.resolve_cameras = boom
        self.assertEqual(ask._known_cameras(self.store), ["cam_a", "cam_b"])

    def test_the_tool_schema_and_the_normalizer_agree_on_the_field_names(self):
        # A field the planner can emit but `normalize_plan` drops is a filter
        # that silently disappears, which is the whole class of bug Phase 4 is
        # about. Checked mechanically so a later edit to one must touch both.
        emitted = set(ask.QUERY_TOOL["input_schema"]["properties"])
        understood = set(normalize_plan({}))
        self.assertEqual(emitted - understood, set())

    def test_every_required_field_is_one_the_schema_declares(self):
        props = ask.QUERY_TOOL["input_schema"]["properties"]
        for name in ask.QUERY_TOOL["input_schema"]["required"]:
            self.assertIn(name, props)


if __name__ == "__main__":
    unittest.main(verbosity=2)
