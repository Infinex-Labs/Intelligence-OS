"""Filtering in the database (search plan Phase 3).

Phase 3 moved the window, the zone and the words out of a Python loop and into
one `WHERE` clause. Four things are under test:

  1. every filter, on its own and combined, selects exactly what it says
  2. the two conventions that are easy to get backwards — an empty sequence
     matches nothing, and a user's `%` is a literal
  3. the ordering is *total*, so adding a filter cannot reshuffle equal rows
  4. `ask.execute()` no longer issues one query per row

What is NOT here is whether answers got better: Phase 3 is a pure refactor and
`test_search_quality.py` asserts, against the recorded baseline, that they did
not change at all.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from intelligence_os.tests import _stubs   # noqa: F401  (stubs the deps we lack)

from intelligence_os.ask import execute    # noqa: E402
from intelligence_os.store import Store    # noqa: E402

T0 = 1_700_000_000.0


class _Base(unittest.TestCase):
    """A small memory with two people, two zones, two cameras and known rows."""

    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._dir.cleanup)
        self.store = Store(db_path=Path(self._dir.name) / "t.db")
        self.addCleanup(self.store.close)
        s = self.store

        self.bay = s.upsert_location("bay", {"polygon": [[0, 0], [1, 0], [1, 1]]})
        self.yard = s.upsert_location("yard", {"polygon": [[5, 5], [6, 5], [6, 6]]})
        self.ann = s.create_entity("person", label="Ann")
        self.bob = s.create_entity("person", label="Bob")

        # (subject, predicate, zone, camera, t_offset, confidence, origin)
        self.spec = [
            (self.ann, "present",              self.bay,  "cam_a",   0, 0.9, "detector"),
            (self.ann, "state:smoking",        self.bay,  "cam_a",   0, 0.8, "vlm"),
            (self.ann, "state:on phone",       self.bay,  "cam_a",  60, 0.4, "vlm"),
            (self.bob, "present",              self.yard, "cam_b", 120, 0.9, "detector"),
            (self.bob, "rule_fired:loitering", self.yard, "cam_b", 180, 1.0, "rule"),
        ]
        self.ids = [
            s.add_observation(subj, pred, location_id=zone, camera_id=cam,
                              timestamp=T0 + off, confidence=conf, origin=origin,
                              text=pred)
            for subj, pred, zone, cam, off, conf, origin in self.spec
        ]

    def preds(self, rows):
        return [r["predicate"] for r in rows]


class TestEachFilter(_Base):

    def test_no_filter_returns_everything_in_time_order(self):
        self.assertEqual(self.preds(self.store.observations()),
                         [s[1] for s in self.spec])

    def test_window_is_inclusive_at_both_ends(self):
        rows = self.store.observations(since=T0 + 60, until=T0 + 120)
        self.assertEqual(self.preds(rows), ["state:on phone", "present"])

    def test_location_ids(self):
        rows = self.store.observations(location_ids=[self.yard])
        self.assertEqual(self.preds(rows), ["present", "rule_fired:loitering"])

    def test_several_locations_are_a_union(self):
        rows = self.store.observations(location_ids=[self.bay, self.yard])
        self.assertEqual(len(rows), 5)

    def test_entity_ids(self):
        rows = self.store.observations(entity_ids=[self.bob])
        self.assertEqual(self.preds(rows), ["present", "rule_fired:loitering"])

    def test_observation_ids(self):
        rows = self.store.observations(observation_ids=[self.ids[1], self.ids[4]])
        self.assertEqual(self.preds(rows), ["state:smoking", "rule_fired:loitering"])

    def test_predicate_contains_is_a_substring_anywhere(self):
        self.assertEqual(self.preds(self.store.observations(predicate_contains="phon")),
                         ["state:on phone"])

    def test_predicate_contains_is_case_insensitive(self):
        self.assertEqual(len(self.store.observations(predicate_contains="SMOK")), 1)

    def test_predicate_prefixes_anchor_at_the_start(self):
        self.assertEqual(self.preds(self.store.observations(
            predicate_prefixes=["rule_fired:"])), ["rule_fired:loitering"])
        # 'smoking' appears in a predicate, but never at the front of one
        self.assertEqual(self.store.observations(predicate_prefixes=["smoking"]), [])

    def test_exclude_predicates_is_exact_not_substring(self):
        rows = self.store.observations(exclude_predicates=["present"])
        self.assertEqual(self.preds(rows),
                         ["state:smoking", "state:on phone", "rule_fired:loitering"])

    def test_origins(self):
        self.assertEqual(len(self.store.observations(origins=["vlm"])), 2)
        self.assertEqual(len(self.store.observations(origins=["vlm", "rule"])), 3)

    def test_min_confidence_is_inclusive(self):
        self.assertEqual(len(self.store.observations(min_confidence=0.8)), 4)
        self.assertEqual(len(self.store.observations(min_confidence=0.9)), 3)

    def test_camera_and_user(self):
        self.assertEqual(len(self.store.observations(camera_id="cam_a")), 3)
        self.assertEqual(len(self.store.observations(user_id="nobody")), 0)

    def test_filters_compose_as_and(self):
        rows = self.store.observations(
            since=T0, until=T0 + 60, location_ids=[self.bay],
            origins=["vlm"], min_confidence=0.5)
        self.assertEqual(self.preds(rows), ["state:smoking"])

    def test_a_filter_that_excludes_everything_returns_nothing(self):
        self.assertEqual(self.store.observations(
            location_ids=[self.bay], entity_ids=[self.bob]), [])


class TestTheTwoEasilyInvertedConventions(_Base):
    """None means no filter; an empty sequence means a filter nothing satisfies.

    Collapsing those two would turn 'in any of these zones' over an empty zone
    list into 'anywhere' — a search that silently returns the whole table at the
    moment it was asked to narrow.
    """

    def test_none_is_no_filter(self):
        self.assertEqual(len(self.store.observations(location_ids=None)), 5)

    def test_empty_permits_nothing(self):
        for kwargs in ({"location_ids": []}, {"entity_ids": []},
                       {"observation_ids": []}, {"origins": []},
                       {"predicate_prefixes": []}):
            with self.subTest(**kwargs):
                self.assertEqual(self.store.observations(**kwargs), [])

    def test_an_empty_exclusion_excludes_nothing(self):
        # The one inverted filter: read from the other side, the same rule.
        self.assertEqual(len(self.store.observations(exclude_predicates=[])), 5)

    def test_a_wildcard_in_the_query_is_a_literal(self):
        self.store.add_observation(self.ann, "state:100% sure", location_id=self.bay,
                                   timestamp=T0 + 300, origin="vlm", text="x")
        # If '%' reached LIKE unescaped this would match every row instead of one.
        self.assertEqual(len(self.store.observations(predicate_contains="%")), 1)
        self.assertEqual(len(self.store.observations(predicate_contains="100%")), 1)
        self.assertEqual(self.store.observations(predicate_contains="pres_nt"), [])
        # '_' is LIKE's single-character wildcard; escaped, it finds the one
        # predicate that really contains an underscore rather than all of them.
        self.assertEqual(self.preds(self.store.observations(predicate_contains="_")),
                         ["rule_fired:loitering"])


class TestOrderingIsTotal(_Base):
    """One frame writes several rows at one instant, so ties are the common case
    and not an edge case. If ties broke by query plan, adding a zone filter could
    reorder the states listed under a person for no reason the user can see."""

    def setUp(self):
        super().setUp()
        for i in range(6):
            self.store.add_observation(self.ann, f"state:tied {i}", location_id=self.bay,
                                       timestamp=T0 + 999, origin="vlm", text="tied")

    def _tied(self, rows):
        return [r["predicate"] for r in rows if r["predicate"].startswith("state:tied")]

    def test_ties_break_by_insertion_order(self):
        self.assertEqual(self._tied(self.store.observations()),
                         [f"state:tied {i}" for i in range(6)])

    def test_a_narrower_filter_does_not_reshuffle_them(self):
        wide = self._tied(self.store.observations())
        for kwargs in ({"location_ids": [self.bay]}, {"entity_ids": [self.ann]},
                       {"origins": ["vlm"]}, {"since": T0 + 999}):
            with self.subTest(**kwargs):
                self.assertEqual(self._tied(self.store.observations(**kwargs)), wide)

    def test_desc_is_the_exact_reverse(self):
        self.assertEqual(self._tied(self.store.observations(order="desc")),
                         list(reversed(self._tied(self.store.observations()))))

    def test_limit_takes_from_the_ordered_end(self):
        self.assertEqual(
            self.preds(self.store.observations(order="desc", limit=1)),
            ["state:tied 5"])

    def test_an_unknown_order_is_refused_rather_than_interpolated(self):
        with self.assertRaises(ValueError):
            self.store.observations(order="asc; DROP TABLE observations")


class TestCounting(_Base):

    def test_count_matches_the_rows_for_every_filter(self):
        for kwargs in ({}, {"since": T0 + 60}, {"location_ids": [self.bay]},
                       {"origins": ["vlm"]}, {"min_confidence": 0.9},
                       {"predicate_contains": "state:"},
                       {"exclude_predicates": ["present"]},
                       {"location_ids": []}):
            with self.subTest(**kwargs):
                self.assertEqual(self.store.count_observations(**kwargs),
                                 len(self.store.observations(**kwargs)))

    def test_count_takes_the_subject_positionally_like_observations(self):
        self.assertEqual(self.store.count_observations(self.ann), 3)


class TestTheUnionIsExpressedInSql(_Base):
    """Phase 2's rule — the ranked ids OR the substring — now lives in the WHERE
    clause. It must widen recall without ever crossing a hard filter."""

    def test_match_any_is_a_union(self):
        rows = self.store.observations(observation_ids=[self.ids[0]],
                                       predicate_contains="rule_fired",
                                       match_any=True)
        self.assertEqual(self.preds(rows), ["present", "rule_fired:loitering"])

    def test_without_match_any_the_same_two_intersect_to_nothing(self):
        self.assertEqual(self.store.observations(observation_ids=[self.ids[0]],
                                                 predicate_contains="rule_fired"), [])

    def test_hard_filters_still_bound_the_union(self):
        # Both branches would match; the zone says no, and the zone wins.
        rows = self.store.observations(location_ids=[self.bay],
                                       observation_ids=[self.ids[3]],
                                       predicate_contains="rule_fired",
                                       match_any=True)
        self.assertEqual(rows, [])


class TestBatchEntityLoad(_Base):

    def test_returns_a_map_and_omits_ids_that_do_not_exist(self):
        got = self.store.get_entities([self.ann, "ent_nope", self.bob])
        self.assertEqual(set(got), {self.ann, self.bob})
        self.assertEqual(got[self.ann]["label"], "Ann")

    def test_duplicates_are_asked_for_once(self):
        self.assertEqual(len(self.store.get_entities([self.ann] * 50)), 1)

    def test_more_ids_than_sqlite_allows_in_one_statement(self):
        # The chunking is the point: 900 is under the 999-variable cap of the
        # older builds, and the id set here is one per row in the worst case.
        made = [self.store.create_entity("object") for _ in range(5)]
        got = self.store.get_entities(made + [f"ent_missing_{i}" for i in range(2000)])
        self.assertEqual(set(got), set(made))

    def test_nothing_asked_is_nothing_queried(self):
        self.assertEqual(self.store.get_entities([]), {})


class TestAskIssuesOneQueryPerEntitySet(_Base):
    """The N+1 this phase removed: `execute()` used to call `get_entity()` once
    per observation row, so the cost of answering grew with how long someone
    stood in front of the camera rather than with how many people there were."""

    def _statements(self, query):
        seen: list[str] = []
        self.store.conn.set_trace_callback(seen.append)
        try:
            result = execute(self.store, query)
        finally:
            self.store.conn.set_trace_callback(None)
        return result, seen

    def test_one_entity_statement_however_many_rows(self):
        for i in range(40):
            self.store.add_observation(self.ann, f"state:pacing {i}",
                                       location_id=self.bay, timestamp=T0 + 400 + i,
                                       origin="vlm", text="pacing")
        result, seen = self._statements({"start": None, "end": None, "zone": None,
                                         "entity_label": None,
                                         "predicate_contains": None})
        self.assertGreater(result["total_observations"], 40)
        entity_reads = [s for s in seen if "FROM entities" in s]
        self.assertEqual(len(entity_reads), 1, entity_reads)

    def test_a_named_person_costs_the_same_whatever_else_was_recorded(self):
        """The cost of asking about Ann must not depend on how long Bob stood
        there. It used to: every one of Bob's rows was a `get_entity` call."""
        query = {"start": None, "end": None, "zone": None,
                 "entity_label": "Ann", "predicate_contains": None}
        result, few = self._statements(query)
        self.assertEqual([e["label"] for e in result["entities"]], ["Ann"])

        for i in range(200):
            self.store.add_observation(self.bob, f"state:waiting {i}",
                                       location_id=self.yard, timestamp=T0 + 500 + i,
                                       origin="vlm", text="waiting")
        result, many = self._statements(query)
        self.assertEqual([e["label"] for e in result["entities"]], ["Ann"])
        self.assertEqual(len(many), len(few), "cost grew with unrelated rows")

    def test_a_name_is_resolved_to_ids_and_the_rest_never_read(self):
        for i in range(50):
            self.store.add_observation(self.bob, f"state:waiting {i}",
                                       location_id=self.yard, timestamp=T0 + 500 + i,
                                       origin="vlm", text="waiting")
        result, seen = self._statements({"start": None, "end": None, "zone": None,
                                         "entity_label": "Ann",
                                         "predicate_contains": None})
        self.assertEqual(result["total_observations"], 3)   # Ann's rows, not Bob's
        # The statement that fetches rows is narrowed to the ids the name
        # resolved to; the other observation read is the scene-subject range
        # scan, which is answered from the index and never touches a row.
        reads = [s for s in seen if s.startswith("SELECT * FROM observations")]
        self.assertEqual(len(reads), 1)
        self.assertIn("subject_entity_id IN", reads[0])

    def test_the_window_is_not_read_and_then_discarded(self):
        # A question about one minute must not scan the rows outside it.
        _, seen = self._statements({"start": T0 + 60, "end": T0 + 60, "zone": None,
                                    "entity_label": None, "predicate_contains": None})
        # The trace callback reports statements with their values substituted in.
        reads = [s for s in seen if "FROM observations" in s]
        self.assertTrue(reads, "no observation query was issued at all")
        self.assertTrue(all("timestamp>=" in s and "timestamp<=" in s
                            for s in reads), reads)


class TestANameIsNotAColumn(_Base):
    """The label filter is defined on the *displayed* name, which is derived:
    an unlabeled entity shows as "Person 4f2a91", and a fact nobody owns shows
    as the zone it happened in. Pushing that into SQL means resolving it to ids
    first, and the resolution has to agree with what the loop displays — every
    subject reachable by name before must still be reachable by name."""

    def _labels(self, entity_label):
        return sorted(e["label"] for e in execute(self.store, {
            "start": None, "end": None, "zone": None,
            "entity_label": entity_label, "predicate_contains": None})["entities"])

    def test_a_labelled_person(self):
        self.assertEqual(self._labels("ann"), ["Ann"])

    def test_an_unlabelled_entity_by_its_synthesized_name(self):
        ghost = self.store.create_entity("person")
        self.store.add_observation(ghost, "present", location_id=self.bay,
                                   timestamp=T0 + 10, origin="detector")
        self.assertEqual(self._labels(f"person {ghost[-6:]}"),
                         [f"Person {ghost[-6:]}"])

    def test_a_place_owned_fact_by_the_zone_name(self):
        from intelligence_os.store import scene_subject
        self.store.add_observation(scene_subject(self.bay), "object:gate — open",
                                   location_id=self.bay, timestamp=T0 + 20,
                                   origin="vlm", text="gate open")
        self.assertEqual(self._labels("bay"), ["bay"])

    def test_a_place_owned_fact_keyed_by_camera_rather_than_zone(self):
        # scene_subject() falls back to the camera when there is no zone, and
        # then the displayed name is the camera id itself.
        from intelligence_os.store import scene_subject
        sid = scene_subject(None, "cam_roof")
        self.store.add_observation(sid, "area:roof — hatch open", timestamp=T0 + 30,
                                   camera_id="cam_roof", origin="vlm", text="hatch")
        self.assertEqual(self._labels("cam_roof"), ["cam_roof"])

    def test_a_merged_away_entity_is_still_reachable_by_name(self):
        # get_entity() never filtered on status, so neither may the id set.
        gone = self.store.create_entity("person", label="Zed")
        self.store.add_observation(gone, "present", location_id=self.bay,
                                   timestamp=T0 + 40, origin="detector")
        self.store.conn.execute(
            "UPDATE entities SET status='merged_into:x' WHERE entity_id=?", (gone,))
        self.store.conn.commit()
        self.assertEqual(self._labels("zed"), ["Zed"])

    def test_a_name_nothing_matches_returns_nothing(self):
        self.assertEqual(self._labels("mallory"), [])

    def test_too_many_matches_falls_back_instead_of_overflowing_sqlite(self):
        # More candidate ids than can be bound in one statement: the filter goes
        # back to the loop rather than the query failing.
        import intelligence_os.ask as ask
        crowd = [self.store.create_entity("person", label=f"Crowd Member {i}")
                 for i in range(40)]
        for i, eid in enumerate(crowd):
            self.store.add_observation(eid, "present", location_id=self.bay,
                                       timestamp=T0 + 100 + i, origin="detector")
        self.addCleanup(setattr, ask, "MAX_PUSHED_IDS", ask.MAX_PUSHED_IDS)
        ask.MAX_PUSHED_IDS = 5

        plan = ask.normalize_plan({"entity_labels": ["crowd"]})
        self.assertIsNone(ask._SubjectFilter(self.store, plan, {}).ids,
                          "too many to bind — the pushdown must be dropped")
        # Dropped, not weakened: the loop is the authority, so the answer is the
        # same one the pushdown would have given, only slower.
        self.assertEqual(len(self._labels("crowd")), 40)
        self.assertEqual(self._labels("ann"), ["Ann"])   # still answers


class TestIndexes(_Base):

    def _indexes(self):
        return {r["name"] for r in self.store.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' "
            "AND tbl_name='observations'")}

    def test_the_composites_exist(self):
        self.assertLessEqual({"idx_obs_time_loc", "idx_obs_subj_time"}, self._indexes())

    def test_the_redundant_singles_are_gone(self):
        self.assertFalse({"idx_obs_time", "idx_obs_subject"} & self._indexes())

    def test_an_older_database_has_them_retired_on_open(self):
        path = Path(self._dir.name) / "old.db"
        old = Store(db_path=path)
        old.conn.execute("CREATE INDEX idx_obs_time ON observations(timestamp)")
        old.conn.execute("CREATE INDEX idx_obs_subject ON observations(subject_entity_id)")
        old.conn.commit()
        old.close()

        reopened = Store(db_path=path)
        self.addCleanup(reopened.close)
        names = {r["name"] for r in reopened.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' "
            "AND tbl_name='observations'")}
        self.assertFalse({"idx_obs_time", "idx_obs_subject"} & names)
        self.assertLessEqual({"idx_obs_time_loc", "idx_obs_subj_time"}, names)

    def test_a_windowed_zone_query_uses_the_composite(self):
        plan = self.store.conn.execute(
            "EXPLAIN QUERY PLAN SELECT * FROM observations "
            "WHERE timestamp>=? AND timestamp<=? AND location_id IN (?) "
            "ORDER BY timestamp, rowid", (T0, T0 + 60, self.bay)).fetchall()
        self.assertIn("idx_obs_time_loc", " ".join(r["detail"] for r in plan))


if __name__ == "__main__":       # pragma: no cover
    unittest.main()
