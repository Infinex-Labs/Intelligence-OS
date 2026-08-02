"""The lexical index (search plan Phase 2).

Four things are under test, and they are separable on purpose:
  1. the index exists, stems, and stays in step with the base table
  2. a user's words are data, never FTS5 syntax
  3. a database written before the index existed is backfilled, once
  4. everything degrades to substring matching when FTS5 is absent

What is NOT tested here is whether a question gets a better answer — that is
scored end to end in `test_search_quality.py` against the recorded baseline.
This file is about the machinery underneath it.
"""
from __future__ import annotations

import sqlite3
import tempfile
import time
import unittest
from pathlib import Path

from intelligence_os.tests import _stubs   # noqa: F401  (stubs the deps we lack)

from intelligence_os.ask import execute    # noqa: E402
from intelligence_os.store import Store    # noqa: E402

NOW = time.time()


class TestFts5IsAvailable(unittest.TestCase):
    """Mirrors the thread-safety assertion in test_store_threads: a build-time
    assumption that is invisible until it is wrong."""

    def test_this_sqlite_has_fts5(self):
        con = sqlite3.connect(":memory:")
        try:
            con.execute("CREATE VIRTUAL TABLE t USING fts5(x)")
        except sqlite3.OperationalError:      # pragma: no cover - not our builds
            self.fail("SQLite built without FTS5; search degrades to substrings")
        finally:
            con.close()

    def test_the_store_reports_it(self):
        with tempfile.TemporaryDirectory() as d:
            s = Store(db_path=Path(d) / "t.db")
            self.assertTrue(s.fts_enabled)
            s.close()


class TestIndexing(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = Store(db_path=Path(self.tmp.name) / "t.db")
        self.subject = self.store.create_entity("person", label="Dave")

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()

    def _write(self, text, **kw):
        return self.store.add_observation(
            self.subject, "state:" + text, origin="vlm", text=text,
            timestamp=kw.pop("timestamp", NOW), **kw)

    def _ids(self, query, **kw):
        return {r["observation_id"] for r in self.store.search_text(query, **kw)}

    def test_a_new_row_is_searchable_immediately(self):
        oid = self._write("having a cigarette")
        self.assertEqual(self._ids("cigarette"), {oid})

    def test_stemming_is_the_point(self):
        """A substring test can never get from 'cigarettes' to 'cigarette'."""
        oid = self._write("having a cigarette")
        self.assertEqual(self._ids("cigarettes"), {oid})
        self.assertEqual(self._ids("smoked"), set(), "stems, not synonyms")

    def test_word_order_does_not_matter(self):
        oid = self._write("aisle light flickering")
        self.assertEqual(self._ids("flickering light"), {oid})

    def test_stopwords_are_dropped_not_required(self):
        """Stored 'on phone', asked 'on the phone'. Requiring 'the' would fail on
        the one word in the question carrying no meaning."""
        oid = self._write("on phone")
        self.assertEqual(self._ids("on the phone"), {oid})

    def test_all_terms_must_match(self):
        """The union of terms would return the whole table for any question."""
        self._write("gate — open, unlatched")
        self._write("forklift — parked, forks raised")
        self.assertEqual(len(self._ids("gate open")), 1)

    def test_absent_words_return_nothing(self):
        """The true-negative guard: no answer is an answer."""
        self._write("having a cigarette")
        self.assertEqual(self._ids("dog"), set())

    def test_ranking_is_returned_and_ordered(self):
        self._write("gate")
        self._write("gate gate gate open unlatched")
        rows = self.store.search_text("gate")
        self.assertEqual(len(rows), 2)
        self.assertGreaterEqual(rows[0]["score"], rows[1]["score"],
                                "bm25 is negated so bigger means more relevant")

    def test_filters_apply_in_the_same_statement(self):
        old = self._write("having a cigarette", timestamp=NOW - 86400)
        new = self._write("having a cigarette", timestamp=NOW)
        self.assertEqual(self._ids("cigarette", since=NOW - 60), {new})
        self.assertEqual(self._ids("cigarette", until=NOW - 60), {old})

    def test_location_filter(self):
        loc = self.store.upsert_location("aisle_3", {"polygon": [[0, 0]]})
        here = self._write("spill in aisle_3", location_id=loc)
        self._write("spill somewhere else")
        self.assertEqual(self._ids("spill", location_ids=[loc]), {here})

    def test_deleting_a_row_removes_it_from_the_index(self):
        """External-content FTS keeps no copy of the text, but it does keep the
        postings — a stale posting would resurrect a deleted observation."""
        oid = self._write("having a cigarette")
        with self.store.tx() as c:
            c.execute("DELETE FROM observations WHERE observation_id=?", (oid,))
        self.assertEqual(self._ids("cigarette"), set())

    def test_updating_a_row_reindexes_it(self):
        oid = self._write("gate open")
        with self.store.tx() as c:
            c.execute("UPDATE observations SET text='gate closed' WHERE observation_id=?",
                      (oid,))
        self.assertEqual(self._ids("closed"), {oid})
        self.assertEqual(self._ids("open"), set())

    def test_rows_without_text_are_simply_absent(self):
        """Detector rows carry no prose. They must not break the index, and they
        must not become searchable by accident either."""
        self.store.add_observation(self.subject, "present", origin="detector")
        self.assertEqual(self._ids("present"), set())

    def test_descriptions_are_indexed_too(self):
        did = self.store.add_scene_description(
            model="test", raw={"a": 1}, text="the gate is open and unlatched")
        hits = self.store.search_descriptions("gates")
        self.assertEqual([r["description_id"] for r in hits], [did])


class TestUserWordsAreNeverSyntax(unittest.TestCase):
    """`fts_query` is the boundary. Everything past it is FTS5 grammar, so a
    question containing a quote or the word OR must not reach it intact."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = Store(db_path=Path(self.tmp.name) / "t.db")
        self.subject = self.store.create_entity("person", label="Dave")
        self.store.add_observation(self.subject, "state:x", origin="vlm",
                                   text="gate open", timestamp=NOW)

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()

    def test_quotes_and_operators_do_not_raise(self):
        for hostile in ['gate "open', "gate*", "gate OR forklift", "gate NEAR/2",
                        "(gate", "gate^2", '"', "-gate", "gate:open", "*"]:
            with self.subTest(q=hostile):
                self.store.search_text(hostile)      # must not raise

    def test_or_is_a_word_not_an_operator(self):
        """'gate or forklift' as an operator returns both. As words it returns
        neither, because no row says 'forklift'. The narrow reading is the honest
        one — the user typed English, not a query language."""
        self.store.add_observation(self.subject, "state:y", origin="vlm",
                                   text="forklift parked", timestamp=NOW)
        self.assertEqual(self.store.search_text("gate or forklift"), [])

    def test_an_empty_query_is_not_a_match_all(self):
        self.assertIsNone(Store.fts_query("   "))
        self.assertIsNone(Store.fts_query("!!!"))
        self.assertEqual(self.store.search_text(""), [])

    def test_a_query_of_only_stopwords_keeps_them(self):
        """Dropping every term would leave a match-everything query."""
        self.assertEqual(Store.fts_query("the a of"), '"the" "a" "of"')


class TestBackfill(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "old.db"

    def tearDown(self):
        self.tmp.cleanup()

    def _pre_phase2_db(self) -> str:
        """A database that has rows but no index — what every existing install is
        on the first run after this ships."""
        store = Store(db_path=self.path)
        eid = store.create_entity("person", label="Dave")
        oid = store.add_observation(eid, "state:x", origin="vlm",
                                    text="having a cigarette", timestamp=NOW)
        for fts, _b, _c in Store._FTS_SPECS:
            store.conn.execute(f"DROP TABLE {fts}")
            for suffix in ("ai", "ad", "au"):
                store.conn.execute(f"DROP TRIGGER {fts}_{suffix}")
        store.conn.commit()
        store.close()
        return oid

    def test_an_existing_database_is_backfilled_on_open(self):
        oid = self._pre_phase2_db()
        store = Store(db_path=self.path)
        try:
            self.assertEqual([r["observation_id"] for r in
                              store.search_text("cigarettes")], [oid])
        finally:
            store.close()

    def test_reopening_does_not_rebuild_again(self):
        """`rebuild` re-tokenises the whole table. On a memory with a year of
        footage that is minutes of startup, every startup."""
        self._pre_phase2_db()
        Store(db_path=self.path).close()          # first open: backfills
        store = Store(db_path=self.path)
        try:
            seen: list[str] = []
            store.conn.set_trace_callback(seen.append)
            store._migrate_v6_fts()
            store.conn.set_trace_callback(None)
            self.assertFalse([s for s in seen if "rebuild" in s], seen)
        finally:
            store.close()

    def test_reindex_text_is_available_when_the_triggers_were_bypassed(self):
        oid = self._pre_phase2_db()
        store = Store(db_path=self.path)
        try:
            with store.tx() as c:                 # rewrite behind the index
                c.execute("PRAGMA writable_schema=OFF")
                c.execute("UPDATE observations SET text='carrying a parcel' "
                          "WHERE observation_id=?", (oid,))
            store.reindex_text()
            self.assertEqual([r["observation_id"] for r in
                              store.search_text("parcels")], [oid])
        finally:
            store.close()


class TestDegradesWithoutFts(unittest.TestCase):
    """Invariant 5 of the plan: a missing optional capability degrades, never
    crashes. FTS5 is compiled into the builds we ship on, but a source build with
    -DSQLITE_OMIT_FTS5 would otherwise turn every question into a 500."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = Store(db_path=Path(self.tmp.name) / "t.db")
        self.dave = self.store.create_entity("person", label="Dave")
        self.store.add_observation(self.dave, "state:having a cigarette",
                                   origin="vlm", text="having a cigarette",
                                   timestamp=NOW)
        self.store.fts_enabled = False

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()

    def test_search_text_returns_nothing_rather_than_raising(self):
        self.assertEqual(self.store.search_text("cigarette"), [])
        self.assertEqual(self.store.search_descriptions("cigarette"), [])

    def test_substring_matching_still_answers(self):
        """The pre-Phase-2 behaviour is the floor, and it is still there."""
        res = execute(self.store, {"start": None, "end": None,
                                   "predicate_contains": "cigarette"})
        self.assertEqual([e["label"] for e in res["entities"]], ["Dave"])


class TestPredicateContainsIsAUnion(unittest.TestCase):
    """The older field keeps its substring behaviour and gains the index. A phase
    that widens recall must never take an existing hit away."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = Store(db_path=Path(self.tmp.name) / "t.db")
        self.dave = self.store.create_entity("person", label="Dave")
        self.store.add_observation(self.dave, "rule_fired:loiter_at_bay",
                                   origin="rule", timestamp=NOW)
        self.store.add_observation(self.dave, "state:having a cigarette",
                                   origin="vlm", text="having a cigarette",
                                   timestamp=NOW)

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()

    def _run(self, **plan):
        return execute(self.store, {"start": None, "end": None, **plan})

    def test_a_predicate_with_no_text_is_still_matched_by_substring(self):
        """Rule rows carry no prose, so the index cannot see them at all."""
        res = self._run(predicate_contains="rule_fired")
        self.assertEqual([ev["rule"] for e in res["entities"]
                          for ev in e["rule_events"]], ["loiter_at_bay"])

    def test_and_a_stemmed_word_is_matched_by_the_index(self):
        self.assertTrue(self._run(predicate_contains="cigarettes")["entities"])

    def test_text_is_strict_where_predicate_contains_is_not(self):
        """`text` is the search surface: match, or you are not a result."""
        self.assertFalse(self._run(text="rule_fired")["entities"])
        self.assertTrue(self._run(predicate_contains="rule_fired")["entities"])

    def test_word_questions_come_back_ranked(self):
        res = self._run(predicate_contains="cigarettes")
        self.assertTrue(res["ranked"])
        self.assertIsNotNone(res["entities"][0]["match_score"])

    def test_a_substring_only_hit_carries_no_score(self):
        """bm25 never saw it. Reporting 0.0 would rank it worst rather than
        unranked, and the two are not the same claim."""
        res = self._run(predicate_contains="rule_fired")
        self.assertIsNone(res["entities"][0]["match_score"])

    def test_a_window_question_stays_a_timeline(self):
        res = self._run()
        self.assertFalse(res["ranked"], "ordering by relevance to nothing")


if __name__ == "__main__":
    unittest.main()
