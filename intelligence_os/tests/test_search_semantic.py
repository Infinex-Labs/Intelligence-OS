"""Semantic retrieval, fusion and selection (search plan Phase 6).

Almost everything here runs with **no embedding model installed**, and that is
the design rather than a convenience. A stub encoder — twenty lines of
deterministic arithmetic — exercises the storage, the hard filters, the fusion,
the floor, the collapsing and the keyframe selection. What a real model adds is
one thing only: whether "loitering" is near "standing around, waiting". That is
a property of the model, and it is measured by the scorecard on a machine that
has one, not asserted here.

The tests that DO need the real dependency skip when it is absent, and say so.

Run: .venv/bin/python -m unittest intelligence_os.tests.test_search_semantic
"""
from __future__ import annotations

import os
import tempfile
import unittest

import numpy as np

from intelligence_os.tests import _stubs  # noqa: F401  (headless dep stubs)
from intelligence_os import semantic
from intelligence_os.ask import _collapse, _diverse_keyframes, _fuse, execute
from intelligence_os.config import CONFIG
from intelligence_os.store import Store

T0 = 1_780_408_800.0
STUB = "stub-encoder"


# --- the stub encoder --------------------------------------------------------
# One dimension per CONCEPT, not per word — which is the whole point. A
# bag-of-words stub would be a lexical index wearing a hat: it could never place
# "smoking" near "having a cigarette", so it could not exercise the one
# behaviour this layer exists for. Here several surface forms share a dimension,
# which is a crude imitation of what an embedding does and a sufficient one for
# testing the storage, the filters, the fusion and the floor.
#
# It is not a language model and claims nothing about one. Whether the REAL
# model places those words together is asserted in TestTheRealModel, and only
# on a machine that has one.
_CONCEPTS = {
    "smoking":    ("smok", "cigarett", "tobacco"),
    "waiting":    ("loiter", "linger", "wait", "standing", "hanging"),
    "phone":      ("phone", "call", "ringing"),
    "cycle":      ("bike", "bicycl", "cycl"),
    "gate":       ("gate",),
    "unattended": ("unattend", "unlock", "abandon"),
    "forklift":   ("forklift",),
    "parcel":     ("parcel", "packag", "deliver"),
    "animal":     ("dog", "cat", "animal"),
}
_DIMS = list(_CONCEPTS)


def stub_encode(texts):
    """`(n, len(_DIMS))` concept counts. Deliberately dumb and deterministic."""
    out = np.zeros((len(texts), len(_DIMS)), dtype=np.float32)
    for i, text in enumerate(texts):
        words = (text or "").lower().replace(",", " ").replace("—", " ").split()
        for j, concept in enumerate(_DIMS):
            stems = _CONCEPTS[concept]
            out[i, j] = sum(1.0 for w in words if w.startswith(stems))
    return out


class _SemanticBase(unittest.TestCase):
    """A small memory with prose on it, embedded by the stub."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = Store(os.path.join(self.tmp.name, "sem.db"))
        # Forced on regardless of INTELLIGENCE_OS_NO_SEMANTIC. That switch is a
        # deployment choice about whether to READ the index; this suite is what
        # proves the index works when it is read. The disabled behaviour has its
        # own tests below, which set the flag themselves.
        self._was_enabled = CONFIG.semantic.enabled
        CONFIG.semantic.enabled = True
        semantic.set_encoder(stub_encode, STUB)

        self.bay = self.store.upsert_location(
            "loading_bay", {"polygon": [[0, 0], [1, 0], [1, 1]]}, camera_id="cam_a")
        self.gate = self.store.upsert_location(
            "side_gate", {"polygon": [[2, 0], [3, 0], [3, 1]]}, camera_id="cam_b")
        self.dave = self.store.create_entity("person", label="Dave")
        self.nadia = self.store.create_entity("person", label="Nadia")

        self.rows = {}
        for key, subj, loc, cam, off, text in (
            ("smoke", self.dave, self.bay, "cam_a", 0, "having a cigarette"),
            ("wait", self.nadia, self.gate, "cam_b", 60, "standing around, waiting"),
            ("bike", self.nadia, self.gate, "cam_b", 120,
             "bicycle — unlocked, against the fence"),
        ):
            self.rows[key] = self.store.add_observation(
                subj, f"state:{text}", location_id=loc, camera_id=cam,
                timestamp=T0 + off, origin="vlm", text=text,
                source_ref=f"/kf/{key}.jpg")
        # A detector row: no prose, so nothing to embed. It is here to prove the
        # index does not grow with the rows that make a memory large.
        self.plain = self.store.add_observation(
            self.dave, "present", location_id=self.bay, camera_id="cam_a",
            timestamp=T0 + 30, origin="detector")
        semantic.backfill(self.store)

    def tearDown(self):
        semantic.set_encoder(None)
        CONFIG.semantic.enabled = self._was_enabled
        self.store.close()
        self.tmp.cleanup()

    def _run(self, **plan):
        return execute(self.store, {"start": None, "end": None, **plan})


class TestTheIndexIsBuilt(_SemanticBase):

    def test_only_prose_is_embedded(self):
        """The write path's volume is `present`; the search surface is not.

        A year of a busy site is millions of detector rows and thousands of
        described scenes. Embedding the former would spend the entire index on
        the word "present" — at exactly the scale where scanning it hurts.
        """
        held = self.store.embedded_count(self.store.EMBED_OBSERVATION, STUB)
        self.assertEqual(3, held)
        for _ids, mat in self.store.embedded_chunks(STUB):
            self.assertEqual(len(_DIMS), mat.shape[1])
        self.assertNotIn(self.plain, [oid for oid, _ in
                                      semantic.search(self.store, "cigarette",
                                                      min_similarity=0.0)])

    def test_a_second_pass_does_no_work(self):
        """The backlog is defined by absence, so re-running it is idempotent.

        It is a batch job that will be interrupted and re-run, and by the
        nightly pass every night forever. Without this it would add another
        copy of every vector each time and quietly weight those rows higher.
        """
        self.assertEqual(0, semantic.backfill(self.store))
        self.assertEqual(3, self.store.embedded_count(
            self.store.EMBED_OBSERVATION, STUB))

    def test_new_prose_is_picked_up_by_the_next_pass(self):
        oid = self.store.add_observation(
            self.dave, "state:on phone", location_id=self.bay, camera_id="cam_a",
            timestamp=T0 + 200, origin="vlm", text="on phone")
        # Not embedded on insert, on purpose: the observation write path runs per
        # frame per camera and must not carry a matrix multiply.
        self.assertEqual(3, self.store.embedded_count(
            self.store.EMBED_OBSERVATION, STUB))
        self.assertEqual(1, semantic.backfill(self.store))
        self.assertIn(oid, [o for o, _ in semantic.search(
            self.store, "phone call", min_similarity=0.0)])

    def test_re_embedding_replaces_rather_than_duplicates(self):
        """The one time a ref is re-embedded is when the encoder behind a name
        changed. Keeping both vectors would leave a row answering to a space
        nothing else is in."""
        vec = np.ones((1, len(_DIMS)), dtype=np.float32)
        self.store.add_embeddings(self.store.EMBED_OBSERVATION, STUB,
                                  [(self.rows["smoke"], vec[0])])
        self.assertEqual(3, self.store.embedded_count(
            self.store.EMBED_OBSERVATION, STUB))

    def test_two_models_do_not_share_an_index(self):
        """A cosine between two embedding spaces is a number with no meaning.

        So the model name is part of the key, and a search under one name can
        never read vectors written under another.
        """
        semantic.set_encoder(stub_encode, "other-model")
        self.assertEqual(0, self.store.embedded_count(
            self.store.EMBED_OBSERVATION, "other-model"))
        self.assertEqual([], semantic.search(self.store, "cigarette",
                                             min_similarity=0.0))
        self.assertEqual(3, semantic.backfill(self.store))
        self.assertEqual(3, self.store.embedded_count(
            self.store.EMBED_OBSERVATION, "other-model"))
        # ...and the first model's vectors are untouched.
        self.assertEqual(3, self.store.embedded_count(
            self.store.EMBED_OBSERVATION, STUB))

    def test_deleting_a_person_deletes_their_vectors(self):
        """§11 privacy removal. An embedding of "having a cigarette" is derived
        data about a person: leaving it behind would delete them by name and
        leave the sentence searchable by meaning."""
        counts = self.store.cascade_delete(self.dave)
        self.assertEqual(1, counts["embeddings"])
        self.assertEqual(2, self.store.embedded_count(
            self.store.EMBED_OBSERVATION, STUB))


class TestTheHardFiltersAreNotNegotiable(_SemanticBase):
    """A ranker may reorder what the filters allowed. It may not widen it.

    Every test here searches with the floor at zero, so a row that survives did
    so purely because the SQL let it through. That is the claim: the filters are
    in the scan, not applied to its output — where they would cost recall,
    because a ranked list is CUT at k.
    """

    def _ids(self, **filters):
        return [oid for oid, _s in semantic.search(
            self.store, "cigarette", min_similarity=0.0, **filters)]

    def test_the_zone_filter_reaches_the_semantic_scan(self):
        self.assertNotIn(self.rows["smoke"], self._ids(location_ids=[self.gate]),
                         "a semantic hit crossed a zone filter")
        self.assertIn(self.rows["smoke"], self._ids(location_ids=[self.bay]))

    def test_the_window_filter_reaches_the_semantic_scan(self):
        self.assertIn(self.rows["smoke"], self._ids(until=T0 + 10))
        self.assertNotIn(self.rows["smoke"], self._ids(since=T0 + 100))

    def test_the_camera_filter_reaches_the_semantic_scan(self):
        self.assertNotIn(self.rows["smoke"], self._ids(camera_ids=["cam_b"]))
        self.assertIn(self.rows["smoke"], self._ids(camera_ids=["cam_a"]))

    def test_the_subject_filter_reaches_the_semantic_scan(self):
        self.assertNotIn(self.rows["smoke"], self._ids(entity_ids=[self.nadia]))
        self.assertIn(self.rows["smoke"], self._ids(entity_ids=[self.dave]))

    def test_an_empty_filter_list_matches_nothing(self):
        """`None` is no filter, `[]` is a filter nothing satisfies — the same
        convention `_obs_where` and `search_text` use. All three read one plan,
        so a zone list that resolved to nothing must mean the same to each."""
        self.assertTrue(self._ids(location_ids=None))
        self.assertEqual([], self._ids(location_ids=[]))
        self.assertEqual([], self._ids(entity_ids=[]))
        self.assertEqual([], self._ids(camera_ids=[]))


class TestTheFloor(_SemanticBase):

    def test_nothing_similar_enough_is_not_an_answer(self):
        """The floor is what stops a meaning index inventing evidence.

        Without it every question returns the nearest sentence in memory,
        whether or not anything in memory answers it — and a nearest match
        presented as a hit is indistinguishable from a real one.
        """
        self.assertEqual([], semantic.search(self.store, "dog",
                                             min_similarity=0.3))

    def test_and_a_close_one_is(self):
        hits = semantic.search(self.store, "smoking a cigarette",
                               min_similarity=0.3)
        self.assertEqual(self.rows["smoke"], hits[0][0])

    def test_the_floor_is_absolute_not_relative(self):
        """bm25 orders and never gates, because its magnitude is corpus-relative.
        Cosine is an angle, so it can gate — and a floor that adapted to the best
        hit in the result would let a query with no good match through on the
        strength of its own worst option."""
        hits = semantic.search(self.store, "dog", min_similarity=0.0)
        self.assertTrue(hits, "the fixture has no candidates at all")
        self.assertTrue(all(score < 0.3 for _oid, score in hits))

    def test_results_are_ordered_strongest_first_and_stably(self):
        a = semantic.search(self.store, "bicycle", min_similarity=0.0)
        b = semantic.search(self.store, "bicycle", min_similarity=0.0)
        self.assertEqual(a, b, "two runs disagreed; RRF input must be stable")
        self.assertEqual(sorted(a, key=lambda r: -r[1]), a)


class TestFusion(unittest.TestCase):
    """RRF, on its own. No store, no encoder — it is arithmetic over ranks."""

    def test_one_list_alone_preserves_its_order(self):
        """The property that lets the fusion be unconditional.

        A deployment with no semantic index gets EXACTLY Phase 2's ordering,
        because 1/(k+rank) is monotone in rank. One code path, and no config
        under which the ranking silently becomes a different algorithm.
        """
        scores, reasons = _fuse(("lexical", ["a", "b", "c"]), ("semantic", []))
        self.assertEqual(["a", "b", "c"],
                         sorted(scores, key=lambda o: -scores[o]))
        self.assertEqual({"a": "lexical", "b": "lexical", "c": "lexical"}, reasons)

    def test_agreement_beats_a_single_confident_hit(self):
        """The whole reason to fuse. A row both indexes found is a better answer
        than one either found alone, and RRF says so without either score
        needing to be on the other's scale."""
        scores, reasons = _fuse(("lexical", ["solo", "shared"]),
                                ("semantic", ["other", "shared"]))
        self.assertEqual("shared", max(scores, key=lambda o: scores[o]))
        self.assertEqual("both", reasons["shared"])
        self.assertEqual("lexical", reasons["solo"])
        self.assertEqual("semantic", reasons["other"])

    def test_a_semantic_hit_cannot_evict_a_lexical_one(self):
        """Fusion may only ever ADD candidates. The rows bm25 found are all
        still there afterwards, whatever the other index thought of them."""
        scores, _ = _fuse(("lexical", ["a", "b"]), ("semantic", ["x", "y", "z"]))
        self.assertLessEqual({"a", "b"}, set(scores))

    def test_rank_not_score_is_what_is_fused(self):
        """bm25 is relative to a corpus and cosine is an angle: there is no
        conversion between them, so there is no calibration step to drift."""
        k = CONFIG.semantic.rrf_k
        scores, _ = _fuse(("lexical", ["a"]))
        self.assertAlmostEqual(1.0 / (k + 1), scores["a"])


class TestCollapsingNearDuplicates(unittest.TestCase):

    @staticmethod
    def _hit(pred, ts, score=1.0, reason="lexical"):
        return {"predicate": pred, "text": pred, "first_seen": ts,
                "last_seen": ts, "n": 1, "score": score,
                "match_reason": reason, "observation_id": f"o{ts}"}

    def test_a_scene_redescribed_every_few_seconds_is_one_hit(self):
        """One open gate becomes thirty rows. Listed individually they read as
        thirty events and crowd every other fact out of a limited answer."""
        hits = [self._hit("gate open", 100 + i * 10) for i in range(5)]
        out = _collapse(hits, 120.0)
        self.assertEqual(1, len(out))
        self.assertEqual(5, out[0]["n"])
        self.assertEqual((100, 140), (out[0]["first_seen"], out[0]["last_seen"]))

    def test_a_gap_wider_than_the_window_is_two_hits(self):
        """The gate opened, closed, and opened again — which is two events, and
        merging them would report a recurrence as a single occasion."""
        out = _collapse([self._hit("gate open", 100), self._hit("gate open", 400)],
                        120.0)
        self.assertEqual([1, 1], [h["n"] for h in out])

    def test_different_predicates_never_merge(self):
        out = _collapse([self._hit("gate open", 100), self._hit("on phone", 105)],
                        120.0)
        self.assertEqual(2, len(out))

    def test_a_run_keeps_its_best_evidence(self):
        """The run is one claim, so it is as well-evidenced as its best row —
        taking the first would rate a claim by the weakest thing that opened it."""
        out = _collapse([self._hit("gate open", 100, 0.2, "lexical"),
                         self._hit("gate open", 110, 0.9, "semantic")], 120.0)
        self.assertEqual(0.9, out[0]["score"])
        self.assertEqual("o110", out[0]["observation_id"])
        self.assertEqual("both", out[0]["match_reason"])


class TestKeyframesAreChosenForCoverage(unittest.TestCase):

    def test_the_endpoints_are_always_kept(self):
        """"When did this start" and "what did it look like by the end" are the
        two questions a strip of evidence is actually read for."""
        frames = [(float(i), f"/kf/{i}.jpg") for i in range(20)]
        picked = _diverse_keyframes(frames, 4)
        self.assertEqual("/kf/0.jpg", picked[0])
        self.assertEqual("/kf/19.jpg", picked[-1])

    def test_it_is_not_the_first_n(self):
        """Rows arrive in timestamp order, so the first four are four pictures
        of somebody walking in and nothing of what they then did."""
        frames = [(float(i), f"/kf/{i}.jpg") for i in range(20)]
        self.assertNotEqual([f"/kf/{i}.jpg" for i in range(4)],
                            _diverse_keyframes(frames, 4))

    def test_fewer_frames_than_asked_for_are_all_returned(self):
        frames = [(1.0, "/kf/a.jpg"), (2.0, "/kf/b.jpg")]
        self.assertEqual(["/kf/a.jpg", "/kf/b.jpg"], _diverse_keyframes(frames, 4))

    def test_it_is_deterministic(self):
        """A stored conversation turn replays the strip it showed at the time,
        so the same rows must always yield the same pictures."""
        frames = [(float(i), f"/kf/{i}.jpg") for i in range(17)]
        self.assertEqual(_diverse_keyframes(frames, 4),
                         _diverse_keyframes(frames, 4))

    def test_no_frames_is_not_a_crash(self):
        self.assertEqual([], _diverse_keyframes([], 4))


class TestTheAnswerSaysWhy(_SemanticBase):

    def test_a_row_found_by_meaning_alone_says_so(self):
        """The risk this layer carries is a hit nobody can explain. `smoking`
        shares no stem with `cigarette`, so only one index can have found it."""
        res = self._run(text="smoking")
        self.assertTrue(res["entities"])
        top = res["entities"][0]
        self.assertEqual("semantic", top["match_reason"])
        self.assertEqual("having a cigarette", top["hits"][0]["text"])

    def test_a_row_both_indexes_found_says_both(self):
        res = self._run(text="bicycle")
        self.assertEqual("both", res["entities"][0]["match_reason"])

    def test_the_trace_distinguishes_absent_from_unindexed(self):
        """"Nothing matched" and "the index that would have found it was never
        built" look identical from outside, and only one is fixed by rephrasing
        the question."""
        res = self._run(text="smoking")
        self.assertEqual(3, res["retrieval"]["semantic_index"])
        self.assertEqual(res["retrieval"]["lexical"] + res["retrieval"]["semantic"] > 0,
                         bool(res["entities"]))

    def test_an_unranked_question_carries_no_explanation(self):
        """A window over a period is a timeline, not a match. There is nothing
        to explain, and a null `match_reason` on every entity would be noise."""
        res = self._run()
        self.assertTrue(res["entities"])
        self.assertIsNone(res["entities"][0]["match_reason"])
        self.assertEqual([], res["entities"][0]["hits"])
        self.assertNotIn("retrieval", res)


class TestItDegradesRatherThanFails(_SemanticBase):

    def test_with_no_encoder_the_lexical_answer_still_stands(self):
        """The property the optional dependency rests on. Phase 2's behaviour
        was never wrong — only narrower — so losing the model must cost recall
        and nothing else."""
        semantic.set_encoder(None)
        CONFIG.semantic.enabled = False
        try:
            res = self._run(text="cigarette")
            self.assertTrue(res["entities"], "the lexical hit went missing")
            self.assertEqual("lexical", res["entities"][0]["match_reason"])
            self.assertEqual(0, res["retrieval"]["semantic"])
        finally:
            CONFIG.semantic.enabled = True
            semantic.set_encoder(stub_encode, STUB)

    def test_the_switch_leaves_the_index_alone(self):
        """Turning it off stops the vectors being READ, not being kept — so
        turning it back on costs nothing and needs no re-index."""
        CONFIG.semantic.enabled = False
        try:
            self.assertEqual([], semantic.search(self.store, "cigarette"))
        finally:
            CONFIG.semantic.enabled = True
        self.assertEqual(3, self.store.embedded_count(
            self.store.EMBED_OBSERVATION, STUB))

    def test_an_encoder_that_raises_is_not_a_failed_search(self):
        """A search is answerable without this layer. A model that throws must
        cost the semantic half of the answer, never the whole reply."""
        def broken(_texts):
            raise RuntimeError("model exploded")

        semantic.set_encoder(broken, STUB)
        try:
            self.assertEqual([], semantic.search(self.store, "cigarette"))
            res = self._run(text="cigarette")
            self.assertTrue(res["entities"])
        finally:
            semantic.set_encoder(stub_encode, STUB)

    def test_a_zero_vector_matches_nothing(self):
        """Text that carried no signal has no direction, so it has no similarity
        to anything — and must not become a division by zero on the query path."""
        vecs = semantic.encode(["qqqq zzzz"])          # no vocabulary term at all
        self.assertEqual(0.0, float(np.linalg.norm(vecs[0])))
        self.assertEqual([], semantic.search(self.store, "qqqq zzzz"))

    def test_the_query_cache_cannot_outlive_its_encoder(self):
        """It is keyed on the text alone, so a stale entry would answer the next
        question with the previous model's vector — in the one space where
        nothing can tell you that is what happened."""
        first = semantic._encode_query("cigarette")
        semantic.set_encoder(lambda t: np.ones((len(t), len(_DIMS)), np.float32),
                             "ones")
        try:
            second = semantic._encode_query("cigarette")
            self.assertFalse(np.array_equal(first, second))
        finally:
            semantic.set_encoder(stub_encode, STUB)


@unittest.skipUnless(semantic.available(),
                     "no local embedding model (pip install sentence-transformers)")
class TestTheRealModel(unittest.TestCase):
    """The one claim a stub cannot make: that different words mean the same thing.

    Skipped without the optional dependency, which is the point — these assert a
    property of a model, and a machine without one has no opinion to assert.
    """

    def test_different_vocabulary_is_reached(self):
        pairs = [("smoking", "having a cigarette"),
                 ("loitering", "standing around, waiting"),
                 ("taking a call", "on phone"),
                 ("bike left unattended", "bicycle — unlocked, against the fence")]
        vecs = semantic.encode([doc for _q, doc in pairs])
        for i, (query, doc) in enumerate(pairs):
            sim = float(semantic.encode([query])[0] @ vecs[i])
            self.assertGreaterEqual(
                sim, CONFIG.semantic.min_similarity,
                f"{query!r} no longer reaches {doc!r} ({sim:.3f}) — the model or "
                f"the floor moved; re-measure both before changing either")

    def test_something_absent_stays_below_the_floor(self):
        """The other half, and the harder one. A model that matches everything
        is as useless as one that matches nothing, and only this direction
        catches a floor that was lowered until the tests went green."""
        docs = ["having a cigarette", "standing around, waiting", "on phone",
                "bicycle — unlocked, against the fence", "carrying a parcel"]
        vecs = semantic.encode(docs)
        for absent in ("dog", "helicopter", "swimming pool"):
            best = float(max(semantic.encode([absent])[0] @ vecs.T))
            self.assertLess(best, CONFIG.semantic.min_similarity,
                            f"{absent!r} scores {best:.3f} against a memory that "
                            f"never saw one")

    def test_the_configured_model_is_the_one_being_scored(self):
        self.assertEqual(CONFIG.semantic.model, semantic.model_name())


if __name__ == "__main__":
    unittest.main()
