"""Visual re-ranking over keyframes (search plan Phase 8).

This suite is mostly about ONE claim, because the phase is mostly one claim:

    visual may reorder candidates another index retrieved.
    visual may never introduce one.

That is not a stylistic preference about layering. It is what the measurement in
`config.VisualConfig` forces. CLIP scores "a hospital bed" at 0.953 on a dim
room containing a dog and a sofa — higher than nine of the twelve things
actually in shot — so no threshold turns a CLIP score into the claim "this
memory contains that". A ranker that cannot detect absence is safe only where
something else has already established the row is a match, and the tests below
are the fence around that.

Everything here runs with **no CLIP model installed**. A stub encoder — a dozen
lines of arithmetic over made-up "images" — exercises the storage, the
de-duplication, the offset walk past pruned frames, the re-rank arithmetic and
the invariant. What a real model would add is whether a photograph of a dog is
near the word "dog", which is a property of the model and is measured in
docs/search-baseline.md rather than asserted here.

Run: .venv/bin/python -m unittest intelligence_os.tests.test_search_visual
"""
from __future__ import annotations

import os
import tempfile
import unittest

import numpy as np

from intelligence_os.tests import _stubs  # noqa: F401  (headless dep stubs)
from intelligence_os import visual
from intelligence_os.config import CONFIG, FRAMES_DIR
from intelligence_os.store import Store

T0 = 1_780_408_800.0
STUB = "stub-clip"

# One dimension per visible CONCEPT. "Images" are named by what they show, so a
# stub encoder can be a lookup: the file dog.jpg encodes to the dog axis, and
# the word "dog" encodes to the same axis. That is enough to test every piece of
# machinery here, and it claims nothing whatever about real CLIP.
_AXES = ["dog", "van", "laptop", "gate"]


def _axis(token: str) -> np.ndarray:
    v = np.zeros(len(_AXES), dtype=np.float32)
    for i, a in enumerate(_AXES):
        if a in token:
            v[i] = 1.0
    return v


def stub_images(paths):
    return np.stack([_axis(os.path.basename(p)) for p in paths])


def stub_texts(texts):
    return np.stack([_axis(t.lower()) for t in texts])


class _VisualBase(unittest.TestCase):
    """A memory whose rows cite pictures, and real files for them to cite.

    The files have to exist: `visual.backfill` goes through `retained_keyframe`,
    which asks the filesystem whether there is still a picture here. A fixture
    that faked that would skip the one branch retention actually exercises.
    """

    def setUp(self):
        self.store = Store(os.path.join(tempfile.mkdtemp(), "visual.db"))
        self._was = (CONFIG.visual.enabled, CONFIG.visual.weight,
                     CONFIG.visual.top_k)
        CONFIG.visual.enabled = True
        visual.set_encoder(stub_images, stub_texts, name=STUB)

        self.bay = self.store.upsert_location("loading_bay", {"polygon": []})
        self.dave = self.store.create_entity("person", label="Dave")
        self.frames = []
        for name in ("dog.jpg", "van.jpg", "laptop.jpg"):
            path = FRAMES_DIR / f"vistest_{name}"
            path.write_bytes(b"not really a jpeg, and nothing here opens it")
            self.frames.append(str(path))
        self.addCleanup(self._cleanup)

    def _cleanup(self):
        (CONFIG.visual.enabled, CONFIG.visual.weight,
         CONFIG.visual.top_k) = self._was
        visual.set_encoder(None)
        for p in self.frames:
            try:
                os.unlink(p)
            except OSError:
                pass
        self.store.close()

    def _obs(self, predicate, ts, ref, text=None):
        return self.store.add_observation(
            self.dave, predicate, location_id=self.bay, timestamp=ts,
            source_ref=ref, origin="vlm", text=text or predicate)


# --- the invariant, which is the phase ---------------------------------------

class TestItReordersAndNeverIntroduces(unittest.TestCase):
    """`rerank` is a permutation of its input. Nothing enters, nothing leaves."""

    def setUp(self):
        self.store = Store(os.path.join(tempfile.mkdtemp(), "v.db"))
        self._was = CONFIG.visual.enabled
        CONFIG.visual.enabled = True
        visual.set_encoder(stub_images, stub_texts, name=STUB)
        self.addCleanup(self._restore)

    def _restore(self):
        CONFIG.visual.enabled = self._was
        visual.set_encoder(None)
        self.store.close()

    def _seed(self):
        """Three rows with vectors, and one the visual index has never seen."""
        loc = self.store.upsert_location("bay", {"polygon": []})
        e = self.store.create_entity("person", label="Dave")
        ids = {}
        for tag in ("dog", "van", "laptop"):
            ids[tag] = self.store.add_observation(
                e, f"saw a {tag}", location_id=loc, timestamp=T0, text=f"a {tag}")
        vecs = stub_images([f"{t}.jpg" for t in ("dog", "van", "laptop")])
        self.store.add_embeddings(
            self.store.EMBED_KEYFRAME, STUB,
            [(ids[t], v) for t, v in zip(("dog", "van", "laptop"), vecs)])
        ids["unseen"] = self.store.add_observation(
            e, "no picture", location_id=loc, timestamp=T0, text="no picture")
        return ids

    def test_the_id_set_is_unchanged(self):
        ids = self._seed()
        scores = {oid: 0.01 for oid in ids.values()}
        out = visual.rerank(self.store, "dog", scores)
        self.assertEqual(set(scores), set(out),
                         "re-ranking may permute the list, never resize it")

    def test_a_row_with_no_keyframe_vector_survives_re_ranking(self):
        # The one that would be quietly dropped by an implementation that
        # rebuilt the dict from the vectors it found instead of adjusting it.
        ids = self._seed()
        scores = {oid: 0.01 for oid in ids.values()}
        out = visual.rerank(self.store, "dog", scores)
        self.assertIn(ids["unseen"], out)
        self.assertEqual(0.01, out[ids["unseen"]],
                         "nothing to say about it means say nothing, not demote it")

    def test_the_matching_picture_is_promoted_above_its_equals(self):
        ids = self._seed()
        scores = {oid: 0.01 for oid in ids.values()}
        out = visual.rerank(self.store, "dog", scores)
        best = max(out, key=lambda k: out[k])
        self.assertEqual(ids["dog"], best)

    def test_it_cannot_outvote_two_indexes_that_agree(self):
        """The weight is 0.5 of one list's contribution, and that is deliberate.

        A row both other indexes found sits at 2/(k+rank). A visual boost is at
        most weight/(k+1). If a picture could overturn that, CLIP would be
        deciding answers rather than ordering them — and it is the component
        here least able to say when it is wrong.
        """
        ids = self._seed()
        k = CONFIG.semantic.rrf_k
        scores = {ids["van"]: 2.0 / (k + 1),        # found by lexical AND semantic
                  ids["dog"]: 1.0 / (k + 2)}        # found by one, and pictured
        out = visual.rerank(self.store, "dog", scores)
        self.assertEqual(ids["van"], max(out, key=lambda x: out[x]),
                         "a picture may break a tie, not overturn a consensus")

    def test_an_empty_result_stays_empty(self):
        """Phase 7's ladder can only report a dead end that is allowed to exist.

        This is the assertion that keeps Phase 8 from repealing Phase 7. An
        index that returned its nearest frame regardless would make every
        question non-empty, and "I could not find it" would once again be
        indistinguishable from "it did not happen".
        """
        self._seed()
        self.assertEqual({}, visual.rerank(self.store, "dog", {}))


# --- it is off unless asked for ----------------------------------------------

class TestItIsOffByDefault(_VisualBase):

    def test_disabled_is_the_identity_function(self):
        self._obs("a dog", T0, self.frames[0])
        visual.backfill(self.store)
        scores = {r["observation_id"]: 0.5 for r in self.store.conn.execute(
            "SELECT observation_id FROM observations")}
        CONFIG.visual.enabled = False
        self.assertEqual(scores, visual.rerank(self.store, "dog", scores))

    def test_disabled_indexes_nothing(self):
        self._obs("a dog", T0, self.frames[0])
        CONFIG.visual.enabled = False
        self.assertEqual(0, visual.backfill(self.store))
        self.assertEqual(0, self.store.embedded_count(
            self.store.EMBED_KEYFRAME, STUB))

    def test_an_unbuilt_index_is_not_an_error(self):
        # No backfill has run. A question still answers, in the order the other
        # indexes put it in.
        oid = self._obs("a dog", T0, self.frames[0])
        self.assertEqual({oid: 0.5}, visual.rerank(self.store, "dog", {oid: 0.5}))


# --- building the index ------------------------------------------------------

class TestTheBackfill(_VisualBase):

    def test_it_embeds_rows_that_cite_a_picture(self):
        oid = self._obs("a dog", T0, self.frames[0])
        self.assertEqual(1, visual.backfill(self.store))
        ids, mat = self.store.keyframe_vectors(STUB, [oid])
        self.assertEqual([oid], ids)
        self.assertEqual((1, len(_AXES)), mat.shape)

    def test_it_is_idempotent(self):
        self._obs("a dog", T0, self.frames[0])
        self.assertEqual(1, visual.backfill(self.store))
        self.assertEqual(0, visual.backfill(self.store),
                         "the backlog is defined by absence, so a done row is done")

    def test_one_picture_cited_six_times_is_encoded_once(self):
        """The de-duplication, asserted on the ENCODER rather than on the output.

        Counting rows written would pass whether or not the file was decoded six
        times, and the cost this avoids is the decode.
        """
        calls = []

        def counting(paths):
            calls.append(list(paths))
            return stub_images(paths)

        visual.set_encoder(counting, stub_texts, name=STUB)
        for i in range(6):
            self._obs("a dog", T0 + i, self.frames[0])
        self.assertEqual(6, visual.backfill(self.store))
        self.assertEqual([[self.frames[0]]], calls,
                         "six rows, one image, one forward pass")

    def test_a_row_with_no_picture_is_not_in_the_backlog(self):
        self.store.add_observation(self.dave, "present", location_id=self.bay,
                                   timestamp=T0)
        self.assertEqual(0, visual.backfill(self.store))

    def test_it_steps_past_pictures_retention_has_deleted(self):
        """The offset walk. Without it this loops on the dead batch forever.

        A row whose JPEG was pruned can never be embedded, but it stays in the
        backlog — which is defined by "has no vector" and cannot see the
        filesystem. The newest-first walk has to be able to get past it to reach
        the older rows that DO still have pictures.
        """
        self._obs("a dog", T0 + 100, "/gone/deleted_by_retention.jpg")
        alive = self._obs("a van", T0, self.frames[1])
        CONFIG.visual.batch_size = 1          # force the dead row into its own batch
        self.addCleanup(setattr, CONFIG.visual, "batch_size", 32)
        self.assertEqual(1, visual.backfill(self.store))
        ids, _ = self.store.keyframe_vectors(STUB, [alive])
        self.assertEqual([alive], ids, "the older, still-present frame was reached")

    def test_it_respects_a_limit(self):
        for i, f in enumerate(self.frames):
            self._obs(f"pic {i}", T0 + i, f)
        self.assertEqual(2, visual.backfill(self.store, limit=2))


# --- the two spaces must not touch -------------------------------------------

class TestTheSpacesStayApart(_VisualBase):
    """CLIP vectors and MiniLM vectors share a table and nothing else.

    They are stored under different `kind`s AND different model names, and the
    query path encodes with the matching tower. Getting this wrong would not
    raise — it would return confidently ordered nonsense.
    """

    def test_keyframe_vectors_are_a_different_kind(self):
        oid = self._obs("a dog", T0, self.frames[0])
        visual.backfill(self.store)
        self.assertEqual(1, self.store.embedded_count(
            self.store.EMBED_KEYFRAME, STUB))
        self.assertEqual(0, self.store.embedded_count(
            self.store.EMBED_OBSERVATION, STUB),
            "a keyframe vector must never be read as a prose vector")
        self.assertEqual(([], None),
                         self.store.keyframe_vectors("some-other-model", [oid]))

    def test_a_dimension_mismatch_is_refused_rather_than_scored(self):
        """Two models under one name is the corruption `model` exists to stop.

        If it happens anyway, the re-rank declines instead of dotting vectors
        from different spaces — which would produce an order, and no way to tell
        it was meaningless.
        """
        oid = self._obs("a dog", T0, self.frames[0])
        visual.backfill(self.store)
        visual.set_encoder(stub_images,
                           lambda ts: np.ones((len(ts), len(_AXES) + 3),
                                              dtype=np.float32),
                           name=STUB)
        scores = {oid: 0.5}
        self.assertEqual(scores, visual.rerank(self.store, "dog", scores))

    def test_dropping_one_index_leaves_the_other(self):
        oid = self._obs("a dog", T0, self.frames[0], text="a dog")
        visual.backfill(self.store)
        self.store.add_embeddings(self.store.EMBED_OBSERVATION, "minilm-ish",
                                  [(oid, np.ones(4, dtype=np.float32))])
        self.store.drop_embeddings(kind=self.store.EMBED_KEYFRAME)
        self.assertEqual(0, self.store.embedded_count(
            self.store.EMBED_KEYFRAME, STUB))
        self.assertEqual(1, self.store.embedded_count(
            self.store.EMBED_OBSERVATION, "minilm-ish"))


# --- what the trace says ------------------------------------------------------

class TestTheTraceIsHonest(_VisualBase):

    def test_the_retrieval_summary_reports_the_index_and_the_re_rank(self):
        from intelligence_os.ask import execute
        self._obs("a dog in the bay", T0, self.frames[0], text="a dog in the bay")
        visual.backfill(self.store)
        r = execute(self.store, {"text": "dog", "start": None, "end": None})
        tr = r.get("retrieval")
        self.assertIsNotNone(tr, "a word question records how it retrieved")
        self.assertEqual(1, tr["visual_index"])
        self.assertGreaterEqual(tr["reranked"], 1)

    def test_an_unbuilt_index_reads_as_zero_not_as_absent(self):
        """"Nothing was re-ranked" and "there is no index" are different
        problems with different fixes, and the trace has to tell them apart —
        the second is repaired by running a backfill, the first by nothing the
        asker can type."""
        from intelligence_os.ask import execute
        self._obs("a dog in the bay", T0, self.frames[0], text="a dog in the bay")
        r = execute(self.store, {"text": "dog", "start": None, "end": None})
        self.assertEqual(0, r["retrieval"]["visual_index"])


if __name__ == "__main__":
    unittest.main()
