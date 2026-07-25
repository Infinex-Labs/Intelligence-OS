"""V4-M5b: /api/graph says what a thing *is*; the client decides how it looks.

The pane styles nodes off the theme's CSS vars (FR-NG-7), so a server that also
ships `shape`/`image` would silently win — and the old `image` was a call out to
ui-avatars.com, which a local surveillance box has no business making.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from intelligence_os.tests import _stubs   # noqa: F401  (stubs the deps we lack)

from intelligence_os import web                      # noqa: E402
from intelligence_os.store import Store              # noqa: E402


class TestGraphPayload(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        db = Path(self.tmp.name) / "t.db"
        s = Store(db_path=db)
        self.person = s.create_entity("person", label="Kalki")
        self.thing = s.create_entity("object", label="bike")
        # reinforce until the edge is 'confirmed' — serve_graph ships those only
        for _ in range(20):
            s.reinforce_relation("relation", self.person, "uses",
                                 object_entity_id=self.thing)
        s.close()
        self._real = web.Store
        web.Store = lambda *a, **k: self._real(db_path=db)

    def tearDown(self):
        web.Store = self._real
        self.tmp.cleanup()

    def _graph(self):
        h = object.__new__(web.RequestHandler)
        h.sent = []
        h.send_json = h.sent.append
        h.path = "/api/graph"
        h.serve_graph()
        return h.sent[0]

    def test_nodes_carry_type_and_first_seen_only(self):
        nodes = {n["id"]: n for n in self._graph()["nodes"]}
        self.assertEqual(nodes[self.person]["group"], "person")
        self.assertEqual(nodes[self.thing]["group"], "object")
        for n in nodes.values():
            self.assertIsNotNone(n["t"], "scrub/first-seen needs a timestamp")

    def test_no_presentation_and_no_outbound_urls(self):
        for n in self._graph()["nodes"]:
            self.assertNotIn("shape", n, "shape is the client's decision")
            self.assertNotIn("image", n, "no third-party avatar fetch")

    def test_confirmed_relation_becomes_an_edge(self):
        edges = self._graph()["edges"]
        self.assertEqual(len(edges), 1)
        self.assertEqual((edges[0]["from"], edges[0]["to"], edges[0]["label"]),
                         (self.person, self.thing, "uses"))


if __name__ == "__main__":
    unittest.main()
