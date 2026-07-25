"""One Store, many threads — the shape run.py actually uses.

`run_pipeline` builds a single Store on the main thread and hands it to every
camera thread and to the shared resolvers (ObjectRegistry, IdentityResolver,
RuleEngine, Observer). SQLite connections are thread-bound by default, so that
arrangement raised `ProgrammingError: SQLite objects created in a thread can
only be used in that same thread` on the first detection and killed the capture
loop outright — frames=0, silently, in a thread nobody was watching.
"""
from __future__ import annotations

import sqlite3
import tempfile
import threading
import unittest
from pathlib import Path

from intelligence_os.tests import _stubs   # noqa: F401  (stubs the deps we lack)

from intelligence_os.store import Store    # noqa: E402


class TestStoreAcrossThreads(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = Store(db_path=Path(self.tmp.name) / "t.db")

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()

    def test_a_store_survives_the_thread_that_made_it(self):
        eid = self.store.create_entity("person", label="Dock worker")
        errors, seen = [], []

        start = threading.Barrier(4)

        def worker(n):
            try:
                start.wait()            # all four inside add_observation at once
                for i in range(25):
                    self.store.add_observation(eid, "present", timestamp=1000.0 + n * 100 + i)
                seen.append(len(self.store.observations(eid)))
            except Exception as e:      # the thread would otherwise die unnoticed
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(n,), name=f"cam-{n}")
                   for n in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(errors, [], "a camera thread could not touch the shared store")
        self.assertEqual(len(self.store.observations(eid)), 100)
        self.assertEqual(len(seen), 4)

    def test_the_module_is_serialized(self):
        """Sharing one handle is only safe because sqlite3 serializes access.
        If a build ever ships threadsafety < 3 this fails here, not at 3am."""
        self.assertEqual(sqlite3.threadsafety, 3)


if __name__ == "__main__":
    unittest.main()
