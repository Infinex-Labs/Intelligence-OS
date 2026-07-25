"""V4-M10: the assistant's threads must outlive the page.

Pins the two halves of that: the store (threads, turns, titles, cascade,
KPI aggregate) and the /api/chats handlers the sidebar reads.
"""
from __future__ import annotations

import json
import os
import tempfile
import unittest

from intelligence_os.tests import _stubs   # noqa: F401  (stubs the deps we lack)

from intelligence_os.store import Store    # noqa: E402
from intelligence_os import web            # noqa: E402


class ChatBase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = Store(os.path.join(self.tmp.name, "chat.db"))

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()


class TestChatStore(ChatBase):
    def test_thread_takes_its_title_from_the_first_question(self):
        cid = self.store.create_conversation()
        self.assertEqual(self.store.get_conversation(cid)["title"], "New chat")
        self.store.add_chat_turn(cid, "Who was in the loading bay yesterday?", "Two people.")
        self.assertEqual(self.store.get_conversation(cid)["title"],
                         "Who was in the loading bay yesterday?")
        # ...and a later question does not rename the thread out from under you
        self.store.add_chat_turn(cid, "What about the aisle?", "Nobody.")
        self.assertEqual(self.store.get_conversation(cid)["title"],
                         "Who was in the loading bay yesterday?")

    def test_long_title_is_cut_at_a_word_boundary(self):
        t = Store.title_from_question("who " * 40)
        self.assertLessEqual(len(t), 59)
        self.assertTrue(t.endswith("…"), t)
        self.assertNotIn("wh…", t)          # never mid-word

    def test_turns_come_back_in_order_with_their_payload(self):
        cid = self.store.create_conversation()
        for i in range(3):
            self.store.add_chat_turn(cid, f"q{i}", f"a{i}", json.dumps({"n": i}))
        turns = self.store.conversation_turns(cid)
        self.assertEqual([t["question"] for t in turns], ["q0", "q1", "q2"])
        self.assertEqual(json.loads(turns[1]["payload"]), {"n": 1})

    def test_list_is_newest_activity_first_and_counts_turns(self):
        a = self.store.create_conversation()
        b = self.store.create_conversation()
        self.store.add_chat_turn(a, "older", "x")
        self.store.add_chat_turn(b, "newer", "x")
        self.store.add_chat_turn(b, "newer still", "x")
        rows = self.store.conversations()
        self.assertEqual([r["conversation_id"] for r in rows], [b, a])
        self.assertEqual([r["n_turns"] for r in rows], [2, 1])

    def test_list_is_scoped_to_the_operator_who_asked(self):
        mine = self.store.create_conversation(user_id="usr_a")
        self.store.create_conversation(user_id="usr_b")
        self.assertEqual([r["conversation_id"] for r in self.store.conversations("usr_a")],
                         [mine])
        self.assertEqual(len(self.store.conversations()), 2)   # None = every thread

    def test_deleting_a_thread_takes_its_turns(self):
        cid = self.store.create_conversation()
        self.store.add_chat_turn(cid, "q", "a")
        self.assertTrue(self.store.delete_conversation(cid))
        self.assertIsNone(self.store.get_conversation(cid))
        self.assertEqual(self.store.conversation_turns(cid), [])
        self.assertFalse(self.store.delete_conversation(cid))   # idempotent, not a crash

    def test_answering_a_deleted_thread_is_refused_not_orphaned(self):
        cid = self.store.create_conversation()
        self.store.delete_conversation(cid)
        self.assertIsNone(self.store.add_chat_turn(cid, "q", "a"))

    def test_rename_rejects_blank_and_trims(self):
        cid = self.store.create_conversation()
        self.assertFalse(self.store.rename_conversation(cid, "   "))
        self.assertTrue(self.store.rename_conversation(cid, "  Night   shift  "))
        self.assertEqual(self.store.get_conversation(cid)["title"], "Night shift")

    def test_kpis_are_sums_over_stored_turns(self):
        a = self.store.create_conversation(user_id="usr_a")
        self.store.add_chat_turn(a, "q1", "a1", None, 800,
                                 counts={"entities": 2, "observations": 14,
                                         "keyframes": 3, "rule_events": 1})
        self.store.add_chat_turn(a, "q2", "a2", None, 1200,
                                 counts={"entities": 1, "observations": 6,
                                         "keyframes": 2, "rule_events": 0})
        self.store.create_conversation(user_id="usr_b")      # another operator's thread
        s = self.store.chat_stats("usr_a")
        self.assertEqual((s["threads"], s["questions"]), (1, 2))
        self.assertEqual((s["entities"], s["observations"]), (3, 20))
        self.assertEqual((s["keyframes"], s["rule_events"]), (5, 1))
        self.assertEqual(s["avg_latency_ms"], 1000)

    def test_kpis_on_an_empty_store_are_zero_not_null(self):
        s = self.store.chat_stats()
        self.assertEqual((s["threads"], s["questions"]), (0, 0))
        self.assertEqual(s["entities"], 0)
        self.assertIsNone(s["avg_latency_ms"])
        self.assertIsNone(s["last_asked"])


class TestChatRoutes(ChatBase):
    """The handlers, driven directly — same shape as test_camera_config."""

    def _handler(self, body=None, user="usr_a"):
        h = object.__new__(web.RequestHandler)
        h.sent, h.errors = [], []
        h.send_json = h.sent.append
        h.send_error = lambda code, msg=None: h.errors.append((code, msg))
        h.send_error_json = lambda code, msg=None: h.errors.append((code, msg))
        h._read_json = lambda: (body or {})
        h.current_user_id = user
        return h

    def setUp(self):
        super().setUp()
        # the handlers open their own Store(); point it at the temp db
        self._real_env = os.environ.get("INTELLIGENCE_OS_DB")
        os.environ["INTELLIGENCE_OS_DB"] = self.store.db_path

    def tearDown(self):
        if self._real_env is None:
            os.environ.pop("INTELLIGENCE_OS_DB", None)
        else:
            os.environ["INTELLIGENCE_OS_DB"] = self._real_env
        super().tearDown()

    def test_create_then_list_then_open(self):
        h = self._handler({"title": "Night shift"})
        h.serve_chat_create()
        cid = h.sent[0]["conversation"]["conversation_id"]

        h = self._handler()
        h.serve_chats()
        self.assertEqual([c["title"] for c in h.sent[0]["conversations"]], ["Night shift"])
        self.assertEqual(h.sent[0]["stats"]["threads"], 1)

        self.store.add_chat_turn(cid, "who was here?", "Nobody.",
                                 json.dumps({"total_observations": 0}))
        h = self._handler()
        h.serve_chat(cid)
        turns = h.sent[0]["turns"]
        self.assertEqual(turns[0]["question"], "who was here?")
        self.assertEqual(turns[0]["payload"], {"total_observations": 0})   # parsed, not a string

    def test_another_operators_thread_is_not_readable(self):
        cid = self.store.create_conversation(user_id="usr_b")
        h = self._handler(user="usr_a")
        h.serve_chat(cid)
        self.assertEqual(h.sent, [])
        self.assertEqual(h.errors[0][0], 404)

    def test_a_corrupt_payload_does_not_break_the_thread(self):
        cid = self.store.create_conversation(user_id="usr_a")
        self.store.add_chat_turn(cid, "q", "a", "{not json")
        h = self._handler()
        h.serve_chat(cid)
        self.assertIsNone(h.sent[0]["turns"][0]["payload"])
        self.assertEqual(h.sent[0]["turns"][0]["answer"], "a")

    def test_rename_and_delete_routes(self):
        cid = self.store.create_conversation(user_id="usr_a")

        h = self._handler({"title": ""})
        h.serve_chat_rename(cid)
        self.assertEqual(h.errors[0][0], 400)

        h = self._handler({"title": "Renamed"})
        h.serve_chat_rename(cid)
        self.assertEqual(h.sent[0]["conversation"]["title"], "Renamed")

        h = self._handler()
        h.serve_chat_delete(cid)
        self.assertEqual(h.sent[0]["deleted"], cid)
        self.assertIsNone(self.store.get_conversation(cid))

        h = self._handler()
        h.serve_chat_delete(cid)
        self.assertEqual(h.errors[0][0], 404)

    def test_another_operators_thread_cannot_be_renamed_or_deleted(self):
        """Knowing the id is not authority over the thread (IDOR)."""
        theirs = self.store.create_conversation(user_id="usr_b")

        h = self._handler({"title": "hijacked"}, user="usr_a")
        h.serve_chat_rename(theirs)
        self.assertEqual(h.errors[0][0], 404)
        self.assertEqual(self.store.get_conversation(theirs)["title"], "New chat")

        h = self._handler(user="usr_a")
        h.serve_chat_delete(theirs)
        self.assertEqual(h.errors[0][0], 404)
        self.assertIsNotNone(self.store.get_conversation(theirs))

        # ...and the owner still can
        h = self._handler({"title": "mine"}, user="usr_b")
        h.serve_chat_rename(theirs)
        self.assertEqual(h.sent[0]["conversation"]["title"], "mine")

    def test_a_turn_cannot_be_written_into_another_operators_thread(self):
        theirs = self.store.create_conversation(user_id="usr_b")
        self.assertIsNone(self.store.add_chat_turn_owned(theirs, "usr_a", "q", "a"))
        self.assertEqual(self.store.conversation_turns(theirs), [])
        self.assertIsNotNone(self.store.add_chat_turn_owned(theirs, "usr_b", "q", "a"))

    def test_an_unowned_thread_stays_reachable(self):
        """CLI-created threads carry no user_id; scoping must not orphan them."""
        cid = self.store.create_conversation()
        self.assertIsNotNone(self.store.get_conversation(cid, "usr_a"))
        self.assertTrue(self.store.rename_conversation(cid, "still mine", "usr_a"))
        self.assertTrue(self.store.delete_conversation(cid, "usr_a"))

    def test_ask_counts_are_read_off_the_evidence(self):
        result = {
            "total_observations": 9,
            "entities": [
                {"keyframes": ["a", "b"], "rule_events": [{"rule": "linger"}]},
                {"keyframes": ["c"], "rule_events": []},
            ],
        }
        self.assertEqual(web.RequestHandler._ask_counts(result),
                         {"entities": 2, "observations": 9, "keyframes": 3, "rule_events": 1})
        self.assertEqual(web.RequestHandler._ask_counts({}),
                         {"entities": 0, "observations": 0, "keyframes": 0, "rule_events": 0})


if __name__ == "__main__":
    unittest.main()
