"""V4-M8 cases + reports.

Cases are a flat container over ids that already exist, so what matters is that
an attached id resolves back to the thing it names and that status sticks.
Reports are a frozen copy of a digest, so what matters is that the body stays
put once written.
"""
from __future__ import annotations

import tempfile
import time
import unittest
from unittest import mock

from intelligence_os.tests import _stubs   # noqa: F401  (stubs the deps we lack)

from intelligence_os import web                            # noqa: E402
from intelligence_os.store import Store                    # noqa: E402


class TestCases(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = Store(db_path=self.tmp.name + "/c.db")
        self.close = self.store.close
        self.store.close = lambda: None   # the handler closes what it opens
        self.now = time.time()

        self.p = self.store.create_entity("person", label="Dock worker")
        self.alert = self.store.add_observation(
            self.p, "rule_fired:after_hours", origin="rule", timestamp=self.now - 60)

        self.h = object.__new__(web.RequestHandler)
        self.h.path = "/api/cases"
        self.h.current_user_id = "u1"
        self.sent = {}
        self.h.send_json = lambda d: self.sent.update(d)
        self.h.send_error = lambda *a: self.sent.update({"error": a})
        fake = lambda *a, **k: self.store
        fake.CASE_STATUSES = Store.CASE_STATUSES   # the handler reads it off the class
        self._patch = mock.patch.object(web, "Store", fake)
        self._patch.start()

    def tearDown(self):
        self._patch.stop()
        self.close()
        self.tmp.cleanup()

    def post(self, path, body):
        self.h.path = path
        self.h._read_json = lambda: body
        self.sent.clear()
        return self.h

    def get_cases(self, qs=""):
        self.h.path = "/api/cases" + qs
        self.sent.clear()
        self.h.serve_cases()
        return self.sent

    def test_a_case_created_from_an_alert_carries_it(self):
        """FR-CS-3: the + Case button on an alert row is one call, not two."""
        self.post("/api/cases", {"title": "Dock incident", "description": "night entry",
                                 "kind": "alert", "ref_id": self.alert}).serve_case_create()
        cid = self.sent["id"]
        c = self.get_cases()["cases"][0]
        self.assertEqual(c["id"], cid)
        self.assertEqual(c["owner"], "u1")
        self.assertEqual(c["counts"], {"alert": 1, "entity": 0, "event": 0})
        # the id resolves to the rule that fired, not the raw observation id
        self.assertEqual(c["items"][0]["label"], "after hours")

    def test_an_attached_entity_resolves_to_its_label(self):
        cid = self.store.create_case("Follow-up")
        self.post(f"/api/cases/{cid}/attach",
                  {"kind": "entity", "ref_id": self.p}).serve_case_attach(cid)
        item = self.get_cases()["cases"][0]["items"][0]
        self.assertEqual((item["kind"], item["label"]), ("entity", "Dock worker"))

    def test_attaching_twice_is_not_a_duplicate(self):
        cid = self.store.create_case("Follow-up")
        for _ in range(2):
            self.post(f"/api/cases/{cid}/attach",
                      {"kind": "entity", "ref_id": self.p}).serve_case_attach(cid)
        self.assertEqual(len(self.get_cases()["cases"][0]["items"]), 1)

    def test_attach_rejects_an_unknown_case_and_an_unknown_kind(self):
        self.post("/api/cases/nope/attach",
                  {"kind": "entity", "ref_id": self.p}).serve_case_attach("nope")
        self.assertEqual(self.sent["error"][0], 404)
        cid = self.store.create_case("Follow-up")
        self.post(f"/api/cases/{cid}/attach",
                  {"kind": "camera", "ref_id": "x"}).serve_case_attach(cid)
        self.assertEqual(self.sent["error"][0], 400)

    def test_status_moves_and_the_filter_follows_it(self):
        cid = self.store.create_case("Dock incident")
        self.post(f"/api/cases/{cid}/status", {"status": "closed"}).serve_case_status(cid)
        d = self.get_cases()
        self.assertEqual(d["counts"], {"all": 1, "open": 0, "review": 0, "closed": 1})
        self.assertIsNotNone(d["cases"][0]["closed_at"])
        self.assertEqual(self.get_cases("?status=open")["cases"], [])
        self.assertEqual(len(self.get_cases("?status=closed")["cases"]), 1)
        # counts are global so the chips never look like cases vanished
        self.assertEqual(self.get_cases("?status=open")["counts"]["all"], 1)

    def test_a_bad_status_is_rejected_not_stored(self):
        cid = self.store.create_case("Dock incident")
        self.post(f"/api/cases/{cid}/status", {"status": "wontfix"}).serve_case_status(cid)
        self.assertEqual(self.sent["error"][0], 400)
        self.assertEqual(self.get_cases()["cases"][0]["status"], "open")

    def test_a_case_needs_a_title(self):
        self.post("/api/cases", {"description": "no title"}).serve_case_create()
        self.assertEqual(self.sent["error"][0], 400)

    def test_a_deleted_item_still_renders(self):
        """A case outlives the rows it points at; a card must not blow up."""
        cid = self.store.create_case("Dock incident")
        self.store.attach_to_case(cid, "entity", "ent_gone")
        self.assertEqual(self.get_cases()["cases"][0]["items"][0]["label"], "(deleted)")


class TestReports(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = Store(db_path=self.tmp.name + "/r.db")
        self.close = self.store.close
        self.store.close = lambda: None
        self.now = time.time()
        self.p = self.store.create_entity("person", label="Dock worker")
        self.store.add_observation(self.p, "rule_fired:after_hours", origin="rule",
                                   timestamp=self.now - 60)

        self.h = object.__new__(web.RequestHandler)
        self.sent = {}
        self.h.send_json = lambda d: self.sent.update(d)
        self.h.send_error = lambda *a: self.sent.update({"error": a})
        fake = lambda *a, **k: self.store
        fake.CASE_STATUSES = Store.CASE_STATUSES   # the handler reads it off the class
        self._patch = mock.patch.object(web, "Store", fake)
        self._patch.start()

    def tearDown(self):
        self._patch.stop()
        self.close()
        self.tmp.cleanup()

    def gen(self, body):
        self.h.path = "/api/reports"
        self.h._read_json = lambda: body
        self.sent.clear()
        self.h.serve_report_create()
        return self.sent

    def test_csv_holds_the_fired_rule(self):
        d = self.gen({"hours": 24, "format": "csv"})
        body = self.store.get_report(d["id"])["body"]
        # three stacked sections, SUMMARY first — the findings header is further down
        lines = body.splitlines()
        self.assertEqual(lines[0], "SUMMARY")
        self.assertIn("FINDINGS", lines)
        self.assertTrue(any(l.startswith("band,title,entity") for l in lines), body)
        self.assertIn("Rule fired: after_hours", body)
        self.assertIn("Dock worker", body)

    def test_the_listing_names_the_range_and_omits_the_body(self):
        self.gen({"hours": 168, "format": "csv"})
        self.h.serve_reports()
        r = self.sent["reports"][0]
        self.assertEqual(r["name"], "Digest · last 7 days")
        self.assertNotIn("body", r)
        self.assertGreater(r["bytes"], 0)
        self.assertAlmostEqual(r["range_end"] - r["range_start"], 168 * 3600, delta=2)

    def test_a_stored_report_does_not_change_when_memory_does(self):
        """FR-RP-1 is a record of what was true then, not a live query."""
        rid = self.gen({"hours": 24, "format": "csv"})["id"]
        before = self.store.get_report(rid)["body"]
        self.store.add_observation(self.p, "rule_fired:loitering", origin="rule",
                                   timestamp=time.time())
        self.assertEqual(self.store.get_report(rid)["body"], before)

    def test_an_unknown_format_is_refused(self):
        self.assertEqual(self.gen({"format": "pdf"})["error"][0], 400)


if __name__ == "__main__":
    unittest.main()
