"""V4-M4: alert triage state (FR-AL-1..4).

Fire -> new -> acknowledged -> resolved, and the faceted chip counts that the
filter bar renders from the same payload.
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

from intelligence_os.tests import _stubs   # noqa: F401  (stubs the deps we lack)

from intelligence_os import web                      # noqa: E402
from intelligence_os.store import Store              # noqa: E402


def _seed(store):
    """Two fired rules + one ordinary detection that must never be an alert."""
    ids = {}
    for rule, cam in (("loitering", "front"), ("smoking", "back")):
        eid = store.create_entity("person", label=f"{rule} guy")
        ids[rule] = store.add_observation(
            eid, f"rule_fired:{rule}", origin="rule", camera_id=cam,
            source_ref=f"/frames/{rule}.jpg")
    eid = store.create_entity("person")
    store.add_observation(eid, "seen_at", origin="detector", camera_id="front")
    return ids


class TestAlertStore(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = Store(db_path=Path(self.tmp.name) / "t.db")
        self.ids = _seed(self.store)

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()

    def test_fired_rules_start_new_and_detections_are_not_alerts(self):
        alerts = self.store.alerts()
        self.assertEqual(len(alerts), 2, "only origin='rule' rows are alerts")
        self.assertTrue(all(a["status"] == "new" for a in alerts))

    def test_transition_is_attributed_and_narrows_the_new_set(self):
        self.assertTrue(self.store.set_alert_status(
            self.ids["smoking"], "acknowledged", user_id="u1"))
        self.assertEqual(len(self.store.alerts(status="new")), 1)
        ack = self.store.alerts(status="acknowledged")
        self.assertEqual([a["assignee"] for a in ack], ["u1"])

    def test_unknown_id_and_bad_status_are_rejected(self):
        self.assertFalse(self.store.set_alert_status("obs_nope", "resolved"))
        with self.assertRaises(ValueError):
            self.store.set_alert_status(self.ids["smoking"], "archived")

    def test_migration_backfills_rows_written_before_the_column(self):
        self.store.conn.execute("UPDATE observations SET status=NULL")
        self.store.conn.commit()
        self.store._migrate_v4()
        self.assertEqual(len(self.store.alerts(status="new")), 2)


class TestAlertsEndpoint(unittest.TestCase):
    """The handler, socket-less: only the alerts path is exercised."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Path(self.tmp.name) / "t.db"
        s = Store(db_path=self.db)
        self.ids = _seed(s)
        s.close()
        self._real_store = web.Store
        # Store() is constructed inside the handler with no args.
        db = self.db
        web.Store = lambda *a, **k: self._real_store(db_path=db)
        web.Store.ALERT_STATUSES = self._real_store.ALERT_STATUSES
        # severity lives in rules.yaml; pin it so the test doesn't read the disk
        self._real_rules = sys.modules["intelligence_os.rules"].load_rules
        sys.modules["intelligence_os.rules"].load_rules = lambda: [
            {"name": "loitering", "severity": "high"}, {"name": "smoking"}]

    def tearDown(self):
        web.Store = self._real_store
        sys.modules["intelligence_os.rules"].load_rules = self._real_rules
        self.tmp.cleanup()

    def _get(self, query=""):
        h = object.__new__(web.RequestHandler)
        h.sent = []
        h.send_json = h.sent.append
        h.send_error = lambda *a: h.sent.append({"error": a})
        h.path = "/api/alerts" + query
        h.serve_alerts()
        return h.sent[0]

    def _post(self, oid, body):
        h = object.__new__(web.RequestHandler)
        h.sent = []
        h.send_json = h.sent.append
        h.send_error = lambda *a: h.sent.append({"error": a})
        h._read_json = lambda: body
        h.current_user_id = "u1"
        h.serve_alert_status(oid)
        return h.sent[0]

    def test_severity_comes_from_the_rule_and_defaults_to_medium(self):
        by_rule = {a["rule"]: a for a in self._get()["alerts"]}
        self.assertEqual(by_rule["loitering"]["severity"], "high")
        self.assertEqual(by_rule["smoking"]["severity"], "medium")

    def test_counts_are_faceted_so_a_chip_shows_what_clicking_it_gives(self):
        c = self._get()["counts"]
        self.assertEqual((c["all"], c["high"], c["medium"], c["new"]), (2, 1, 1, 2))
        # filtering to high must not zero the *other* severity chips, or you could
        # never click your way back out
        c = self._get("?severity=high")["counts"]
        self.assertEqual((c["all"], c["high"], c["medium"]), (1, 1, 1))
        self.assertEqual(c["new"], 1, "status chips respect the severity filter")

    def test_badge_and_dashboard_ignore_the_filter_bar(self):
        """Filtering the table must not make the rail badge look like alerts
        vanished — unread/recent are global."""
        r = self._get("?severity=high&status=new&q=smok")
        self.assertEqual(len(r["alerts"]), 0, "the filter really does bite")
        self.assertEqual(r["unread"], 2)
        self.assertEqual(len(r["recent"]), 2)

    def test_search_matches_rule_entity_and_camera(self):
        for q, n in (("smok", 1), ("front", 1), ("guy", 2), ("zzz", 0)):
            self.assertEqual(len(self._get("?q=" + q)["alerts"]), n, q)

    def test_ack_decrements_the_badge_count(self):
        self.assertEqual(self._get()["counts"]["new"], 2)
        self._post(self.ids["smoking"], {"status": "acknowledged"})
        c = self._get()["counts"]
        self.assertEqual((c["new"], c["acknowledged"]), (1, 1))

    def test_mark_all_read_clears_new_and_leaves_resolved_alone(self):
        self._post(self.ids["smoking"], {"status": "resolved"})
        self.assertEqual(self._post("ignored", {"status": "acknowledged", "all": True}),
                         {"ok": True, "updated": 1})
        c = self._get()["counts"]
        self.assertEqual((c["new"], c["acknowledged"], c["resolved"]), (0, 1, 1))

    def test_bad_input_is_rejected(self):
        self.assertIn("error", self._get("?status=bogus"))
        self.assertIn("error", self._post("obs_nope", {"status": "resolved"}))
        self.assertIn("error", self._post(self.ids["smoking"], {"status": "bogus"}))


if __name__ == "__main__":
    unittest.main()
