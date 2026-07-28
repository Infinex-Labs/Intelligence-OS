"""SQLite data model + correctness machinery (§6, §7).

This module is the *product*: persistent, correctable, evidence-weighted memory.
The vision models are commodity glue around it. Everything here is reversible and
carries provenance; nothing is stored as unconditional fact.

Schema (one DB, SQLite for the POC -> Postgres/graph at scale):
  entities          person|object, with status for merge/split/delete
  signatures        embeddings per entity (face=long-term, body=short-term, appearance)
  locations         fixed-camera regions
  observations      perceived, frequent, cheap; predicate is an OPEN string (§5b)
  scene_snapshots   per-location inventory over time
  relations         distilled, weighted, decaying, auditable edges (relation|habit|event)
  predicate_schema  legal (subject_type, predicate, object_type) triples (grows over time)
"""
from __future__ import annotations

import json
import math
import os
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from .config import CONFIG, DB_PATH, ensure_dirs


def _uid(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def now() -> float:
    return time.time()


# Search plan Phase 1. Some facts have no owner: an open gate, a spill, a stack
# of pallets. The old write path had one option — pin them on the first person in
# frame — which reads as a claim about that person. These get a subject of their
# own instead, named for the place, so the fact stays queryable without inventing
# an entity or libelling a bystander.
SCENE_PREFIX = "scene:"


def scene_subject(location_id: Optional[str] = None,
                  camera_id: Optional[str] = None) -> str:
    """Subject id for an unowned fact. Prefers the zone; falls back to the camera."""
    return SCENE_PREFIX + (location_id or camera_id or "unknown")


SCHEMA = """
CREATE TABLE IF NOT EXISTS entities (
    entity_id   TEXT PRIMARY KEY,
    type        TEXT NOT NULL CHECK (type IN ('person','object')),
    label       TEXT,
    created_at  REAL NOT NULL,
    status      TEXT NOT NULL DEFAULT 'active'   -- active | merged_into:<id> | deleted
);

CREATE TABLE IF NOT EXISTS signatures (
    sig_id      TEXT PRIMARY KEY,
    entity_id   TEXT NOT NULL,
    kind        TEXT NOT NULL,        -- face | body | appearance
    dim         INTEGER NOT NULL,
    vec         BLOB NOT NULL,        -- float32 L2-normalized
    created_at  REAL NOT NULL,
    FOREIGN KEY (entity_id) REFERENCES entities(entity_id)
);

CREATE TABLE IF NOT EXISTS locations (
    location_id TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    region      TEXT NOT NULL,        -- json polygon / bbox in frame coords
    camera_id   TEXT                  -- M6: zone belongs to one camera's frame (NULL = legacy)
);

CREATE TABLE IF NOT EXISTS observations (
    observation_id    TEXT PRIMARY KEY,
    subject_entity_id TEXT NOT NULL,
    predicate         TEXT NOT NULL,  -- OPEN string, not an enum (§5b)
    object_entity_id  TEXT,
    location_id       TEXT,
    timestamp         REAL NOT NULL,
    confidence        REAL NOT NULL,
    source_ref        TEXT,           -- frame/keyframe path for audit
    origin            TEXT NOT NULL,  -- detector | vlm
    camera_id         TEXT,           -- M6: which camera produced this (NULL = legacy single cam)
    user_id           TEXT            -- M7: user attribution (NULL = legacy/system)
);

CREATE TABLE IF NOT EXISTS scene_snapshots (
    snapshot_id        TEXT PRIMARY KEY,
    location_id        TEXT NOT NULL,
    timestamp          REAL NOT NULL,
    present_entity_ids TEXT NOT NULL  -- json list
);

CREATE TABLE IF NOT EXISTS relations (
    relation_id              TEXT PRIMARY KEY,
    kind                     TEXT NOT NULL,  -- relation | habit | event
    subject_entity_id        TEXT NOT NULL,
    predicate                TEXT NOT NULL,  -- canonical (§5b)
    object_entity_id         TEXT,
    location_id              TEXT,
    weight                   REAL NOT NULL,
    supporting_observation_ids TEXT NOT NULL,  -- json list (provenance)
    last_reinforced_at       REAL NOT NULL,
    created_at               REAL NOT NULL,
    status                   TEXT NOT NULL DEFAULT 'candidate'  -- candidate | confirmed
);

CREATE TABLE IF NOT EXISTS predicate_schema (
    predicate    TEXT NOT NULL,
    subject_type TEXT NOT NULL,   -- person | object | any
    object_type  TEXT NOT NULL,   -- person | object | none | any
    canonical    TEXT,            -- canonical form for normalization (§5b)
    PRIMARY KEY (predicate, subject_type, object_type)
);

CREATE TABLE IF NOT EXISTS users (
    user_id       TEXT PRIMARY KEY,
    username      TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    created_at    REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    user_id    TEXT NOT NULL,
    expires_at REAL NOT NULL,
    FOREIGN KEY(user_id) REFERENCES users(user_id)
);

-- V4-M8: an investigation is a flat container over things already in memory,
-- so it needs a title and a list of ids — not graph edges.
CREATE TABLE IF NOT EXISTS cases (
    case_id     TEXT PRIMARY KEY,
    title       TEXT NOT NULL,
    description TEXT,
    status      TEXT NOT NULL DEFAULT 'open',   -- open | review | closed
    owner       TEXT,
    opened_at   REAL NOT NULL,
    closed_at   REAL
);

CREATE TABLE IF NOT EXISTS case_items (
    case_id  TEXT NOT NULL,
    kind     TEXT NOT NULL,       -- alert | entity | event
    ref_id   TEXT NOT NULL,
    added_at REAL NOT NULL,
    PRIMARY KEY (case_id, kind, ref_id),
    FOREIGN KEY(case_id) REFERENCES cases(case_id) ON DELETE CASCADE
);

-- V4-M8: the rendered body is stored, not a file path — a report is a record of
-- what memory said then, and re-rendering later would quietly change it.
CREATE TABLE IF NOT EXISTS reports (
    report_id    TEXT PRIMARY KEY,
    name         TEXT NOT NULL,
    format       TEXT NOT NULL,   -- html | csv
    range_start  REAL NOT NULL,
    range_end    REAL NOT NULL,
    generated_at REAL NOT NULL,
    body         TEXT NOT NULL
);

-- V4-M10: the assistant's chat history. A thread is a title plus ordered turns,
-- and each turn stores the evidence payload it was rendered from — so reopening
-- a thread shows what memory said then, not a silent re-run of the question.
-- The counters are denormalized out of that payload on write: the KPI strip is
-- then a SQL aggregate instead of a JSON scan over every answer ever given.
CREATE TABLE IF NOT EXISTS conversations (
    conversation_id TEXT PRIMARY KEY,
    user_id         TEXT,
    title           TEXT NOT NULL,
    created_at      REAL NOT NULL,
    updated_at      REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS chat_turns (
    turn_id         TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    asked_at        REAL NOT NULL,
    question        TEXT NOT NULL,
    answer          TEXT,
    payload         TEXT,            -- the /api/ask result, exactly as rendered
    latency_ms      INTEGER,
    n_entities      INTEGER NOT NULL DEFAULT 0,
    n_observations  INTEGER NOT NULL DEFAULT 0,
    n_keyframes     INTEGER NOT NULL DEFAULT 0,
    n_rule_events   INTEGER NOT NULL DEFAULT 0,
    FOREIGN KEY(conversation_id) REFERENCES conversations(conversation_id) ON DELETE CASCADE
);

-- Search plan Phase 1: the COMPLETE vision-model report, kept verbatim.
-- Until this table existed, `SceneDescription.states()` picked three of the four
-- sections it returns and the rest was dropped before the insert — an open gate
-- was seen, described, and then deleted. `raw` is the report as JSON so nothing
-- is lost even if the flattening rules change later; `text` is the flattened
-- prose, which is the surface the lexical and semantic indexes are built on.
CREATE TABLE IF NOT EXISTS scene_descriptions (
    description_id TEXT PRIMARY KEY,
    camera_id      TEXT,
    location_id    TEXT,
    timestamp      REAL NOT NULL,
    source_ref     TEXT,            -- keyframe, for provenance
    model          TEXT NOT NULL,   -- which VLM produced it
    raw            TEXT NOT NULL,   -- the whole report, verbatim JSON
    text           TEXT NOT NULL    -- flattened prose
);

CREATE INDEX IF NOT EXISTS idx_obs_subject ON observations(subject_entity_id);
CREATE INDEX IF NOT EXISTS idx_obs_time    ON observations(timestamp);
CREATE INDEX IF NOT EXISTS idx_desc_time   ON scene_descriptions(timestamp);
CREATE INDEX IF NOT EXISTS idx_sig_entity  ON signatures(entity_id);
CREATE INDEX IF NOT EXISTS idx_snap_loc    ON scene_snapshots(location_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_rel_subject ON relations(subject_entity_id);
CREATE INDEX IF NOT EXISTS idx_convo_user  ON conversations(user_id, updated_at);
CREATE INDEX IF NOT EXISTS idx_turn_convo  ON chat_turns(conversation_id, asked_at);
"""


class Store:
    """Thin, explicit wrapper over SQLite. No ORM; the schema IS the contract."""

    def __init__(self, db_path: Optional[Path | str] = None):
        ensure_dirs()
        if db_path is None:
            # Respect runtime environment variable changes (critical for test overrides)
            env_db = os.environ.get("INTELLIGENCE_OS_DB")
            self.db_path = str(Path(env_db)) if env_db else str(DB_PATH)
        else:
            self.db_path = str(db_path)
        # run.py builds ONE Store on the main thread and hands it to every camera
        # thread and to the shared resolvers, so the connection has to outlive its
        # creating thread. SQLite is compiled SQLITE_THREADSAFE=1 (serialized) on
        # every build we ship on, which is what makes sharing the handle safe --
        # test_store_threads asserts it via PRAGMA compile_options, because the
        # sqlite3.threadsafety attribute only reports the real mode on 3.11+; WAL below is about
        # concurrent *connections* (the web server opens its own per request) and
        # never covered this.
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._wlock = threading.RLock()      # see tx()
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        # M6: WAL mode allows concurrent writes from multiple camera threads
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.executescript(SCHEMA)
        self._migrate_m6()
        self._migrate_m7()
        self._migrate_m9()
        self._migrate_v4()
        self._migrate_v5_search()
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def _migrate_m6(self) -> None:
        """Additive migration: camera_id on observations + locations. Existing rows
        get NULL = the pre-existing single camera. Safe to re-run (try/except)."""
        for stmt in (
            "ALTER TABLE observations ADD COLUMN camera_id TEXT",
            "ALTER TABLE locations ADD COLUMN camera_id TEXT",
        ):
            try:
                self.conn.execute(stmt)
            except sqlite3.OperationalError:
                pass  # column already exists
        try:
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_obs_camera ON observations(camera_id)")
        except sqlite3.OperationalError:
            pass

    def _migrate_m7(self) -> None:
        """Additive migration for M7: user_id column on observations."""
        try:
            self.conn.execute("ALTER TABLE observations ADD COLUMN user_id TEXT")
        except sqlite3.OperationalError:
            pass  # column already exists
        try:
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_obs_user ON observations(user_id)")
        except sqlite3.OperationalError:
            pass

    def _migrate_m9(self) -> None:
        """Additive migration for M9: delivery settings columns on users table."""
        # Add columns one by one in case any already exist.
        # SQLite doesn't support multiple ADD COLUMN statements in one query easily.
        for col, col_def in (
            ("delivery_schedule", "TEXT NOT NULL DEFAULT 'off'"),
            ("delivery_sink", "TEXT NOT NULL DEFAULT 'in_app'"),
            ("delivery_destination", "TEXT"),
            ("email", "TEXT"),
            ("last_delivered_at", "REAL DEFAULT 0"),
        ):
            try:
                self.conn.execute(f"ALTER TABLE users ADD COLUMN {col} {col_def}")
            except sqlite3.OperationalError:
                pass  # column already exists

    def _migrate_v4(self) -> None:
        """V4-M4: a fired alert *is* an observation, so triage state lives on the
        row rather than in a parallel alerts table that would need syncing."""
        for stmt in (
            "ALTER TABLE observations ADD COLUMN status TEXT",
            "ALTER TABLE observations ADD COLUMN assignee TEXT",
        ):
            try:
                self.conn.execute(stmt)
            except sqlite3.OperationalError:
                pass  # column already exists
        # Backfill + catch rows written by an older add_observation. Idempotent.
        self.conn.execute(
            "UPDATE observations SET status='new' WHERE origin='rule' AND status IS NULL")
        try:
            self.conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_obs_status ON observations(status)")
        except sqlite3.OperationalError:
            pass

    def _migrate_v5_search(self) -> None:
        """Search plan Phase 1: link each row back to the full report it came
        from, and give it a human-readable `text`.

        `predicate` stays exactly what it was — the machine contract that rules
        and distillation match on. `text` is the search surface, so rewording a
        row for searchability can never break a rule.
        """
        for stmt in (
            "ALTER TABLE observations ADD COLUMN description_id TEXT",
            "ALTER TABLE observations ADD COLUMN text TEXT",
        ):
            try:
                self.conn.execute(stmt)
            except sqlite3.OperationalError:
                pass  # column already exists
        try:
            self.conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_obs_desc ON observations(description_id)")
        except sqlite3.OperationalError:
            pass

    # --- alerts (V4-M4) ------------------------------------------------------
    ALERT_STATUSES = ("new", "acknowledged", "resolved")

    def alerts(self, *, status: Optional[str] = None, camera_id: Optional[str] = None,
               since: Optional[float] = None) -> list[sqlite3.Row]:
        """Fired rules, newest first. Severity is not stored — it comes from the
        rule spec in rules.yaml, so the caller joins it on."""
        q = ["SELECT * FROM observations WHERE origin='rule'"]
        args: list = []
        if status:
            q.append("AND IFNULL(status,'new')=?"); args.append(status)
        if camera_id:
            q.append("AND camera_id=?"); args.append(camera_id)
        if since is not None:
            q.append("AND timestamp>=?"); args.append(since)
        q.append("ORDER BY timestamp DESC")
        return self.conn.execute(" ".join(q), args).fetchall()

    def set_alert_status(self, observation_id: str, status: str,
                         user_id: Optional[str] = None) -> bool:
        """Returns False if there is no such fired-rule observation."""
        if status not in self.ALERT_STATUSES:
            raise ValueError(f"status must be one of {self.ALERT_STATUSES}")
        with self.tx() as c:
            cur = c.execute(
                "UPDATE observations SET status=?, assignee=? "
                "WHERE observation_id=? AND origin='rule'",
                (status, user_id, observation_id))
        return cur.rowcount > 0

    # --- cases (V4-M8) -------------------------------------------------------
    CASE_STATUSES = ("open", "review", "closed")

    def create_case(self, title: str, description: Optional[str] = None,
                    owner: Optional[str] = None) -> str:
        cid = _uid("case")
        with self.tx() as c:
            c.execute("INSERT INTO cases(case_id,title,description,status,owner,opened_at)"
                      " VALUES (?,?,?,'open',?,?)", (cid, title, description, owner, now()))
        return cid

    def cases(self, status: Optional[str] = None) -> list[sqlite3.Row]:
        q = "SELECT * FROM cases"
        args: list = []
        if status:
            q += " WHERE status=?"; args.append(status)
        return self.conn.execute(q + " ORDER BY opened_at DESC", args).fetchall()

    def case_items(self, case_id: str) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM case_items WHERE case_id=? ORDER BY added_at", (case_id,)).fetchall()

    def attach_to_case(self, case_id: str, kind: str, ref_id: str) -> bool:
        """False if there is no such case. Attaching twice is a no-op, not an error."""
        if not self.conn.execute("SELECT 1 FROM cases WHERE case_id=?", (case_id,)).fetchone():
            return False
        with self.tx() as c:
            c.execute("INSERT OR IGNORE INTO case_items(case_id,kind,ref_id,added_at)"
                      " VALUES (?,?,?,?)", (case_id, kind, ref_id, now()))
        return True

    def set_case_status(self, case_id: str, status: str) -> bool:
        if status not in self.CASE_STATUSES:
            raise ValueError(f"status must be one of {self.CASE_STATUSES}")
        with self.tx() as c:
            cur = c.execute("UPDATE cases SET status=?, closed_at=? WHERE case_id=?",
                            (status, now() if status == "closed" else None, case_id))
        return cur.rowcount > 0

    # --- reports (V4-M8) -----------------------------------------------------
    def add_report(self, name: str, fmt: str, start: float, end: float, body: str) -> str:
        rid = _uid("rep")
        with self.tx() as c:
            c.execute("INSERT INTO reports(report_id,name,format,range_start,range_end,"
                      "generated_at,body) VALUES (?,?,?,?,?,?,?)",
                      (rid, name, fmt, start, end, now(), body))
        return rid

    def reports(self) -> list[sqlite3.Row]:
        """Without the body — the list view would otherwise carry every report ever."""
        return self.conn.execute(
            "SELECT report_id,name,format,range_start,range_end,generated_at,LENGTH(body) AS bytes"
            " FROM reports ORDER BY generated_at DESC").fetchall()

    def get_report(self, report_id: str) -> Optional[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM reports WHERE report_id=?", (report_id,)).fetchone()

    # --- assistant chat history (V4-M10) -------------------------------------
    NEW_CHAT_TITLE = "New chat"

    @staticmethod
    def title_from_question(question: str, limit: int = 58) -> str:
        """A thread names itself after its opening question, cut at a word
        boundary so the sidebar never shows half a word."""
        q = " ".join(str(question or "").split())
        if not q:
            return Store.NEW_CHAT_TITLE
        if len(q) <= limit:
            return q
        cut = q[:limit].rsplit(" ", 1)[0] or q[:limit]
        return cut.rstrip(" ,;:.") + "…"

    def create_conversation(self, title: Optional[str] = None,
                            user_id: Optional[str] = None) -> str:
        cid = _uid("chat")
        ts = now()
        with self.tx() as c:
            c.execute("INSERT INTO conversations(conversation_id,user_id,title,"
                      "created_at,updated_at) VALUES (?,?,?,?,?)",
                      (cid, user_id, (title or self.NEW_CHAT_TITLE).strip()
                       or self.NEW_CHAT_TITLE, ts, ts))
        return cid

    def conversations(self, user_id: Optional[str] = None,
                      limit: int = 200) -> list[sqlite3.Row]:
        """Newest activity first, with the counts the sidebar shows. Passing a
        user_id scopes the list to that operator; None means every thread."""
        return self.conn.execute(
            "SELECT c.*, COUNT(t.turn_id) AS n_turns, MAX(t.asked_at) AS last_asked "
            "FROM conversations c LEFT JOIN chat_turns t USING(conversation_id) "
            "WHERE (? IS NULL OR c.user_id = ?) "
            "GROUP BY c.conversation_id ORDER BY c.updated_at DESC LIMIT ?",
            (user_id, user_id, limit)).fetchall()

    def get_conversation(self, conversation_id: str,
                         user_id: Optional[str] = None) -> Optional[sqlite3.Row]:
        """Passing a user_id makes this the ownership check too: another
        operator's thread reads as absent rather than forbidden, so a thread id
        can't be probed for existence. None means unscoped (CLI/tests), and a
        thread with no owner (CLI-created) is visible to anyone signed in."""
        return self.conn.execute(
            "SELECT * FROM conversations WHERE conversation_id=? "
            "AND (? IS NULL OR user_id IS NULL OR user_id = ?)",
            (conversation_id, user_id, user_id)).fetchone()

    def conversation_turns(self, conversation_id: str) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM chat_turns WHERE conversation_id=? ORDER BY asked_at, rowid",
            (conversation_id,)).fetchall()

    def add_chat_turn(self, conversation_id: str, question: str,
                      answer: Optional[str] = None, payload: Optional[str] = None,
                      latency_ms: Optional[int] = None, *, counts: Optional[dict] = None
                      ) -> Optional[str]:
        """Append a Q/A to a thread, bump its activity, and adopt the first
        question as the title. Returns None if the thread is gone (deleted in
        another tab while the answer was in flight)."""
        if not self.get_conversation(conversation_id):
            return None
        tid = _uid("turn")
        k = counts or {}
        ts = now()
        with self.tx() as c:
            c.execute("INSERT INTO chat_turns(turn_id,conversation_id,asked_at,question,"
                      "answer,payload,latency_ms,n_entities,n_observations,n_keyframes,"
                      "n_rule_events) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                      (tid, conversation_id, ts, question, answer, payload, latency_ms,
                       int(k.get("entities", 0)), int(k.get("observations", 0)),
                       int(k.get("keyframes", 0)), int(k.get("rule_events", 0))))
            c.execute("UPDATE conversations SET updated_at=?, title=CASE WHEN title=? "
                      "THEN ? ELSE title END WHERE conversation_id=?",
                      (ts, self.NEW_CHAT_TITLE, self.title_from_question(question),
                       conversation_id))
        return tid

    # ownership lives in the WHERE clause of every mutation below, not in the
    # handler: a caller that forgets to pass user_id can't quietly widen access,
    # and a thread id is never enough on its own to rename or delete a thread.
    _OWNED = " AND (? IS NULL OR user_id IS NULL OR user_id = ?)"

    def rename_conversation(self, conversation_id: str, title: str,
                            user_id: Optional[str] = None) -> bool:
        title = " ".join(str(title or "").split())[:120]
        if not title:
            return False
        with self.tx() as c:
            cur = c.execute("UPDATE conversations SET title=? WHERE conversation_id=?"
                            + self._OWNED, (title, conversation_id, user_id, user_id))
        return cur.rowcount > 0

    def delete_conversation(self, conversation_id: str,
                            user_id: Optional[str] = None) -> bool:
        """Turns go with it — the ON DELETE CASCADE is live (foreign_keys=ON)."""
        with self.tx() as c:
            cur = c.execute("DELETE FROM conversations WHERE conversation_id=?"
                            + self._OWNED, (conversation_id, user_id, user_id))
        return cur.rowcount > 0

    def add_chat_turn_owned(self, conversation_id: str, user_id: Optional[str],
                            *args, **kw) -> Optional[str]:
        """add_chat_turn, but refusing to write into someone else's thread."""
        if not self.get_conversation(conversation_id, user_id):
            return None
        return self.add_chat_turn(conversation_id, *args, **kw)

    def chat_stats(self, user_id: Optional[str] = None) -> dict:
        """The assistant KPIs, every one a SUM over turns actually stored — the
        UI displays these, it never accumulates its own tally."""
        r = self.conn.execute(
            "SELECT COUNT(DISTINCT c.conversation_id) AS threads, "
            "       COUNT(t.turn_id)                  AS questions, "
            "       IFNULL(SUM(t.n_entities),0)       AS entities, "
            "       IFNULL(SUM(t.n_observations),0)   AS observations, "
            "       IFNULL(SUM(t.n_keyframes),0)      AS keyframes, "
            "       IFNULL(SUM(t.n_rule_events),0)    AS rule_events, "
            "       AVG(t.latency_ms)                 AS avg_latency_ms, "
            "       MAX(t.asked_at)                   AS last_asked "
            "FROM conversations c LEFT JOIN chat_turns t USING(conversation_id) "
            "WHERE (? IS NULL OR c.user_id = ?)", (user_id, user_id)).fetchone()
        out = dict(r)
        out["avg_latency_ms"] = round(out["avg_latency_ms"]) if out["avg_latency_ms"] else None
        return out

    @contextmanager
    def tx(self):
        """Every write in the codebase goes through here, which is why one lock is
        enough. sqlite3 serializes the C library, but the *implicit BEGIN* is a
        Python-side check-then-act: two camera threads inserting at the same moment
        both see "no transaction open" and the loser raises "cannot start a
        transaction within a transaction". Reads take no lock — SELECT never opens
        one. RLock, not Lock: prune_signatures-style nesting must not deadlock."""
        with self._wlock:
            try:
                yield self.conn
                self.conn.commit()
            except Exception:
                self.conn.rollback()
                raise

    # --- vector helpers ------------------------------------------------------
    @staticmethod
    def _pack(vec: np.ndarray) -> bytes:
        return np.asarray(vec, dtype=np.float32).tobytes()

    @staticmethod
    def _unpack(blob: bytes) -> np.ndarray:
        return np.frombuffer(blob, dtype=np.float32)

    # --- entities ------------------------------------------------------------
    def create_entity(self, type_: str, label: Optional[str] = None) -> str:
        eid = _uid("ent")
        with self.tx() as c:
            c.execute(
                "INSERT INTO entities(entity_id,type,label,created_at,status) VALUES (?,?,?,?,'active')",
                (eid, type_, label, now()),
            )
        return eid

    def get_entity(self, entity_id: str) -> Optional[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM entities WHERE entity_id=?", (entity_id,)
        ).fetchone()

    def list_entities(self, type_: Optional[str] = None, active_only: bool = True
                      ) -> list[sqlite3.Row]:
        q = "SELECT * FROM entities"
        conds, args = [], []
        if type_:
            conds.append("type=?"); args.append(type_)
        if active_only:
            conds.append("status='active'")
        if conds:
            q += " WHERE " + " AND ".join(conds)
        q += " ORDER BY created_at"
        return self.conn.execute(q, args).fetchall()

    def appearances(self) -> dict[str, sqlite3.Row]:
        """times_seen / last_seen / newest keyframe per entity, one pass.
        Derived from observations — no counter column that can drift."""
        rows = self.conn.execute("""
            SELECT subject_entity_id AS eid, COUNT(*) AS n, MAX(timestamp) AS last_seen,
                   (SELECT source_ref FROM observations x
                     WHERE x.subject_entity_id = o.subject_entity_id
                       AND x.source_ref IS NOT NULL
                     ORDER BY x.timestamp DESC LIMIT 1) AS keyframe
              FROM observations o GROUP BY subject_entity_id""").fetchall()
        return {r["eid"]: r for r in rows}

    def set_label(self, entity_id: str, label: str) -> None:
        with self.tx() as c:
            c.execute("UPDATE entities SET label=? WHERE entity_id=?", (label, entity_id))

    # --- signatures ----------------------------------------------------------
    def add_signature(self, entity_id: str, kind: str, vec: np.ndarray) -> str:
        sid = _uid("sig")
        v = np.asarray(vec, dtype=np.float32)
        with self.tx() as c:
            c.execute(
                "INSERT INTO signatures(sig_id,entity_id,kind,dim,vec,created_at) VALUES (?,?,?,?,?,?)",
                (sid, entity_id, kind, v.shape[0], self._pack(v), now()),
            )
        return sid

    def signatures(self, kind: Optional[str] = None) -> list[tuple[str, str, np.ndarray]]:
        """Return [(entity_id, sig_id, vec)] for active entities only."""
        q = ("SELECT s.entity_id, s.sig_id, s.vec FROM signatures s "
             "JOIN entities e ON e.entity_id=s.entity_id WHERE e.status='active'")
        args: list[Any] = []
        if kind:
            q += " AND s.kind=?"; args.append(kind)
        out = []
        for row in self.conn.execute(q, args):
            out.append((row["entity_id"], row["sig_id"], self._unpack(row["vec"])))
        return out

    def entity_signatures(self, entity_id: str, kind: Optional[str] = None
                         ) -> list[np.ndarray]:
        q = "SELECT vec FROM signatures WHERE entity_id=?"
        args: list[Any] = [entity_id]
        if kind:
            q += " AND kind=?"; args.append(kind)
        return [self._unpack(r["vec"]) for r in self.conn.execute(q, args)]

    def prune_signatures(self, entity_id: str, kind: str, keep: int) -> None:
        """Keep only the most recent `keep` signatures of a kind (cap, §C)."""
        rows = self.conn.execute(
            "SELECT sig_id FROM signatures WHERE entity_id=? AND kind=? ORDER BY created_at DESC",
            (entity_id, kind),
        ).fetchall()
        to_drop = [r["sig_id"] for r in rows[keep:]]
        if to_drop:
            with self.tx() as c:
                c.executemany("DELETE FROM signatures WHERE sig_id=?", [(s,) for s in to_drop])

    # --- locations -----------------------------------------------------------
    def upsert_location(self, name: str, region: dict, location_id: Optional[str] = None,
                        camera_id: Optional[str] = None) -> str:
        lid = location_id or _uid("loc")
        with self.tx() as c:
            c.execute(
                "INSERT INTO locations(location_id,name,region,camera_id) VALUES (?,?,?,?) "
                "ON CONFLICT(location_id) DO UPDATE SET name=excluded.name, region=excluded.region, "
                "camera_id=excluded.camera_id",
                (lid, name, json.dumps(region), camera_id),
            )
        return lid

    def locations(self, camera_id: Optional[str] = None) -> list[sqlite3.Row]:
        if camera_id is not None:
            return self.conn.execute(
                "SELECT * FROM locations WHERE camera_id=? OR camera_id IS NULL",
                (camera_id,)).fetchall()
        return self.conn.execute("SELECT * FROM locations").fetchall()

    # --- observations --------------------------------------------------------
    def add_observation(self, subject_entity_id: str, predicate: str, *,
                        object_entity_id: Optional[str] = None,
                        location_id: Optional[str] = None,
                        confidence: float = 0.5, source_ref: Optional[str] = None,
                        origin: str = "detector", timestamp: Optional[float] = None,
                        camera_id: Optional[str] = None,
                        user_id: Optional[str] = None,
                        description_id: Optional[str] = None,
                        text: Optional[str] = None
                        ) -> str:
        oid = _uid("obs")
        with self.tx() as c:
            c.execute(
                "INSERT INTO observations(observation_id,subject_entity_id,predicate,"
                "object_entity_id,location_id,timestamp,confidence,source_ref,origin,camera_id,user_id,status,"
                "description_id,text) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (oid, subject_entity_id, predicate, object_entity_id, location_id,
                 timestamp or now(), confidence, source_ref, origin, camera_id, user_id,
                 "new" if origin == "rule" else None, description_id, text),
            )
        return oid

    def observations(self, subject_entity_id: Optional[str] = None,
                     since: Optional[float] = None,
                     camera_id: Optional[str] = None,
                     user_id: Optional[str] = None,
                     until: Optional[float] = None) -> list[sqlite3.Row]:
        q = "SELECT * FROM observations"
        conds, args = [], []
        if subject_entity_id:
            conds.append("subject_entity_id=?"); args.append(subject_entity_id)
        if since is not None:
            conds.append("timestamp>=?"); args.append(since)
        if until is not None:
            conds.append("timestamp<=?"); args.append(until)
        if camera_id is not None:
            conds.append("camera_id=?"); args.append(camera_id)
        if user_id is not None:
            conds.append("user_id=?"); args.append(user_id)
        if conds:
            q += " WHERE " + " AND ".join(conds)
        q += " ORDER BY timestamp"
        return self.conn.execute(q, args).fetchall()

    def get_observation(self, observation_id: str) -> Optional[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM observations WHERE observation_id=?", (observation_id,)).fetchone()

    def last_observation_time(self, subject_entity_id: str, predicate: str,
                             object_entity_id: Optional[str]) -> Optional[float]:
        row = self.conn.execute(
            "SELECT MAX(timestamp) t FROM observations WHERE subject_entity_id=? "
            "AND predicate=? AND IFNULL(object_entity_id,'')=IFNULL(?,'')",
            (subject_entity_id, predicate, object_entity_id),
        ).fetchone()
        return row["t"] if row and row["t"] is not None else None

    # --- scene descriptions (search plan Phase 1) ----------------------------
    def add_scene_description(self, *, model: str, raw: dict, text: str,
                              timestamp: Optional[float] = None,
                              camera_id: Optional[str] = None,
                              location_id: Optional[str] = None,
                              source_ref: Optional[str] = None) -> str:
        """Store one VLM report whole, before any of it is flattened into rows.

        This is the audit copy. If a later phase changes how reports are
        flattened, it can be re-run over these rows; if the flattening ever
        drops something again, `raw` is the evidence that it was there.
        """
        did = _uid("desc")
        with self.tx() as c:
            c.execute(
                "INSERT INTO scene_descriptions(description_id,camera_id,location_id,"
                "timestamp,source_ref,model,raw,text) VALUES (?,?,?,?,?,?,?,?)",
                (did, camera_id, location_id, timestamp or now(), source_ref, model,
                 json.dumps(raw, sort_keys=True), text),
            )
        return did

    def get_scene_description(self, description_id: str) -> Optional[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM scene_descriptions WHERE description_id=?",
            (description_id,)).fetchone()

    def scene_descriptions(self, *, since: Optional[float] = None,
                           until: Optional[float] = None,
                           camera_id: Optional[str] = None) -> list[sqlite3.Row]:
        q = "SELECT * FROM scene_descriptions"
        conds, args = [], []
        if since is not None:
            conds.append("timestamp>=?"); args.append(since)
        if until is not None:
            conds.append("timestamp<=?"); args.append(until)
        if camera_id is not None:
            conds.append("camera_id=?"); args.append(camera_id)
        if conds:
            q += " WHERE " + " AND ".join(conds)
        q += " ORDER BY timestamp"
        return self.conn.execute(q, args).fetchall()

    # --- scene snapshots -----------------------------------------------------
    def add_snapshot(self, location_id: str, present_entity_ids: Sequence[str],
                    timestamp: Optional[float] = None) -> str:
        sid = _uid("snap")
        with self.tx() as c:
            c.execute(
                "INSERT INTO scene_snapshots(snapshot_id,location_id,timestamp,present_entity_ids) "
                "VALUES (?,?,?,?)",
                (sid, location_id, timestamp or now(), json.dumps(list(present_entity_ids))),
            )
        return sid

    def snapshots(self, location_id: Optional[str] = None,
                 since: Optional[float] = None) -> list[sqlite3.Row]:
        q = "SELECT * FROM scene_snapshots"
        conds, args = [], []
        if location_id:
            conds.append("location_id=?"); args.append(location_id)
        if since is not None:
            conds.append("timestamp>=?"); args.append(since)
        if conds:
            q += " WHERE " + " AND ".join(conds)
        q += " ORDER BY timestamp"
        return self.conn.execute(q, args).fetchall()

    def latest_snapshot(self, location_id: str) -> Optional[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM scene_snapshots WHERE location_id=? ORDER BY timestamp DESC LIMIT 1",
            (location_id,),
        ).fetchone()

    # --- relations (weighted, decaying, auditable) ---------------------------
    def find_relation(self, kind: str, subject_entity_id: str, predicate: str,
                     object_entity_id: Optional[str], location_id: Optional[str]
                     ) -> Optional[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM relations WHERE kind=? AND subject_entity_id=? AND predicate=? "
            "AND IFNULL(object_entity_id,'')=IFNULL(?,'') AND IFNULL(location_id,'')=IFNULL(?,'')",
            (kind, subject_entity_id, predicate, object_entity_id, location_id),
        ).fetchone()

    def reinforce_relation(self, kind: str, subject_entity_id: str, predicate: str,
                          *, object_entity_id: Optional[str] = None,
                          location_id: Optional[str] = None,
                          supporting_observation_ids: Sequence[str] = (),
                          obs_confidence: float = 1.0,
                          at: Optional[float] = None) -> str:
        """Create-or-strengthen a weighted edge (§7). Bounded additive update
        scaled by observation confidence; promotes to confirmed past threshold."""
        cfg = CONFIG.distill
        at = at or now()
        inc = cfg.weight_increment * max(0.0, min(1.0, obs_confidence))
        existing = self.find_relation(kind, subject_entity_id, predicate,
                                      object_entity_id, location_id)
        with self.tx() as c:
            if existing and existing["status"] == "suppressed":
                # operator killed this belief; don't let distillation resurrect it
                return existing["relation_id"]
            if existing:
                supp = set(json.loads(existing["supporting_observation_ids"]))
                supp.update(supporting_observation_ids)
                new_w = min(cfg.weight_cap, existing["weight"] + inc)
                status = "confirmed" if new_w >= cfg.confirm_weight else existing["status"]
                c.execute(
                    "UPDATE relations SET weight=?, supporting_observation_ids=?, "
                    "last_reinforced_at=?, status=? WHERE relation_id=?",
                    (new_w, json.dumps(sorted(supp)), at, status, existing["relation_id"]),
                )
                return existing["relation_id"]
            rid = _uid("rel")
            w = min(cfg.weight_cap, inc)
            status = "confirmed" if w >= cfg.confirm_weight else "candidate"
            c.execute(
                "INSERT INTO relations(relation_id,kind,subject_entity_id,predicate,"
                "object_entity_id,location_id,weight,supporting_observation_ids,"
                "last_reinforced_at,created_at,status) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (rid, kind, subject_entity_id, predicate, object_entity_id, location_id,
                 w, json.dumps(sorted(set(supporting_observation_ids))), at, at, status),
            )
            return rid

    def suppress_relation(self, relation_id: str) -> bool:
        """Operator kills one wrong belief (e.g. 'uses potted plant' from mere
        co-occurrence). Marked 'suppressed' + weight 0 so distillation's
        reinforce_relation won't re-mine it. Reversible (row is kept)."""
        with self.tx() as c:
            n = c.execute(
                "UPDATE relations SET status='suppressed', weight=0 WHERE relation_id=?",
                (relation_id,)).rowcount
        return n > 0

    def decay_relations(self, at: Optional[float] = None) -> int:
        """Apply time-based exponential decay to un-reinforced edges (§7).
        Edges that fall below confirm threshold drop back to candidate."""
        at = at or now()
        cfg = CONFIG.distill
        half_life = cfg.decay_half_life_days * 86400.0
        n = 0
        with self.tx() as c:
            for row in c.execute("SELECT relation_id, weight, last_reinforced_at, status FROM relations"):
                dt = at - row["last_reinforced_at"]
                if dt <= 0:
                    continue
                factor = math.pow(0.5, dt / half_life)
                new_w = row["weight"] * factor
                status = row["status"]
                if status == "confirmed" and new_w < cfg.confirm_weight:
                    status = "candidate"
                c.execute("UPDATE relations SET weight=?, status=? WHERE relation_id=?",
                          (new_w, status, row["relation_id"]))
                n += 1
        return n

    def relations(self, subject_entity_id: Optional[str] = None,
                 kind: Optional[str] = None, min_weight: float = 0.0
                 ) -> list[sqlite3.Row]:
        q = "SELECT * FROM relations WHERE weight>=?"
        args: list[Any] = [min_weight]
        if subject_entity_id:
            q += " AND subject_entity_id=?"; args.append(subject_entity_id)
        if kind:
            q += " AND kind=?"; args.append(kind)
        q += " ORDER BY weight DESC"
        return self.conn.execute(q, args).fetchall()

    # --- predicate schema (legal triples, §6) --------------------------------
    def register_predicate(self, predicate: str, subject_type: str = "any",
                          object_type: str = "any", canonical: Optional[str] = None
                          ) -> None:
        with self.tx() as c:
            c.execute(
                "INSERT OR REPLACE INTO predicate_schema(predicate,subject_type,object_type,canonical) "
                "VALUES (?,?,?,?)",
                (predicate, subject_type, object_type, canonical or predicate),
            )

    def canonical_predicate(self, predicate: str) -> str:
        row = self.conn.execute(
            "SELECT canonical FROM predicate_schema WHERE predicate=? LIMIT 1", (predicate,)
        ).fetchone()
        return row["canonical"] if row and row["canonical"] else predicate

    def is_legal_triple(self, subject_type: str, predicate: str, object_type: str) -> bool:
        """A predicate is legal if a matching schema row exists, or if it's
        unregistered (open vocabulary, §5b allows novel predicates on write)."""
        rows = self.conn.execute(
            "SELECT subject_type, object_type FROM predicate_schema WHERE predicate=?",
            (predicate,),
        ).fetchall()
        if not rows:
            return True  # open vocabulary; unknown predicates allowed as observations
        for r in rows:
            st_ok = r["subject_type"] in ("any", subject_type)
            ot_ok = r["object_type"] in ("any", object_type)
            if st_ok and ot_ok:
                return True
        return False

    # --- merge / split / cascade-delete (§7) ---------------------------------
    def merge(self, src_id: str, dst_id: str) -> None:
        """Two entities are actually one: re-point everything from src onto dst,
        mark src merged_into:dst."""
        with self.tx() as c:
            for tbl, col in [("signatures", "entity_id"),
                             ("observations", "subject_entity_id"),
                             ("observations", "object_entity_id"),
                             ("relations", "subject_entity_id"),
                             ("relations", "object_entity_id")]:
                c.execute(f"UPDATE {tbl} SET {col}=? WHERE {col}=?", (dst_id, src_id))
            # snapshots store entity ids inside a json list -> rewrite
            for snap in c.execute("SELECT snapshot_id, present_entity_ids FROM scene_snapshots"):
                ids = json.loads(snap["present_entity_ids"])
                if src_id in ids:
                    ids = sorted(set(dst_id if x == src_id else x for x in ids))
                    c.execute("UPDATE scene_snapshots SET present_entity_ids=? WHERE snapshot_id=?",
                              (json.dumps(ids), snap["snapshot_id"]))
            c.execute("UPDATE entities SET status=? WHERE entity_id=?",
                      (f"merged_into:{dst_id}", src_id))
        self.prune_signatures(dst_id, "face", CONFIG.identity.max_signatures_per_entity)

    def split(self, entity_id: str, sig_ids: Sequence[str],
             observation_ids: Sequence[str] = ()) -> str:
        """One entity is actually two: carve the given signatures/observations
        out into a brand-new entity and re-point them."""
        src = self.get_entity(entity_id)
        new_id = self.create_entity(src["type"])
        with self.tx() as c:
            for sid in sig_ids:
                c.execute("UPDATE signatures SET entity_id=? WHERE sig_id=?", (new_id, sid))
            for oid in observation_ids:
                c.execute("UPDATE observations SET subject_entity_id=? WHERE observation_id=?",
                          (new_id, oid))
        return new_id

    def cascade_delete(self, entity_id: str) -> dict:
        """Remove an entity and ALL derived data (privacy removal + bad-enrollment
        correction are the same mechanism, §7/§11)."""
        counts = {}
        with self.tx() as c:
            counts["signatures"] = c.execute(
                "DELETE FROM signatures WHERE entity_id=?", (entity_id,)).rowcount
            counts["observations"] = c.execute(
                "DELETE FROM observations WHERE subject_entity_id=? OR object_entity_id=?",
                (entity_id, entity_id)).rowcount
            counts["relations"] = c.execute(
                "DELETE FROM relations WHERE subject_entity_id=? OR object_entity_id=?",
                (entity_id, entity_id)).rowcount
            # strip from snapshot inventories
            for snap in c.execute("SELECT snapshot_id, present_entity_ids FROM scene_snapshots"):
                ids = json.loads(snap["present_entity_ids"])
                if entity_id in ids:
                    ids = [x for x in ids if x != entity_id]
                    c.execute("UPDATE scene_snapshots SET present_entity_ids=? WHERE snapshot_id=?",
                              (json.dumps(ids), snap["snapshot_id"]))
            c.execute("UPDATE entities SET status='deleted', label=NULL WHERE entity_id=?",
                      (entity_id,))
        return counts

    def auto_merge_people(self, threshold: Optional[float] = None,
                         keep_labeled: bool = True) -> int:
        """Retroactively merge person entities that are really the same person.

        Two entities are linked if ANY pair of their face signatures is within
        `threshold` cosine — the same rule the online matcher uses, so this just
        repairs fragments where the gap closed only after more signatures
        accumulated. Connected fragments merge into the member with the most
        observations (or a labeled one). Reversible via split. Conservative: same
        false-merge risk profile as normal matching.
        ponytail: O(n^2 * sigs^2) scan; fine at POC scale (tens of entities), swap
        for a vector index if the gallery grows large.
        """
        from collections import defaultdict
        threshold = (CONFIG.identity.face_match_threshold
                     if threshold is None else threshold)
        people = self.list_entities("person")
        sigs = {e["entity_id"]: self.entity_signatures(e["entity_id"], "face")
                for e in people}
        labeled = {e["entity_id"] for e in people if e["label"]}
        ids = [e["entity_id"] for e in people if sigs[e["entity_id"]]]

        parent = {i: i for i in ids}
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]; x = parent[x]
            return x
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a, b = ids[i], ids[j]
                if find(a) == find(b):
                    continue
                best = max((float(np.dot(x, y)) for x in sigs[a] for y in sigs[b]),
                           default=-1.0)
                if best >= threshold:
                    parent[find(a)] = find(b)

        comps: dict[str, list[str]] = defaultdict(list)
        for i in ids:
            comps[find(i)].append(i)

        merged = 0
        for members in comps.values():
            if len(members) < 2:
                continue
            # prefer a labeled entity as the survivor, else the most-observed one
            def rank(e):
                return (keep_labeled and e in labeled, len(self.observations(e)))
            target = max(members, key=rank)
            for m in members:
                if m != target:
                    self.merge(m, target)
                    merged += 1
        return merged
