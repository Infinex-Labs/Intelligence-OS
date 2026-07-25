# HTTP API

Everything the dashboard does, it does over this API — there is no privileged
private channel. It is served by `intelligence_os/web.py` on the stdlib
`http.server`, and it binds `127.0.0.1` by default.

## Authentication

Session cookie, set by `POST /api/auth/login`. Passwords are hashed with
`hashlib.scrypt` and stored as `salt_hex:hash_hex`.

Every route below requires a session except: `GET /static/login.html`, the login
page's assets, `GET /api/auth/status`, `POST /api/auth/register` (first run
only), and `POST /api/auth/login`. Unauthenticated HTML requests redirect to the
login page; unauthenticated `/api/` requests get a JSON error.

**First run**: with no user in the database, `POST /api/auth/register` creates
one. After that it is closed — there is no open sign-up.

```bash
curl -c jar -X POST localhost:8000/api/auth/login \
     -H 'Content-Type: application/json' \
     -d '{"username":"you","password":"..."}'
curl -b jar localhost:8000/api/stats
```

There is **no CSRF token and no rate limiting**. Do not expose this to a network
you don't control — see [deployment](deployment.md).

---

## Auth

| Method | Path | Notes |
|---|---|---|
| `GET` | `/api/auth/status` | Whether a session is valid, and whether this is a first run. |
| `POST` | `/api/auth/register` | First run only. |
| `POST` | `/api/auth/login` | Sets the session cookie. |
| `POST` | `/api/auth/logout` | Clears it. |

## Memory

| Method | Path | Notes |
|---|---|---|
| `GET` | `/api/entities` | All active entities. |
| `GET` | `/api/entity/<id>` | One entity with its observations. |
| `GET` | `/api/entity/<id>/expand` | Neighbourhood for the graph pane. |
| `GET` | `/api/observations` | The raw transcript. Query params filter it. |
| `GET` | `/api/graph` | Nodes and edges for the graph view. |
| `GET` | `/api/stats` | Dashboard counters and the 24-hour histogram. |
| `GET` | `/api/export` | Export the graph. |
| `GET` | `/keyframe/<name>` | A retained keyframe image. Path-traversal checked. |

### Correction

| Method | Path | Body | Notes |
|---|---|---|---|
| `POST` | `/api/entity/<id>/name` | `{"label": "Raj"}` | Name an entity. |
| `POST` | `/api/entity/<id>/merge` | `{"target": "<entity_id>"}` | Merge `<id>` **into** target. Evidence moves; nothing is discarded. |
| `POST` | `/api/entity/<id>/delete` | — | Cascade delete: signatures, observations, relations. Irreversible, by design. |
| `POST` | `/api/relation/<id>/suppress` | — | Suppress one wrong edge without touching its evidence. |
| `GET` | `/api/relation/<id>/evidence` | — | **The provenance route.** Returns the observations and keyframes a belief was built from. Nothing is computed — it already exists. |

## The assistant

See [assistant.md](assistant.md) for what these do and why.

| Method | Path | Body | Notes |
|---|---|---|---|
| `POST` | `/api/ask` | `{"question": "...", "conversation_id": "..."}` | Ask. Appends the turn and returns the answer, its evidence, per-answer counts and refreshed stats. `conversation_id` is optional — omit it to start a thread. |
| `GET` | `/api/chats` | — | Thread list plus KPI stats, in one round trip. |
| `POST` | `/api/chats` | `{"title": "..."}` | Create an empty thread. |
| `GET` | `/api/chats/<id>` | — | One thread's turns, each with its stored evidence payload. |
| `POST` | `/api/chats/<id>/rename` | `{"title": "..."}` | |
| `POST` | `/api/chats/<id>/delete` | — | Turns cascade with it. |

Threads are scoped to the signed-in user, and that scope is a `WHERE` clause on
every read *and* mutation. Someone else's thread returns **404, not 403** — an
id cannot be probed for existence.

## Rules and alerts

| Method | Path | Notes |
|---|---|---|
| `GET` | `/api/rules` | Compiled rules. |
| `POST` | `/api/rules` | Compile an English rule. Needs an API key; may return a refusal. |
| `POST` | `/api/rules/<name>/toggle` | Enable or disable without deleting. |
| `GET` | `/api/alerts` | Fired rule events. Filter by `?status=` and `?camera_id=`. |
| `POST` | `/api/alerts/<observation_id>/status` | Acknowledge or dismiss. A dismissal becomes a negative example that sharpens the rule. |

## Cases and reports

| Method | Path | Notes |
|---|---|---|
| `GET` `POST` | `/api/cases` | List / create. |
| `POST` | `/api/cases/<id>/attach` | Attach an entity, observation or alert. |
| `POST` | `/api/cases/<id>/status` | |
| `GET` `POST` | `/api/reports` | List / generate. |
| `GET` | `/api/report/<id>/download` | The stored body. |

## Cameras, zones, settings

| Method | Path | Notes |
|---|---|---|
| `GET` | `/api/cameras` | Configured cameras and their live state. |
| `POST` | `/api/camera/source` | Change a camera's source. Writes `config.yaml`. |
| `POST` | `/api/camera/toggle` | Pause / resume one camera's thread. |
| `POST` | `/api/camera/test` | Probe a source before committing to it. |
| `POST` | `/api/zones` | Save zone polygons for a camera. |
| `GET` `POST` | `/api/settings` | Read / write `config.yaml` toggles. |
| `POST` | `/api/settings/test` | Test a delivery channel. |
| `GET` | `/api/digest` | The "what broke pattern" briefing. |
| `POST` | `/api/digest/feedback` | Mark a digest item useful or not. |
| `POST` | `/api/baseline/rebuild` | Re-run distillation to rebuild learned habits. |

## Streams

| Path | Notes |
|---|---|
| `/video_feed?cam=<name>` | MJPEG stream. Omit `cam` for the first camera. This is a long-lived response — one connection per viewer per camera. |

---

## Conventions

- Request and response bodies are JSON. Errors are a JSON object with an
  `error` key, or a plain HTTP error for non-`/api/` paths.
- Timestamps are Unix epoch seconds (floats).
- Ids are prefixed strings minted by `_uid()`: `ent_`, `obs_`, `rel_`, `sig_`,
  `loc_`, `snap_`, `case_`, `rep_`, and `chat_` / `turn_` for the assistant.
- Ownership checks live in SQL, not in handlers. If you add a route, put the
  predicate in the query — see [CONTRIBUTING](../CONTRIBUTING.md).
