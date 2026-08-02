import argparse
import json
import mimetypes
import os
import re
import sys
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
from typing import Optional

import cv2
from intelligence_os.store import Store
from intelligence_os.config import (COCO_CLASSES, FRAMES_DIR, DATA_DIR, CONFIG, ROOT,
                                    kind_for_class, normalize_object_classes,
                                    retained_keyframe)
from intelligence_os.run import run as run_pipeline

# Anchored to the package, not the working directory: an installed console
# script starts wherever the user happens to be, and a UI that 404s unless you
# cd to the repo root isn't installable.
STATIC_DIR = str(ROOT / 'static')

# M6: per-camera frame buffers (keyed by camera name)
latest_frames: dict[str, bytes] = {}   # cam_name -> JPEG bytes
frame_lock = threading.Lock()
pipeline_state: dict = {"paused": False, "cameras": {}}

# M7: the store knows these only as objects; the map legend calls them vehicles
VEHICLES = {"car", "truck", "bus", "motorcycle", "bicycle", "train", "boat"}


# A keyframe the browser can actually load — see config.retained_keyframe for why
# the existence check happens at advertise time rather than in the browser.
kf_name = retained_keyframe

def update_frame(img, cam_name: str = "default"):
    """M6: per-camera frame update. run.py passes cam_name via the on_frame callback."""
    with frame_lock:
        ret, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if ret:
            latest_frames[cam_name] = buffer.tobytes()

class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """Handle requests in a separate thread."""
    pass

class RequestHandler(BaseHTTPRequestHandler):
    def _parse_qs(self) -> dict:
        """Quick query string parser."""
        from urllib.parse import urlparse, parse_qs
        return parse_qs(urlparse(self.path).query)

    def check_auth(self) -> Optional[str]:
        """Returns user_id if session is valid, else None."""
        from http.cookies import SimpleCookie
        from intelligence_os.auth import verify_session
        cookie_str = self.headers.get('Cookie')
        if not cookie_str:
            return None
        try:
            cookie = SimpleCookie()
            cookie.load(cookie_str)
            if 'session' in cookie:
                session_id = cookie['session'].value
                store = Store()
                try:
                    return verify_session(store, session_id)
                finally:
                    store.close()
        except Exception:
            pass
        return None

    def is_first_run(self) -> bool:
        """Checks if there are no users registered in the database yet."""
        from intelligence_os.auth import count_users
        store = Store()
        try:
            return count_users(store) == 0
        finally:
            store.close()

    def redirect_to_login(self):
        self.send_response(302)
        self.send_header('Location', '/static/login.html')
        self.end_headers()

    def redirect_to_home(self):
        self.send_response(302)
        self.send_header('Location', '/')
        self.end_headers()

    def send_error_json(self, code: int, message: str):
        self.send_response(code)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps({"error": message}).encode('utf-8'))

    def serve_register(self):
        body = self._read_json()
        username = (body.get('username') or '').strip()
        password = (body.get('password') or '').strip()
        if not username or not password:
            self.send_error_json(400, "Username and password required")
            return
        if len(password) < 8:
            self.send_error_json(400, "Password must be at least 8 characters")
            return
        store = Store()
        try:
            from intelligence_os.auth import create_user, create_session
            user_id = create_user(store, username, password)
            session_id = create_session(store, user_id)
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Set-Cookie', f'session={session_id}; Path=/; HttpOnly; SameSite=Strict; Max-Age=2592000')
            self.end_headers()
            self.wfile.write(json.dumps({"ok": True, "user_id": user_id}).encode('utf-8'))
        except Exception as e:
            self.send_error_json(500, f"Registration failed: {str(e)}")
        finally:
            store.close()

    def serve_login(self):
        body = self._read_json()
        username = (body.get('username') or '').strip()
        password = (body.get('password') or '').strip()
        if not username or not password:
            self.send_error_json(400, "Username and password required")
            return
        store = Store()
        try:
            from intelligence_os.auth import authenticate_user, create_session
            user_id = authenticate_user(store, username, password)
            if not user_id:
                self.send_error_json(401, "Invalid username or password")
                return
            session_id = create_session(store, user_id)
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Set-Cookie', f'session={session_id}; Path=/; HttpOnly; SameSite=Strict; Max-Age=2592000')
            self.end_headers()
            self.wfile.write(json.dumps({"ok": True, "user_id": user_id}).encode('utf-8'))
        except Exception as e:
            self.send_error_json(500, f"Login failed: {str(e)}")
        finally:
            store.close()

    def serve_logout(self):
        from http.cookies import SimpleCookie
        from intelligence_os.auth import delete_session
        cookie_str = self.headers.get('Cookie')
        if cookie_str:
            try:
                cookie = SimpleCookie()
                cookie.load(cookie_str)
                if 'session' in cookie:
                    session_id = cookie['session'].value
                    store = Store()
                    try:
                        delete_session(store, session_id)
                    finally:
                        store.close()
            except Exception:
                pass
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Set-Cookie', 'session=; Path=/; HttpOnly; SameSite=Strict; Max-Age=0')
        self.end_headers()
        self.wfile.write(json.dumps({"ok": True}).encode('utf-8'))

    def do_GET(self):
        path_clean = self.path.split('?', 1)[0]

        # 1. Determine if resource is publicly accessible
        is_public = (
            path_clean in ('/static/login.html', '/static/style.css',
                           # both brand marks: the login screen shows one pre-auth, and a
                           # theme swap must not 302 to the login page mid-render
                           '/static/logo.png', '/static/logo-light.png',
                           '/api/auth/status') or
            path_clean.startswith('/static/vendor/')
        )

        # 2. Onboarding/first-run check
        first_run = self.is_first_run()

        # 3. User session check
        user_id = self.check_auth()

        # 4. Redirect or block unauthenticated clients
        if not user_id and not is_public:
            if self.path.startswith('/api/'):
                self.send_error_json(401, "Unauthorized")
                return
            self.redirect_to_login()
            return

        self.current_user_id = user_id      # M6: settings marks which row is you

        # 5. Prevent logged-in users from seeing login screen
        if user_id and path_clean == '/static/login.html':
            self.redirect_to_home()
            return

        # 6. Route requests
        if self.path == '/':
            # the whole app, graph pane included (FR-NG-6: no separate graph URL)
            self.serve_static(os.path.join(STATIC_DIR, 'home.html'), 'text/html')
        elif self.path == '/api/auth/status':
            username = None
            delivery_schedule = "off"
            delivery_sink = "in_app"
            delivery_destination = ""
            if user_id:
                from intelligence_os.auth import get_user_by_id
                store = Store()
                try:
                    user = get_user_by_id(store, user_id)
                    if user:
                        username = user.get("username")
                        delivery_schedule = user.get("delivery_schedule", "off")
                        delivery_sink = user.get("delivery_sink", "in_app")
                        delivery_destination = user.get("delivery_destination") or ""
                finally:
                    store.close()
            self.send_json({
                "first_run": first_run,
                "authenticated": user_id is not None,
                "username": username,
                "delivery_schedule": delivery_schedule,
                "delivery_sink": delivery_sink,
                "delivery_destination": delivery_destination
            })
        elif self.path == '/api/rules':
            from intelligence_os.rules import load_rules
            self.send_json({"rules": load_rules()})
        elif self.path.startswith('/static/'):
            static_root = os.path.realpath(STATIC_DIR)
            rel = self.path[len('/static/'):].split('?', 1)[0]
            candidate = os.path.realpath(os.path.join(static_root, rel))
            # containment check: reject any path that escapes static_root (../, symlinks)
            if candidate != static_root and not candidate.startswith(static_root + os.sep):
                self.send_error(403, "Forbidden")
                return
            mime_type, _ = mimetypes.guess_type(candidate)
            self.serve_static(candidate, mime_type or 'application/octet-stream')
        elif self.path.split('?', 1)[0] == '/video_feed':
            self.serve_video_feed()
        elif self.path.split('?', 1)[0] == '/snapshot':
            self.serve_snapshot()
        elif self.path == '/api/cameras':
            self.serve_cameras()
        elif self.path.split('?', 1)[0] == '/api/zones':
            self.serve_zones()
        elif self.path.split('?', 1)[0] == '/api/map':
            self.serve_map()
        elif self.path == '/api/settings':
            self.serve_settings_get()
        elif self.path == '/api/export':
            self.serve_export()
        elif self.path.split('?', 1)[0] == '/api/alerts':
            self.serve_alerts()
        elif self.path.split('?', 1)[0] == '/api/cases':
            self.serve_cases()
        elif self.path.split('?', 1)[0] == '/api/reports':
            self.serve_reports()
        elif self.path.startswith('/api/report/') and self.path.endswith('/download'):
            self.serve_report_download(self.path[len('/api/report/'):-len('/download')])
        elif self.path.split('?', 1)[0] == '/api/observations':
            self.serve_observations()
        elif self.path.startswith('/api/relation/') and self.path.endswith('/evidence'):
            self.serve_relation_evidence(self.path[len('/api/relation/'):-len('/evidence')])
        elif self.path.startswith('/keyframe/'):
            self.serve_keyframe(self.path[len('/keyframe/'):].split('?', 1)[0])
        elif self.path.startswith('/api/entity/') and self.path.split('?', 1)[0].endswith('/expand'):
            self.serve_entity_expand(self.path[len('/api/entity/'):].split('?', 1)[0][:-len('/expand')])
        elif self.path.startswith('/api/entity/'):
            self.serve_entity(self.path[len('/api/entity/'):].split('?', 1)[0])
        elif self.path == '/api/entities':
            self.serve_entities()
        elif self.path == '/api/graph':
            self.serve_graph()
        elif path_clean == '/api/chats':
            self.serve_chats()
        elif path_clean.startswith('/api/chats/'):
            self.serve_chat(path_clean[len('/api/chats/'):])
        elif self.path == '/api/stats':
            self.serve_stats()
        elif self.path.startswith('/api/digest'):
            self.serve_digest()
        else:
            self.send_error(404, "File not found")

    def do_POST(self):
        first_run = self.is_first_run()
        user_id = self.check_auth()
        path_clean = self.path.split('?', 1)[0]

        # Public post routes
        if path_clean == '/api/auth/register':
            if not first_run:
                self.send_error_json(403, "Onboarding already completed")
                return
            self.serve_register()
            return
        elif path_clean == '/api/auth/login':
            if first_run:
                self.send_error_json(400, "Onboarding required")
                return
            self.serve_login()
            return

        # Protected post routes
        if not user_id:
            self.send_error_json(401, "Unauthorized")
            return

        # Attach current authenticated user id to handler instance for attribution
        self.current_user_id = user_id

        if path_clean == '/api/auth/logout':
            self.serve_logout()
        elif path_clean == '/api/camera/toggle':
            self.serve_camera_toggle()
        elif self.path == '/api/ask':
            self.serve_ask()
        elif path_clean == '/api/chats':
            self.serve_chat_create()
        elif path_clean.startswith('/api/chats/') and path_clean.endswith('/rename'):
            self.serve_chat_rename(path_clean[len('/api/chats/'):-len('/rename')])
        elif path_clean.startswith('/api/chats/') and path_clean.endswith('/delete'):
            self.serve_chat_delete(path_clean[len('/api/chats/'):-len('/delete')])
        elif self.path == '/api/digest/feedback':
            self.serve_digest_feedback()
        elif self.path.startswith('/api/rules/') and self.path.endswith('/toggle'):
            self.serve_rule_toggle(self.path[len('/api/rules/'):-len('/toggle')])
        elif self.path == '/api/rules':
            self.serve_rules_compile()
        elif self.path == '/api/zones':
            self.serve_zone_create()
        elif self.path == '/api/camera/source':
            self.serve_camera_source()
        elif self.path == '/api/camera/test':
            self.serve_camera_test()
        elif self.path == '/api/settings/test':
            self.serve_delivery_test()
        elif self.path == '/api/baseline/rebuild':
            self.serve_rebuild_baseline()
        elif self.path == '/api/settings':
            self.serve_settings()
        elif path_clean == '/api/cases':
            self.serve_case_create()
        elif path_clean.startswith('/api/cases/') and path_clean.endswith('/attach'):
            self.serve_case_attach(path_clean[len('/api/cases/'):-len('/attach')])
        elif path_clean.startswith('/api/cases/') and path_clean.endswith('/status'):
            self.serve_case_status(path_clean[len('/api/cases/'):-len('/status')])
        elif path_clean == '/api/reports':
            self.serve_report_create()
        elif path_clean.startswith('/api/alerts/') and path_clean.endswith('/status'):
            self.serve_alert_status(path_clean[len('/api/alerts/'):-len('/status')])
        elif self.path.startswith('/api/entity/'):
            self.serve_entity_action(self.path[len('/api/entity/'):])
        elif self.path.startswith('/api/relation/'):
            self.serve_relation_action(self.path[len('/api/relation/'):])
        else:
            self.send_error(404, "File not found")

    def serve_relation_evidence(self, relation_id):
        """Provenance (§10 NFR-6): the observations + keyframes that produced a
        belief. Data already exists — supporting_observation_ids -> observations
        -> source_ref keyframe. Nothing new computed."""
        store = Store()
        try:
            row = store.conn.execute(
                "SELECT * FROM relations WHERE relation_id=?", (relation_id,)).fetchone()
            if row is None:
                self.send_error(404, "Relation not found")
                return
            obs_ids = json.loads(row["supporting_observation_ids"] or "[]")
            observations, keyframes = [], []
            seen_kf = set()
            for oid in obs_ids[:40]:
                o = store.conn.execute(
                    "SELECT * FROM observations WHERE observation_id=?", (oid,)).fetchone()
                if o is None:
                    continue
                observations.append({
                    "predicate": o["predicate"],
                    "timestamp": o["timestamp"],
                    "origin": o["origin"],
                    "confidence": o["confidence"],
                })
                # source_ref is an absolute keyframe path; expose only its
                # basename, and only if retention has not reclaimed the file
                name = kf_name(o["source_ref"])
                if name and name not in seen_kf:
                    seen_kf.add(name)
                    keyframes.append("/keyframe/" + name)
            self.send_json({
                "predicate": row["predicate"],
                "status": row["status"],
                "weight": round(row["weight"], 2),
                "evidence_count": len(obs_ids),
                "observations": observations,
                "keyframes": keyframes,   # pruned ones are omitted, not 404s
            })
        finally:
            store.close()

    def serve_keyframe(self, name):
        """Serve a retained audit keyframe by basename, contained to FRAMES_DIR."""
        root = os.path.realpath(str(FRAMES_DIR))
        candidate = os.path.realpath(os.path.join(root, name))
        if candidate != root and not candidate.startswith(root + os.sep):
            self.send_error(403, "Forbidden")
            return
        self.serve_static(candidate, 'image/jpeg')

    def serve_relation_action(self, rest):
        """Suppress one wrong edge (§7 correctability, Step 4b)."""
        parts = rest.split('/', 1)
        if len(parts) != 2 or parts[1] != 'suppress':
            self.send_error(404, "Unknown action")
            return
        store = Store()
        try:
            if store.suppress_relation(parts[0]):
                self.send_json({"ok": True, "suppressed": parts[0]})
            else:
                self.send_error(404, "Relation not found")
        finally:
            store.close()

    def serve_digest(self):
        """Three-band digest (§8.1): rule_fired / unusual / routine-collapsed.
        ?since=<epoch> or ?hours=<n> (default 24h)."""
        from urllib.parse import urlparse, parse_qs
        import time as _t
        from intelligence_os.digest import build
        q = parse_qs(urlparse(self.path).query)
        try:
            since = (float(q["since"][0]) if "since" in q
                     else _t.time() - float(q.get("hours", ["24"])[0]) * 3600)
        except ValueError:
            self.send_error(400, "bad since/hours")
            return
        store = Store()
        try:
            self.send_json(build(store, since=since))
        finally:
            store.close()

    def serve_digest_feedback(self):
        """Triage a digest item: {'signature','entity_id','action':'dismiss'|'confirm'}.
        The training signal for learned salience (§8.1, §13)."""
        from intelligence_os.digest import feedback
        body = self._read_json()
        sig = (body.get('signature') or '').strip()
        eid = (body.get('entity_id') or '').strip()
        action = (body.get('action') or '').strip()
        store = Store()
        try:
            if not sig or store.get_entity(eid) is None:
                self.send_error(400, "signature and a valid entity_id required")
                return
            try:
                oid = feedback(store, sig, eid, action, user_id=getattr(self, 'current_user_id', None))
            except ValueError as e:
                self.send_error(400, str(e))
                return
            self.send_json({"ok": True, "observation_id": oid})
        finally:
            store.close()

    def serve_rules_compile(self):
        """Compile an English rule (§9.2). Returns the spec, or the refusal —
        verdicts map to the UI: compiled=pass, infeasible=block, needs_identity/
        no_key=warn. A saved rule arms within a couple of seconds — the engine
        polls rules.yaml (RuleEngine._maybe_reload); no restart needed."""
        from intelligence_os.rules import compile_rule, save_rule, Refusal
        text = (self._read_json().get('text') or '').strip()
        if not text:
            self.send_error(400, "text required")
            return
        result = compile_rule(text)
        if isinstance(result, Refusal):
            self.send_json({"refused": True, "reason": result.reason,
                            "message": result.message})
            return
        save_rule(result)
        self.send_json({"refused": False, "rule": result})

    def serve_snapshot(self):
        """M6: a single latest JPEG frame. ?cam=<name> selects the camera."""
        qs = self._parse_qs()
        cam = (qs.get('cam', [None])[0] or
               next(iter(latest_frames), None))
        with frame_lock:
            buf = latest_frames.get(cam) if cam else None
        if buf is None:
            self.send_error(503, "no frame yet - is the camera running?")
            return
        self.send_response(200)
        self.send_header('Content-type', 'image/jpeg')
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(buf)

    def serve_cameras(self):
        """M6: GET /api/cameras — list all cameras with health info.

        Cameras configured but not yet started (added since the last restart)
        are listed too, as `up: false` — hiding them would make the add form
        look broken."""
        cameras_state = pipeline_state.get("cameras", {})
        now_ts = time.time()
        store = Store()
        try:
            zones = {}
            for l in store.locations():
                zones.setdefault(l["camera_id"], []).append(l["name"])
        finally:
            store.close()

        from intelligence_os.config import resolve_cameras
        try:
            configured = {c["name"]: str(c["source"]) for c in resolve_cameras()}
        except ValueError:
            configured = {}

        out = []
        for name in list(cameras_state) + [n for n in configured if n not in cameras_state]:
            cs = cameras_state.get(name, {})
            last_ts = cs.get("last_frame_ts")
            out.append({
                "name": name,
                "source": cs.get("source") or configured.get(name),
                "zones": zones.get(name, []),
                "up": last_ts is not None and (now_ts - last_ts) < 10,
                "running": name in cameras_state,
                "last_frame_ts": last_ts,
                "uptime_s": round(now_ts - last_ts, 1) if last_ts else None,
                "is_stream": cs.get("is_stream", False),
                "paused": cs.get("paused", False),
                # A crashed camera reports why, instead of looking merely idle.
                "error": cs.get("error"),
                "stopped_at": cs.get("stopped_at"),
            })
        self.send_json({"cameras": out, "unassigned_zones": zones.get(None, [])})

    def serve_camera_toggle(self):
        """M6: toggle pause for a specific camera or all. ?cam=<name> optional."""
        qs = self._parse_qs()
        cam = qs.get('cam', [None])[0]
        cameras_state = pipeline_state.get("cameras", {})
        if cam:
            cs = cameras_state.get(cam)
            if cs is None:
                self.send_error(404, f"Camera '{cam}' not found")
                return
            cs["paused"] = not cs.get("paused", False)
            self.send_json({"name": cam, "paused": cs["paused"]})
        else:
            # toggle all cameras
            results = {}
            for name, cs in cameras_state.items():
                cs["paused"] = not cs.get("paused", False)
                results[name] = cs["paused"]
            self.send_json({"paused": results})

    def serve_zones(self):
        """Existing zones (name + polygon in frame coords), for the wizard + overlay."""
        cam = self._parse_qs().get('cam', [None])[0]
        store = Store()
        try:
            out = []
            for l in store.locations(camera_id=cam):
                region = json.loads(l["region"] or "{}")
                out.append({"id": l["location_id"], "name": l["name"],
                            "camera": l["camera_id"],
                            "polygon": region.get("polygon", [])})
            self.send_json({"zones": out})
        finally:
            store.close()

    def serve_map(self):
        """V4-M7. This product has no site geometry — a zone is a polygon in one
        camera's frame, not a place on a plan — so the map is the cameras' own
        views with their zones drawn on them, and every entity plotted on the
        zone it was last seen in. ?hours= sets the "live" window.

        ponytail: no floorplan, no geo tiles. Add a real plan when a camera row
        carries a site x,y — until then any layout would be invented."""
        try:
            hours = max(1.0, min(720.0, float(self._parse_qs().get('hours', ['24'])[0])))
        except ValueError:
            hours = 24.0
        since = time.time() - hours * 3600

        from intelligence_os.config import resolve_cameras
        from intelligence_os.rules import load_rules
        try:
            cam_names = [c["name"] for c in resolve_cameras()]
        except ValueError:
            cam_names = []
        for name in pipeline_state.get("cameras", {}):
            if name not in cam_names:
                cam_names.append(name)
        # legacy rows predate multi-camera: with one camera they can only be its own
        legacy_cam = cam_names[0] if len(cam_names) == 1 else None

        store = Store()
        try:
            zones, zone_by_id = {}, {}
            for l in store.locations():
                poly = json.loads(l["region"] or "{}").get("polygon", [])
                z = {"id": l["location_id"], "name": l["name"], "polygon": poly,
                     "camera": l["camera_id"],
                     "cx": sum(p[0] for p in poly) / len(poly) if poly else None,
                     "cy": sum(p[1] for p in poly) / len(poly) if poly else None}
                zones.setdefault(l["camera_id"], []).append(z)
                zone_by_id[z["id"]] = z

            ents = {e["entity_id"]: e for e in store.list_entities(active_only=False)}
            sev_of = {r.get("name"): (r.get("severity") or "medium").lower()
                      for r in load_rules()}
            last, alerts = {}, []
            for o in store.observations(since=since):     # ascending: last write wins
                cam = o["camera_id"] or legacy_cam
                ent = ents.get(o["subject_entity_id"])
                # same fallback the entity cards use: a bare id names nothing
                label = (ent["label"] if ent and ent["label"] else
                         f"{(ent['type'] if ent else 'entity').title()} "
                         f"{o['subject_entity_id'][-6:]}")
                kind = ("vehicle" if ent and (ent["label"] or "") in VEHICLES
                        else ent["type"] if ent else "object")
                if o["origin"] == 'rule' and (o["status"] or 'new') != 'resolved':
                    rule = o["predicate"].removeprefix("rule_fired:")
                    alerts.append({"kind": "alert", "id": o["observation_id"],
                                   "label": rule.replace('_', ' '),
                                   "entity_id": o["subject_entity_id"],
                                   "severity": sev_of.get(rule, "medium"),
                                   "camera": cam, "zone": o["location_id"],
                                   "timestamp": o["timestamp"]})
                last[o["subject_entity_id"]] = {
                    "kind": kind, "id": o["subject_entity_id"], "label": label,
                    "entity_id": o["subject_entity_id"], "camera": cam,
                    "zone": o["location_id"], "timestamp": o["timestamp"]}
        finally:
            store.close()

        markers = sorted(list(last.values()) + alerts, key=lambda m: m["timestamp"])
        for m in markers:
            z = zone_by_id.get(m["zone"])
            m["zone_name"] = z["name"] if z else None
            if z and z["camera"] and z["camera"] != m["camera"]:
                m["camera"] = z["camera"]   # the zone knows better than a legacy row
        cams = [{"name": n, "zones": zones.get(n, []),
                 "up": (lambda t: t is not None and time.time() - t < 10)(
                     pipeline_state.get("cameras", {}).get(n, {}).get("last_frame_ts"))}
                for n in cam_names]
        self.send_json({"cameras": cams, "markers": markers,
                        "unassigned_zones": zones.get(None, []), "hours": hours})

    def serve_zone_create(self):
        """Setup wizard (§8.4): save a drawn polygon as a named zone. Frame coords.
        Arms on the next pipeline start (like rules)."""
        body = self._read_json()
        name = (body.get('name') or '').strip()
        poly = body.get('polygon')
        if not name or not isinstance(poly, list) or len(poly) < 3:
            self.send_error(400, "name and a polygon of >=3 points required")
            return
        try:
            clean = [[float(p[0]), float(p[1])] for p in poly]
        except (TypeError, ValueError, IndexError):
            self.send_error(400, "polygon must be [[x,y], ...]")
            return
        cam = (body.get('camera') or '').strip() or None
        store = Store()
        try:
            lid = store.upsert_location(name, {"polygon": clean}, camera_id=cam)
            self.send_json({"id": lid, "name": name, "camera": cam})
        finally:
            store.close()

    def serve_camera_source(self):
        """M6: add or remove a camera in config.yaml (§10). Applies on the next
        start — we don't hot-swap a running capture, and pretending otherwise
        would leave the table disagreeing with the pipeline."""
        from intelligence_os.config import load_app_config, resolve_cameras, update_app_config
        body = self._read_json()
        cfg = load_app_config()
        cams = resolve_cameras(cfg)

        remove = (body.get('remove') or '').strip()
        if remove:
            left = [c for c in cams if c["name"] != remove]
            if len(left) == len(cams):
                self.send_error(404, f"No camera named '{remove}'")
                return
            update_app_config(cameras=left, camera=None)
            self.send_json({"cameras": left, "note": "applies on next start"})
            return

        src = str(body.get('source') or '').strip()
        name = (body.get('name') or '').strip()
        if not src or not name:
            self.send_error(400, "source and name required")
            return
        source = int(src) if src.isdigit() else src
        # nothing configured yet → resolve_cameras' webcam-0 default, not a real list
        keep = cams if (cfg.get("cameras") or "camera" in cfg) else []
        existing = next((c for c in keep if c["name"] == name), None)
        if existing:
            existing["source"] = source          # same name → repoint, don't reject
        else:
            keep = keep + [{"name": name, "source": source}]
        update_app_config(cameras=keep, camera=None)

        # Hot-start a genuinely new camera against the running pipeline. Repointing
        # an already-running camera still needs a restart — we can't yank the open
        # capture out from under its thread mid-stream.
        spawn = pipeline_state.get("_spawn")
        already_running = name in pipeline_state.get("cameras", {})
        if spawn and not already_running:
            spawn({"name": name, "source": source})
            note = "started"
        else:
            note = "applies on next start"
        self.send_json({"cameras": keep, "note": note})

    def serve_camera_test(self):
        """FR-CM-2 'Test connection': open the source, grab one frame, hang up."""
        src = str(self._read_json().get('source') or '').strip()
        if not src:
            self.send_error(400, "source required")
            return
        if src.lower().startswith('rtsp'):        # default UDP drops over open internet
            os.environ.setdefault('OPENCV_FFMPEG_CAPTURE_OPTIONS', 'rtsp_transport;tcp')
        cap = cv2.VideoCapture(int(src) if src.isdigit() else src)
        try:
            ok, frame = False, None
            if cap.isOpened():
                # RTSP buffers the first frame after the handshake — one instant read
                # usually returns False on a perfectly good stream. Retry briefly.
                import time as _t
                for _ in range(40):            # ponytail: ~4s ceiling; bump if slow cameras miss
                    ok, frame = cap.read()
                    if ok:
                        break
                    _t.sleep(0.1)
        finally:
            cap.release()
        if not ok:
            self.send_json({"ok": False, "detail": "could not read a frame"})
            return
        h, w = frame.shape[:2]
        self.send_json({"ok": True, "detail": f"{w}×{h}"})

    def serve_settings_get(self):
        """M6: everything the settings pane renders, in one round trip.
        The key itself is never sent back — only whether one is set."""
        store = Store()
        try:
            team = [{"username": r["username"],
                     "schedule": r["delivery_schedule"],
                     "sink": r["delivery_sink"],
                     "destination": r["delivery_destination"] or "",
                     "you": r["user_id"] == self.current_user_id}
                    for r in store.conn.execute(
                        "SELECT * FROM users ORDER BY created_at").fetchall()]
            # FR-ST-4: "reasoning usage today" = observations the reasoner wrote
            since = time.time() - 86400
            calls = store.conn.execute(
                "SELECT COUNT(*) FROM observations WHERE origin='reasoner' "
                "AND timestamp >= ?", (since,)).fetchone()[0]
        finally:
            store.close()
        self.send_json({
            "retention_days": CONFIG.raw_retention_days,
            "data_dir": str(DATA_DIR),
            "face_matching": CONFIG.identity.enabled,
            "anthropic_key_set": bool(os.environ.get("ANTHROPIC_API_KEY")),
            "telegram_token_set": bool(os.environ.get("TELEGRAM_BOT_TOKEN")),
            "vlm_enabled": CONFIG.vlm.enabled,
            "reasoning_calls_today": calls,
            "team": team,
            # What the detector is currently keeping. `person` is always on and is
            # deliberately absent from the list — it is not a togglable class.
            "object_classes": list(CONFIG.detect.object_classes),
            "available_classes": [c for c in COCO_CLASSES if c != "person"],
            "class_kinds": {c: kind_for_class(c) for c in COCO_CLASSES
                            if c != "person"},
        })

    def serve_delivery_test(self):
        """A channel you can't verify is a support ticket: send one real message
        through whatever sink the user has configured."""
        from intelligence_os.deliver import send_email, send_telegram, send_webhook
        store = Store()
        try:
            u = store.conn.execute("SELECT * FROM users WHERE user_id=?",
                                   (self.current_user_id,)).fetchone()
        finally:
            store.close()
        if u is None:
            self.send_error(404, "no such user")
            return
        sink, dest = u["delivery_sink"], (u["delivery_destination"] or "").strip()
        text = "Intelligence OS test message — delivery is configured correctly."
        if sink == "in_app":
            ok, detail = True, "in-app only — nothing to send"
        elif not dest:
            ok, detail = False, "set a destination first"
        elif sink == "telegram":
            ok = send_telegram(dest, text)
            detail = "sent" if ok else "Telegram refused it — check the bot token and chat id"
        elif sink == "webhook":
            ok = send_webhook(dest, {"event": "test", "text": text})
            detail = "sent" if ok else "the webhook did not accept it"
        else:
            ok = send_email(dest, "Intelligence OS test", f"<p>{text}</p>", text, [])
            detail = "sent" if ok else "SMTP refused it"
        self.send_json({"ok": ok, "detail": detail})

    def serve_export(self):
        """FR-ST-6 Export memory: a consistent copy of the whole memory DB.
        sqlite's own backup API, so it is safe to take while the pipeline writes."""
        import sqlite3, tempfile
        store = Store()
        try:
            with tempfile.NamedTemporaryFile(suffix=".db") as tmp:
                dst = sqlite3.connect(tmp.name)
                with dst:
                    store.conn.backup(dst)
                dst.close()
                body = open(tmp.name, "rb").read()
        finally:
            store.close()
        stamp = time.strftime("%Y%m%d-%H%M")
        self.send_response(200)
        self.send_header('Content-type', 'application/octet-stream')
        self.send_header('Content-Disposition',
                         f'attachment; filename="memory-{stamp}.db"')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def serve_rebuild_baseline(self):
        """FR-ST-6 Rebuild baseline: one distillation pass over existing memory.
        ponytail: runs inline — it takes seconds on this scale; make it a thread
        if a big DB starts blocking the request."""
        from intelligence_os.distill import Distiller
        store = Store()
        try:
            self.send_json(Distiller(store).run())
        finally:
            store.close()

    @staticmethod
    def _set_env(key, value):
        """Secrets live in intelligence_os/.env, never in config.yaml (which is meant to
        be version-controlled) and never in the DB (which gets exported)."""
        from intelligence_os.config import ROOT
        path = str(ROOT / '.env')      # not CWD-relative: the server is started from anywhere
        lines = [l for l in (open(path).readlines() if os.path.exists(path) else [])
                 if not l.startswith(key + '=')]
        lines.append(f'{key}={value}\n')
        with open(path, 'w') as f:
            f.writelines(lines)
        os.environ[key] = value

    def serve_settings(self):
        """Settings pane: persist settings like Anthropic API key to .env and user delivery settings to DB (PLAN M9.1)."""
        body = self._read_json()
        anthropic_key = body.get('anthropic_key')
        delivery_schedule = body.get('delivery_schedule')
        delivery_sink = body.get('delivery_sink')
        delivery_destination = body.get('delivery_destination')

        # 1. Update secrets if supplied
        if anthropic_key is not None:
            self._set_env('ANTHROPIC_API_KEY', anthropic_key)
            CONFIG.vlm.enabled = True   # module-level import: a local one shadows it
        if body.get('telegram_token') is not None:
            self._set_env('TELEGRAM_BOT_TOKEN', body['telegram_token'].strip())

        # 2. Update user delivery settings if supplied
        store = Store()
        try:
            if delivery_schedule is not None or delivery_sink is not None or delivery_destination is not None:
                with store.tx() as c:
                    if delivery_schedule is not None:
                        c.execute("UPDATE users SET delivery_schedule = ? WHERE user_id = ?", (delivery_schedule.strip(), self.current_user_id))
                    if delivery_sink is not None:
                        c.execute("UPDATE users SET delivery_sink = ? WHERE user_id = ?", (delivery_sink.strip(), self.current_user_id))
                    if delivery_destination is not None:
                        c.execute("UPDATE users SET delivery_destination = ? WHERE user_id = ?", (delivery_destination.strip(), self.current_user_id))
        finally:
            store.close()

        # 3. M6: retention + face matching live in config.yaml, so they survive a restart
        from intelligence_os.config import update_app_config
        app = {}
        if body.get('retention_days') is not None:
            app["retention_days"] = max(1, int(body['retention_days']))
        if body.get('face_matching') is not None:
            app["face_matching"] = bool(body['face_matching'])
        rejected = []
        if body.get('object_classes') is not None:
            # A class the model does not carry can never fire. Say which ones were
            # dropped rather than saving a list that quietly watches less than asked.
            kept, rejected = normalize_object_classes(body['object_classes'])
            kept = [c for c in kept if c != "person"]
            if kept:
                app["object_classes"] = kept
        if app:
            update_app_config(**app)

        self.send_json({"ok": True, "retention_days": CONFIG.raw_retention_days,
                        "face_matching": CONFIG.identity.enabled,
                        "object_classes": list(CONFIG.detect.object_classes),
                        "rejected_classes": rejected,
                        # The detector reads its class filter once, at construction
                        # (detect.py). Unlike rules, this one really does wait.
                        "classes_need_restart": bool(app.get("object_classes"))})

    def serve_rule_toggle(self, name):
        """Enable/disable a rule (M5). Flips the flag; the engine picks it up
        within RuleEngine.reload_poll_s seconds."""
        from intelligence_os.rules import load_rules, set_rule_enabled
        cur = next((r for r in load_rules() if r.get("name") == name), None)
        if cur is None:
            self.send_error(404, "Rule not found")
            return
        new = not cur.get("enabled", True)
        set_rule_enabled(name, new)
        self.send_json({"name": name, "enabled": new})

    @staticmethod
    def _drop_pruned_keyframes(result: dict) -> None:
        """Strip links to frames retention has already reclaimed.

        Done here rather than in ask.execute(): whether a file is on disk is a
        serving concern, and execute() runs this per observation row in the query
        hot path. Counts are taken after this, so the UI never reports a keyframe
        it cannot render.
        """
        for e in result.get('entities') or []:
            e['keyframes'] = [u for u in (e.get('keyframes') or [])
                              if retained_keyframe(u)]
            for ev in e.get('rule_events') or []:
                if ev.get('keyframe') and not retained_keyframe(ev['keyframe']):
                    ev['keyframe'] = None

    @staticmethod
    def _ask_counts(result: dict) -> dict:
        """The four numbers every answer is measured by, counted off the evidence
        itself — nothing here is a running total the UI could drift from."""
        ents = result.get('entities') or []
        return {
            "entities": len(ents),
            "observations": int(result.get('total_observations') or 0),
            "keyframes": sum(len(e.get('keyframes') or []) for e in ents),
            "rule_events": sum(len(e.get('rule_events') or []) for e in ents),
        }

    def serve_ask(self):
        """Grounded query (§8.2): LLM parses the question, every fact in the
        response is aggregated from observation rows. Query trace included.

        The turn is appended to a conversation (V4-M10) so the thread survives a
        reload, and the follow-up history is read back out of that thread rather
        than trusted from the client."""
        body = self._read_json()
        question = (body.get('question') or '').strip()
        if not question:
            self.send_error(400, "question required")
            return
        user_id = getattr(self, 'current_user_id', None)
        cid = (body.get('conversation_id') or '').strip() or None
        store = Store()
        try:
            # deleted while the question was in flight, or never this operator's
            # to append to — either way the answer opens a thread of their own
            if cid and not store.get_conversation(cid, user_id):
                cid = None
            fresh = cid is None
            if fresh:
                cid = store.create_conversation(user_id=user_id)
            history = [{"q": t["question"], "a": t["answer"] or ""}
                       for t in store.conversation_turns(cid)]
            from intelligence_os.ask import ask
            started = time.time()
            try:
                result = ask(question, store=store, history=history)
            except RuntimeError as e:
                # nothing was answered, so don't leave a titleless empty thread
                # sitting in the sidebar — only the thread we just opened goes
                if fresh:
                    store.delete_conversation(cid)
                    cid = None
                self.send_json({"error": str(e), "conversation_id": cid})
                return
            result["conversation_id"] = cid
            result["latency_ms"] = round((time.time() - started) * 1000)
            self._drop_pruned_keyframes(result)
            result["counts"] = self._ask_counts(result)
            # stored as rendered: reopening the thread replays this answer, it
            # does not re-run the query against a memory that has moved on
            turn_id = store.add_chat_turn_owned(
                cid, user_id, question, result.get("answer"), json.dumps(result),
                result["latency_ms"], counts=result["counts"])
            row = store.get_conversation(cid, user_id)
            result["turn_id"] = turn_id
            result["conversation_title"] = row["title"] if row else None
            result["stats"] = store.chat_stats(user_id)
            self.send_json(result)
        finally:
            store.close()

    def serve_chats(self):
        """Sidebar payload: the thread list plus the KPI aggregate, one round trip."""
        user_id = getattr(self, 'current_user_id', None)
        store = Store()
        try:
            self.send_json({
                "conversations": [dict(r) for r in store.conversations(user_id)],
                "stats": store.chat_stats(user_id),
            })
        finally:
            store.close()

    def serve_chat(self, conversation_id):
        """One thread, with each turn's stored evidence payload re-attached."""
        store = Store()
        try:
            row = store.get_conversation(conversation_id,
                                         getattr(self, 'current_user_id', None))
            if not row:
                self.send_error_json(404, "No such conversation")
                return
            turns = []
            for t in store.conversation_turns(conversation_id):
                d = dict(t)
                try:
                    d["payload"] = json.loads(t["payload"]) if t["payload"] else None
                except ValueError:
                    d["payload"] = None      # a truncated row must not break the thread
                turns.append(d)
            self.send_json({"conversation": dict(row), "turns": turns})
        finally:
            store.close()

    def serve_chat_create(self):
        store = Store()
        try:
            cid = store.create_conversation(
                (self._read_json().get('title') or '').strip() or None,
                user_id=getattr(self, 'current_user_id', None))
            self.send_json({"conversation": dict(store.get_conversation(cid))})
        finally:
            store.close()

    def serve_chat_rename(self, conversation_id):
        title = (self._read_json().get('title') or '').strip()
        if not title:
            self.send_error_json(400, "title required")
            return
        user_id = getattr(self, 'current_user_id', None)
        store = Store()
        try:
            # scoped: another operator's thread is a 404, not a rename
            if not store.rename_conversation(conversation_id, title, user_id):
                self.send_error_json(404, "No such conversation")
                return
            self.send_json({"conversation":
                            dict(store.get_conversation(conversation_id, user_id))})
        finally:
            store.close()

    def serve_chat_delete(self, conversation_id):
        user_id = getattr(self, 'current_user_id', None)
        store = Store()
        try:
            if not store.delete_conversation(conversation_id, user_id):
                self.send_error_json(404, "No such conversation")
                return
            self.send_json({"deleted": conversation_id,
                            "stats": store.chat_stats(user_id)})
        finally:
            store.close()

    def _read_json(self):
        length = int(self.headers.get('Content-Length', 0) or 0)
        if not length:
            return {}
        try:
            return json.loads(self.rfile.read(length).decode('utf-8'))
        except (ValueError, UnicodeDecodeError):
            return {}

    def serve_entity_action(self, rest):
        """Correction primitives (§I): name / merge / cascade-delete.
        Localhost single-operator tool — same surface as `operator` CLI, no auth."""
        try:
            entity_id, action = rest.split('/', 1)
        except ValueError:
            self.send_error(404, "Not found")
            return
        store = Store()
        try:
            if store.get_entity(entity_id) is None:
                self.send_error(404, "Entity not found")
                return
            if action == 'name':
                label = (self._read_json().get('label') or '').strip()
                if not label:
                    self.send_error(400, "label required")
                    return
                store.set_label(entity_id, label)
                self.send_json({"ok": True, "id": entity_id, "label": label})
            elif action == 'merge':
                target = (self._read_json().get('target') or '').strip()
                if store.get_entity(target) is None:
                    self.send_error(400, "target entity not found")
                    return
                if target == entity_id:
                    self.send_error(400, "cannot merge an entity into itself")
                    return
                store.merge(entity_id, target)   # src merged into target
                self.send_json({"ok": True, "merged_into": target})
            elif action == 'delete':
                counts = store.cascade_delete(entity_id)
                self.send_json({"ok": True, "deleted": counts})
            else:
                self.send_error(404, "Unknown action")
        finally:
            store.close()

    def serve_static(self, filepath, mime_type):
        if not os.path.exists(filepath):
            self.send_error(404, "File not found")
            return
        with open(filepath, 'rb') as f:
            content = f.read()
        self.send_response(200)
        self.send_header('Content-type', mime_type)
        self.send_header('Content-length', str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def serve_video_feed(self):
        """M6: MJPEG stream. ?cam=<name> selects the camera (default = first)."""
        qs = self._parse_qs()
        cam = (qs.get('cam', [None])[0] or
               next(iter(latest_frames), None))
        self.send_response(200)
        self.send_header('Content-type', 'multipart/x-mixed-replace; boundary=frame')
        self.end_headers()
        try:
            while True:
                with frame_lock:
                    frame = latest_frames.get(cam) if cam else None
                if frame is not None:
                    self.wfile.write(b'--frame\r\n')
                    self.send_header('Content-type', 'image/jpeg')
                    self.send_header('Content-length', str(len(frame)))
                    self.end_headers()
                    self.wfile.write(frame)
                    self.wfile.write(b'\r\n')
                time.sleep(0.05)
        except Exception:
            pass # Client disconnected

    SEVERITIES = ("high", "medium", "low")

    # --- cases (V4-M8, FR-CS-1..4) -------------------------------------------
    @staticmethod
    def _case_item(store, it):
        """An attached id means nothing on a card — resolve it to what it names.
        Rows can be deleted out from under a case, so a dangling item still renders."""
        kind, ref = it["kind"], it["ref_id"]
        if kind == "entity":
            e = store.get_entity(ref)
            if e:
                label = e["label"] or f"{e['type'].title()} {ref[-6:]}"
                return {"kind": kind, "ref_id": ref, "label": label,
                        "timestamp": e["created_at"], "keyframe": None}
            o = None
        else:
            o = store.get_observation(ref)
        if not o:
            return {"kind": kind, "ref_id": ref, "label": "(deleted)",
                    "timestamp": None, "keyframe": None}
        pred = o["predicate"]
        label = (pred.removeprefix("rule_fired:").replace('_', ' ') if o["origin"] == 'rule'
                 else pred.replace('_', ' '))
        return {"kind": kind, "ref_id": ref, "label": label, "timestamp": o["timestamp"],
                "keyframe": kf_name(o["source_ref"])}

    def serve_cases(self):
        """?status= . Every case ships its resolved items — a case list with three
        cards is not worth a second round trip per card."""
        want = (self._parse_qs().get('status', [''])[0] or '').strip().lower()
        if want and want not in Store.CASE_STATUSES:
            self.send_error(400, "bad status")
            return
        store = Store()
        try:
            out, counts = [], {"all": 0}
            for s in Store.CASE_STATUSES:
                counts[s] = 0
            for c in store.cases():
                counts["all"] += 1
                counts[c["status"]] = counts.get(c["status"], 0) + 1
                if want and c["status"] != want:
                    continue
                items = [self._case_item(store, i) for i in store.case_items(c["case_id"])]
                out.append({
                    "id": c["case_id"], "title": c["title"], "status": c["status"],
                    "description": c["description"], "owner": c["owner"],
                    "opened_at": c["opened_at"], "closed_at": c["closed_at"],
                    "items": items,
                    "counts": {k: sum(1 for i in items if i["kind"] == k)
                               for k in ("alert", "entity", "event")},
                })
            self.send_json({"cases": out, "counts": counts})
        finally:
            store.close()

    def serve_case_create(self):
        body = self._read_json()
        title = (body.get('title') or '').strip()
        if not title:
            self.send_error(400, "title required")
            return
        store = Store()
        try:
            cid = store.create_case(title, (body.get('description') or '').strip() or None,
                                    body.get('owner') or getattr(self, 'current_user_id', None))
            # FR-CS-3: created straight from an alert/entity, so attach in the same call
            if body.get('kind') and body.get('ref_id'):
                store.attach_to_case(cid, body['kind'], body['ref_id'])
            self.send_json({"ok": True, "id": cid})
        finally:
            store.close()

    def serve_case_attach(self, case_id):
        body = self._read_json()
        kind, ref = (body.get('kind') or '').strip(), (body.get('ref_id') or '').strip()
        if kind not in ("alert", "entity", "event") or not ref:
            self.send_error(400, "kind must be alert|entity|event with a ref_id")
            return
        store = Store()
        try:
            if not store.attach_to_case(case_id, kind, ref):
                self.send_error(404, "no such case")
                return
            self.send_json({"ok": True})
        finally:
            store.close()

    def serve_case_status(self, case_id):
        status = (self._read_json().get('status') or '').strip().lower()
        store = Store()
        try:
            if not store.set_case_status(case_id, status):
                self.send_error(404, "no such case")
                return
            self.send_json({"ok": True, "id": case_id, "status": status})
        except ValueError as e:
            self.send_error(400, str(e))
        finally:
            store.close()

    # --- reports (V4-M8, FR-RP-1..3) -----------------------------------------
    def serve_reports(self):
        """The only real cadence in this product is the delivery schedule the
        digest scheduler already runs on, so that is what Scheduled shows."""
        store = Store()
        try:
            u = store.conn.execute(
                "SELECT delivery_schedule, delivery_sink FROM users ORDER BY created_at"
            ).fetchone()
            self.send_json({
                "reports": [dict(r) for r in store.reports()],
                "schedule": {"cadence": u["delivery_schedule"] if u else "off",
                             "sink": u["delivery_sink"] if u else "in_app"},
            })
        finally:
            store.close()

    def serve_report_create(self):
        """{'hours': 24, 'format': 'html'} — digest.py already decides what a
        period's report says; this only fixes it in place so it can be re-read."""
        from intelligence_os import digest, deliver
        body = self._read_json()
        fmt = (body.get('format') or 'html').lower()
        if fmt not in ('html', 'csv'):
            self.send_error(400, "format must be html or csv")
            return
        try:
            hours = max(1.0, min(8760.0, float(body.get('hours') or 24)))
        except (TypeError, ValueError):
            hours = 24.0
        now = time.time()
        since = now - hours * 3600
        store = Store()
        try:
            d = digest.build(store, since, now)
            if fmt == 'html':
                out = deliver.render_html(d, embed_images=True)  # standalone file
            else:
                out = deliver.render_csv(d)
            label = (f"{int(hours)} hours" if hours < 48 else f"{round(hours / 24)} days")
            name = (body.get('name') or '').strip() or f"Digest · last {label}"
            rid = store.add_report(name, fmt, since, now, out)
            self.send_json({"ok": True, "id": rid, "name": name,
                            "items": len(d["rule_fired"]) + len(d["unusual"])})
        finally:
            store.close()

    def serve_report_download(self, report_id):
        store = Store()
        try:
            r = store.get_report(report_id)
            if not r:
                self.send_error(404, "no such report")
                return
            body = r["body"].encode()
            stamp = time.strftime("%Y%m%d-%H%M", time.localtime(r["generated_at"]))
            fname = re.sub(r'[^A-Za-z0-9._-]+', '-', r["name"]).strip('-')
        finally:
            store.close()
        self.send_response(200)
        self.send_header('Content-type',
                         'text/html' if r["format"] == 'html' else 'text/csv')
        self.send_header('Content-Disposition',
                         f'attachment; filename="{fname}-{stamp}.{r["format"]}"')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def serve_alerts(self):
        """V4-M4. ?status=&severity=&camera=&q= . Counts ship in the same payload,
        faceted: each dimension is counted with its *own* filter lifted, so the
        chips show what you'd get by clicking them (FR-AL-2)."""
        from intelligence_os.rules import load_rules
        qs = self._parse_qs()
        want_status = (qs.get('status', [''])[0] or '').strip().lower()
        want_sev = (qs.get('severity', [''])[0] or '').strip().lower()
        want_cam = (qs.get('camera', [''])[0] or '').strip()
        needle = (qs.get('q', [''])[0] or '').strip().lower()
        if want_status and want_status not in Store.ALERT_STATUSES:
            self.send_error(400, "bad status")
            return
        # severity isn't stored — it's a property of the rule, read from rules.yaml
        sev_of = {r.get("name"): (r.get("severity") or "medium").lower()
                  for r in load_rules()}
        store = Store()
        try:
            rows = []
            for o in store.alerts(camera_id=want_cam or None):
                rule = o["predicate"].removeprefix("rule_fired:")
                ent = store.get_entity(o["subject_entity_id"])
                rows.append({
                    "id": o["observation_id"], "rule": rule,
                    "severity": sev_of.get(rule, "medium"),
                    "status": o["status"] or "new",
                    "assignee": o["assignee"],
                    "camera": o["camera_id"], "timestamp": o["timestamp"],
                    "entity_id": o["subject_entity_id"],
                    "entity": (ent["label"] if ent and ent["label"]
                               else o["subject_entity_id"]),
                    "keyframe": kf_name(o["source_ref"]),
                })
        finally:
            store.close()

        def keep(a, *, skip=""):
            if needle and needle not in f"{a['rule']} {a['entity']} {a['camera'] or ''}".lower():
                return False
            if skip != "status" and want_status and a["status"] != want_status:
                return False
            if skip != "severity" and want_sev and a["severity"] != want_sev:
                return False
            return True

        counts = {"all": sum(1 for a in rows if keep(a))}
        for s in self.SEVERITIES:
            counts[s] = sum(1 for a in rows if keep(a, skip="severity") and a["severity"] == s)
        for s in Store.ALERT_STATUSES:
            counts[s] = sum(1 for a in rows if keep(a, skip="status") and a["status"] == s)
        # The rail badge and the dashboard card are global — they must not follow
        # the filter bar, or filtering to "high" would look like alerts vanished.
        self.send_json({
            "alerts": [a for a in rows if keep(a)],
            "counts": counts,
            "unread": sum(1 for a in rows if a["status"] == "new"),
            "recent": rows[:5],
        })

    def serve_alert_status(self, observation_id):
        """{'status': 'acknowledged'} — or {'status':'x','all':true} to bulk-apply
        to every 'new' alert (the Mark all read button), ignoring the path id."""
        body = self._read_json()
        status = (body.get('status') or '').strip().lower()
        user_id = getattr(self, 'current_user_id', None)
        store = Store()
        try:
            if body.get('all'):
                n = sum(store.set_alert_status(a["observation_id"], status, user_id)
                        for a in store.alerts(status="new"))
                self.send_json({"ok": True, "updated": n})
                return
            if not store.set_alert_status(observation_id, status, user_id):
                self.send_error(404, "no such alert")
                return
            self.send_json({"ok": True, "id": observation_id, "status": status})
        except ValueError as e:
            self.send_error(400, str(e))
        finally:
            store.close()

    def serve_observations(self):
        """V4-M3 timeline: ?since=&until=&type=&camera=&bins= .

        Bucketing happens here — the client must not pull 10k rows to draw 60 bars.
        """
        qs = self._parse_qs()

        def num(key, default):
            try:
                return float(qs.get(key, [''])[0])
            except ValueError:
                return default

        until = num('until', time.time())
        since = num('since', until - 86400)
        want = (qs.get('type', [''])[0] or '').strip().lower()
        cam = (qs.get('camera', [''])[0] or '').strip()
        bins = max(1, min(240, int(num('bins', 48))))
        span = max(until - since, 1.0)

        store = Store()
        try:
            ents = {e["entity_id"]: e for e in store.list_entities(active_only=False)}
            rows = store.observations(since=since, until=until, camera_id=cam or None)
            events = []
            for o in rows:
                ent = ents.get(o["subject_entity_id"])
                # ponytail: "unusual" = an entity the system had never seen before this
                # window. Upgrade to a per-camera hour-of-day baseline once there is
                # enough history to learn one.
                cls = ("fired" if o["origin"] == "rule"
                       else "unusual" if ent and ent["created_at"] >= since
                       else "routine")
                events.append({
                    "id": o["observation_id"],
                    "class": cls,
                    "origin": o["origin"],
                    "predicate": o["predicate"],
                    "rule": o["predicate"].removeprefix("rule_fired:") if cls == "fired" else None,
                    "entity_id": o["subject_entity_id"],
                    "entity": (ent["label"] if ent and ent["label"]
                               else o["subject_entity_id"]),
                    "entity_type": ent["type"] if ent else None,
                    "camera": o["camera_id"],
                    "timestamp": o["timestamp"],
                    "confidence": o["confidence"],
                    "keyframe": kf_name(o["source_ref"]),
                })
        finally:
            store.close()

        def keep(e):
            if want in ("", "all"):
                return True
            if want == "alerts":
                return e["class"] == "fired"
            return e["entity_type"] == want

        # The chips are one mutually-exclusive dimension, so their counts come from
        # the *unfiltered* set — otherwise every chip but the active one reads 0.
        counts = {"all": len(events),
                  "alerts": sum(1 for e in events if e["class"] == "fired")}
        for e in events:
            if e["entity_type"]:
                counts[e["entity_type"]] = counts.get(e["entity_type"], 0) + 1

        # The dashboard card is a global signal — it must not narrow when someone
        # filters the timeline pane, so it ships pre-filter (same split as alerts).
        recent = events[-6:][::-1]

        events = [e for e in events if keep(e)]
        buckets = [{"fired": 0, "unusual": 0, "routine": 0} for _ in range(bins)]
        for e in events:
            i = min(bins - 1, int((e["timestamp"] - since) / span * bins))
            buckets[i][e["class"]] += 1
        self.send_json({
            "since": since, "until": until, "counts": counts, "recent": recent,
            "events": events[-500:],   # newest tail; buckets already cover the rest
            "buckets": buckets,
        })

    def serve_entity(self, entity_id):
        """Profile for one entity: what the system knows about X (§I inspect).
        Includes candidate edges (weaker beliefs) so the UI can grey them."""
        store = Store()
        try:
            ent = store.get_entity(entity_id)
            if ent is None:
                self.send_error(404, "Entity not found")
                return

            def label_of(eid):
                if not eid:
                    return None
                e = store.get_entity(eid)
                return (e["label"] if e and e["label"] else eid)

            obs = store.observations(entity_id)
            rels = store.relations(entity_id, min_weight=0.0)
            edges = [{
                "id": r["relation_id"],
                "kind": r["kind"],
                "predicate": r["predicate"],
                "object": label_of(r["object_entity_id"]),
                "location": r["location_id"],
                "weight": round(r["weight"], 2),
                "status": r["status"],           # confirmed | candidate
            } for r in rels if r["status"] != "suppressed"]

            recent = [{
                "predicate": o["predicate"],
                "timestamp": o["timestamp"],
                "origin": o["origin"],
                # the picture behind the claim — an investigator reads the frame,
                # not the predicate
                "keyframe": kf_name(o["source_ref"]),
            } for o in obs[-15:][::-1]]          # newest first

            self.send_json({
                "id": ent["entity_id"],
                "type": ent["type"],
                # same fallback the lists use — "Unknown" made one entity read as two
                "label": ent["label"] or f"{ent['type'].title()} {ent['entity_id'][-6:]}",
                "created_at": ent["created_at"],
                "times_seen": len(obs),
                "last_seen": obs[-1]["timestamp"] if obs else None,
                "relations": [e for e in edges if e["kind"] == "relation"],
                "habits": [e for e in edges if e["kind"] == "habit"],
                "events": [e for e in edges if e["kind"] == "event"],
                "recent_observations": recent,
            })
        finally:
            store.close()

    def serve_entities(self):
        store = Store()
        try:
            people = store.list_entities("person")
            objects = store.list_entities("object")
            seen = store.appearances()
            now = time.time()

            results = []
            for e in people + objects:
                ent_dict = dict(e)
                ent_dict["id"] = ent_dict["entity_id"]
                a = seen.get(e["entity_id"])
                ent_dict["times_seen"] = a["n"] if a else 0
                ent_dict["last_seen"] = a["last_seen"] if a else None
                ent_dict["keyframe"] = kf_name(a["keyframe"]) if a else None
                # ponytail: frequency = sightings per day over the entity's known
                # lifespan. Cheap and honest; swap for a rolling 7d rate once there
                # is enough history that "since first seen" stops being the window.
                days = max((now - e["created_at"]) / 86400.0, 1.0)
                rate = ent_dict["times_seen"] / days
                ent_dict["frequency"] = ("high" if rate >= 3 else
                                         "med" if rate >= 0.5 else "low")
                rels = store.relations(ent_dict["entity_id"], min_weight=0.0)
                ent_dict["relations"] = [r["predicate"] + (f" {r['object_entity_id']}" if r["object_entity_id"] else "")
                                         for r in rels if r["kind"] == "relation" and r["status"] == "confirmed"]
                ent_dict["habits"] = [r["predicate"] + (f" @{r['location_id']}" if r["location_id"] else "")
                                      for r in rels if r["kind"] == "habit" and r["status"] == "confirmed"]
                results.append(ent_dict)
            self.send_json(results)
        finally:
            store.close()

    @staticmethod
    def _entity_node(row):
        """One vis node for an entity, identical everywhere (graph + expand).
        Shape and colour are the client's business (FR-NG-7 reads them off the
        theme's CSS vars); the server only says what a thing is."""
        person = row["type"] == "person"
        return {
            "id": row["entity_id"],
            "label": row["label"] or ("Unknown Person" if person else "Unknown Object"),
            "group": "person" if person else "object",
            "t": row["created_at"],   # first-seen
        }

    def serve_graph(self):
        store = Store()
        try:
            people = store.list_entities("person")
            objects = store.list_entities("object")
            nodes = [self._entity_node(e) for e in people + objects]
            edges = []

            for e in people + objects:
                rels = store.relations(e["entity_id"], min_weight=0.0)
                for r in rels:
                    if r["status"] != "confirmed":
                        continue
                    if r["kind"] == "relation" and r["object_entity_id"]:
                        edges.append({
                            # stable id so the 2s poll's edgesData.update() is idempotent
                            # (no id => vis inserts a duplicate parallel edge every tick)
                            "id": e["entity_id"] + "|" + r["predicate"] + "|" + r["object_entity_id"],
                            "from": e["entity_id"],
                            "to": r["object_entity_id"],
                            "label": r["predicate"],
                            "arrows": "to",
                            "t": r["created_at"]
                        })
                    elif r["kind"] == "habit" and r["location_id"]:
                        # create an implicit node for the location if not exists
                        loc_id = r["location_id"]
                        if not any(n["id"] == loc_id for n in nodes):
                            nodes.append({"id": loc_id, "label": loc_id,
                                          "group": "location"})
                        edges.append({
                            "id": e["entity_id"] + "|" + r["predicate"] + "|" + loc_id,
                            "from": e["entity_id"],
                            "to": loc_id,
                            "label": r["predicate"],
                            "arrows": "to",
                            "dashes": True,
                            "t": r["created_at"]
                        })

            self.send_json({"nodes": nodes, "edges": edges})
        finally:
            store.close()

    def serve_entity_expand(self, entity_id):
        """Transform (§8.3): grow the graph around one node. Reveals ALL of this
        entity's relations — including the weaker `candidate` beliefs the confirmed-only
        world view hides — plus their neighbour nodes. The link-analysis 'expand' gesture."""
        store = Store()
        try:
            if store.get_entity(entity_id) is None:
                self.send_error(404, "Entity not found")
                return
            nodes, edges, seen = [], [], set()
            for r in store.relations(entity_id, min_weight=0.0):
                if r["status"] == "suppressed":
                    continue
                dashed = r["status"] != "confirmed"   # candidate beliefs drawn tentative
                if r["kind"] == "relation" and r["object_entity_id"]:
                    obj = store.get_entity(r["object_entity_id"])
                    if obj is None:
                        continue
                    if obj["entity_id"] not in seen:
                        seen.add(obj["entity_id"])
                        nodes.append(self._entity_node(obj))
                    edges.append({
                        "id": entity_id + "|" + r["predicate"] + "|" + r["object_entity_id"],
                        "from": entity_id, "to": r["object_entity_id"],
                        "label": r["predicate"], "arrows": "to", "dashes": dashed,
                        "t": r["created_at"],
                    })
                elif r["kind"] == "habit" and r["location_id"]:
                    loc = r["location_id"]
                    if loc not in seen:
                        seen.add(loc)
                        nodes.append({"id": loc, "label": loc, "group": "location"})
                    edges.append({
                        "id": entity_id + "|" + r["predicate"] + "|" + loc,
                        "from": entity_id, "to": loc,
                        "label": r["predicate"], "arrows": "to", "dashes": True,
                        "t": r["created_at"],
                    })
            self.send_json({"nodes": nodes, "edges": edges})
        finally:
            store.close()

    def serve_stats(self):
        store = Store()
        try:
            c = store.conn.cursor()
            total_persons = c.execute("SELECT COUNT(*) FROM entities WHERE type='person' AND status='active'").fetchone()[0]
            total_objects = c.execute("SELECT COUNT(*) FROM entities WHERE type='object' AND status='active'").fetchone()[0]
            total_obs = c.execute("SELECT COUNT(*) FROM observations").fetchone()[0]

            cutoff = time.time() - (24 * 3600)
            rows = c.execute("""
                SELECT
                    CAST(timestamp / 3600 AS INT) * 3600 as hour,
                    COUNT(*) as count
                FROM observations
                WHERE timestamp >= ?
                GROUP BY hour
                ORDER BY hour ASC
            """, (cutoff,)).fetchall()

            timeline = [{"hour": r["hour"], "count": r["count"]} for r in rows]

            # M6: per-camera observation counts
            cam_rows = c.execute("""
                SELECT camera_id, COUNT(*) as count
                FROM observations
                WHERE camera_id IS NOT NULL
                GROUP BY camera_id
            """).fetchall()
            per_camera = {r["camera_id"]: r["count"] for r in cam_rows}

            # M6: per-camera ingest health
            cameras_state = pipeline_state.get("cameras", {})
            ingest = {}
            for name, cs in cameras_state.items():
                ingest[name] = {
                    "last_frame_ts": cs.get("last_frame_ts"),
                    "is_stream": cs.get("is_stream", False),
                    "paused": cs.get("paused", False),
                }
            # backward compat: expose first camera at top level too
            first_cam = next(iter(cameras_state), None)
            first_ingest = ingest.get(first_cam, {}) if first_cam else {}

            self.send_json({
                "system": self._system_health(cameras_state, now_ts=time.time()),
                "kpis": {
                    "total_persons": total_persons,
                    "total_objects": total_objects,
                    "total_observations": total_obs,
                    "per_camera": per_camera,
                },
                "timeline": timeline,
                "vlm_enabled": CONFIG.vlm.enabled,
                "ingest": {
                    "last_frame_ts": first_ingest.get("last_frame_ts"),
                    "is_stream": first_ingest.get("is_stream", False),
                    "paused": first_ingest.get("paused", False),
                    "cameras": ingest,
                },
            })
        finally:
            store.close()

    @staticmethod
    def _system_health(cameras_state, now_ts):
        """Rail-footer numbers (FR-SH-8). stdlib only — no psutil in this env.
        ponytail: CPU is loadavg-derived and memory is *this process* against
        physical RAM; swap both for psutil if system-wide accuracy matters."""
        import shutil, resource
        cams = list(cameras_state.values())
        online = sum(1 for cs in cams
                     if cs.get("last_frame_ts") and (now_ts - cs["last_frame_ts"]) < 10)

        try:
            cpu = min(1.0, os.getloadavg()[0] / (os.cpu_count() or 1))
        except (OSError, AttributeError):
            cpu = None

        try:
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            rss *= 1 if sys.platform == "darwin" else 1024   # mac: bytes, linux: KiB
            total_ram = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
            mem = rss / total_ram
        except (ValueError, OSError, AttributeError):
            mem = None

        try:
            du = shutil.disk_usage(DATA_DIR)
            storage = {"used_bytes": du.used, "total_bytes": du.total,
                       "pct": du.used / du.total}
        except OSError:
            storage = None

        return {
            "cameras_online": online, "cameras_total": len(cams),
            "engine": "running" if pipeline_state.get("running", True) else "stopped",
            "cpu_pct": cpu, "memory_pct": mem, "storage": storage,
        }

    def send_json(self, data):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode('utf-8'))

def start_pipeline(args):
    try:
        run_pipeline(args, on_frame=update_frame, state=pipeline_state)
    except Exception as e:
        print(f"Pipeline crashed: {e}")

def main(argv=None):
    """Console entry point (`intelligence-os`) and `python -m intelligence_os.web`.

    A function rather than a bare `__main__` block so packaging can point at it;
    nothing here is CWD-relative, so it runs from any directory.
    """
    p = argparse.ArgumentParser(prog="intelligence-os")
    # not required: falls back to config.yaml, then webcam 0 (§10 no-config boot)
    g = p.add_mutually_exclusive_group(required=False)
    g.add_argument("--webcam", type=int)
    g.add_argument("--video", type=str)
    p.add_argument("--zones", type=str, default=None)
    p.add_argument("--snapshot-every", type=float, default=30.0)
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--sensitivity", choices=["lazy", "balanced", "eager"], default=None)
    p.add_argument("--no-auto-merge", action="store_true")
    p.add_argument("--show", dest="show", action="store_true", default=None)
    p.add_argument("--no-show", dest="show", action="store_false")
    p.add_argument("--port", type=int, default=8000)
    # Stays 127.0.0.1 by default: there is no TLS, no CSRF token and no rate
    # limiting on login, so the loopback interface is doing real work as a
    # boundary. Containers are the honest exception — the network namespace is
    # the boundary there, and a container that binds loopback is unreachable
    # even from its own published port — so the image passes --host 0.0.0.0 and
    # publishes to 127.0.0.1 on the host instead. See docs/docker.md.
    p.add_argument("--host", type=str, default="127.0.0.1",
                   help="interface to bind (default 127.0.0.1; use 0.0.0.0 in a container)")

    args = p.parse_args(argv)
    from intelligence_os.run import resolve_source
    resolve_source(args)                       # CLI > config.yaml > webcam 0
    if args.sensitivity is None:
        args.sensitivity = "balanced"          # web default: responsive live demo
    if args.show is None:
        args.show = False

    t = threading.Thread(target=start_pipeline, args=(args,), daemon=True)
    t.start()

    server = ThreadedHTTPServer((args.host, args.port), RequestHandler)
    shown = "localhost" if args.host in ("127.0.0.1", "0.0.0.0") else args.host
    print(f"Starting Intelligence OS Web Server at http://{shown}:{args.port} ...")
    if args.host != "127.0.0.1":
        print(f"  Bound to {args.host} — anything that can reach this interface can reach "
              f"the dashboard. Put TLS and access control in front of it.")
    server.serve_forever()


if __name__ == "__main__":
    main()
