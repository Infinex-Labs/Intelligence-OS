"""Rule compiler + cascade filter (FinalPRD §9.2, §9.3, §11 — M1 keystone).

Two halves, one file:

1. COMPILER (authoring time, one LLM call, then never again): plain English ->
   a RuleSpec (trigger / gate / verify / cooldown). Two mandatory gates that are
   trust properties, NOT tunable accuracy:
     - feasibility (§9.2): refuse un-observable rules ("if someone seems stressed")
       OUT LOUD rather than quietly not-watching.
     - identity scope (§11): a rule about ONE named person's movements is a
       different product; require IDENTITY_ENABLED and say so in plain language.

2. ENGINE (frame loop, cheap geometry): consumes already-resolved detections
   (entity_id + class + zone location_id) from run.py, applies zone/dwell/cooldown
   filters, and only then spends a VLM crop-verify call. On a confirmed hit it
   writes the fired event as an observation with origin='rule' + a keyframe.

The LLM is never in the frame loop. Compilation is the only place it touches a rule.
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Callable, Optional

import cv2

from .config import CONFIG, DATA_DIR, FRAMES_DIR

RULES_PATH = DATA_DIR / "rules.yaml"


def identity_enabled() -> bool:
    """§11: identity is opt-in, off unless consciously enabled. Default false."""
    return os.environ.get("IDENTITY_ENABLED", "").strip().lower() in ("1", "true", "yes", "on")


# --- compiler output shapes --------------------------------------------------
@dataclass
class Refusal:
    reason: str          # no_key | infeasible | needs_identity
    message: str         # plain-language explanation for the author


# A compiled rule is a plain dict (yaml-native, single shape across compiler /
# storage / engine). Shape:
#   name:    str
#   source:  str                       # the English the author typed
#   trigger: {class: str, zone: str|None}
#   gate:    {dwell_seconds: float}
#   verify:  {prompt: str} | None      # None -> geometry-only rule, no VLM
#   cooldown:{per_track_seconds: float}
#   needs_identity: bool

COMPILE_TOOL = {
    "name": "compiled_rule",
    "description": "Compile a monitoring rule, or refuse it.",
    "input_schema": {
        "type": "object",
        "properties": {
            "feasible": {"type": "boolean",
                         "description": "Can a camera cascade (person/object detection + "
                         "zones + dwell time + a yes/no VLM look at a crop) RELIABLY "
                         "detect this? False for internal states (stress, intent, "
                         "'looks unsafe') that are not visually observable."},
            "reason": {"type": "string",
                       "description": "If not feasible, plain-language why, addressed to the author."},
            "targets_individual": {"type": "boolean",
                       "description": "True if the rule is about a SPECIFIC named person's "
                       "comings/goings (e.g. 'when my wife leaves'), which requires face "
                       "identity. False for 'anyone in a zone' style rules."},
            "identity_note": {"type": "string",
                       "description": "If targets_individual, restate in plain language what "
                       "is being asked to track about whom."},
            "name": {"type": "string", "description": "short snake_case id, e.g. smoking_loading_bay"},
            "trigger_class": {"type": "string",
                       "description": "the primary object class to key on, e.g. 'person', "
                       "or '' if none applies"},
            "trigger_zone": {"type": "string",
                       "description": "zone name the rule is scoped to, or '' for anywhere"},
            "dwell_seconds": {"type": "number",
                       "description": "seconds the subject must persist before the rule can "
                       "fire (filters passers-by); 0 if instantaneous presence suffices"},
            "needs_verify": {"type": "boolean",
                       "description": "does confirming this require LOOKING at the image "
                       "(e.g. smoking, no-helmet)? False if geometry alone proves it "
                       "(e.g. person-in-zone, dwell)."},
            "verify_prompt": {"type": "string",
                       "description": "if needs_verify, a single yes/no question about the "
                       "cropped subject, ending 'yes/no/unclear.'"},
            "cooldown_seconds": {"type": "number",
                       "description": "min seconds between fires for the same track (default 300)"},
            "trigger_camera": {"type": "string",
                       "description": "camera name this rule is scoped to, or '' for all cameras"},
        },
        "required": ["feasible", "targets_individual", "name", "trigger_class",
                     "needs_verify"],
    },
}

COMPILE_SYSTEM = (
    "You compile a plain-English monitoring rule into a detection spec for a camera "
    "cascade. The cascade can: detect people and common objects, know which named "
    "zone a detection is in, measure how long something dwells, and ask one yes/no "
    "question about a cropped region with a vision model.\n\n"
    "TWO refusals are mandatory and must be loud, never silent:\n"
    "1. FEASIBILITY: if the rule asks for something not visually observable "
    "(emotions, intent, 'looks unsafe', 'seems stressed'), set feasible=false and "
    "explain. A system that quietly doesn't watch what it was asked to is a liability.\n"
    "2. IDENTITY SCOPE: if the rule targets a SPECIFIC individual's movements "
    "('tell me when my wife leaves', 'who visits when I'm out'), set "
    "targets_individual=true and restate plainly in identity_note. This is a "
    "different, higher-stakes product than 'anyone in a zone'.\n\n"
    "Most rules need only 'a person', not 'which person' — do not set "
    "targets_individual for generic presence/zone/PPE rules. Prefer geometry "
    "(needs_verify=false) when zones+dwell alone prove the rule; reserve the VLM "
    "verify for things you must LOOK at. Answer by calling the compiled_rule tool."
)


def compile_rule(text: str) -> "dict | Refusal":
    """One LLM call. Returns a compiled-rule dict or a Refusal. Never fires anything."""
    if not CONFIG.vlm.enabled:
        return Refusal("no_key",
                       "Rule compilation needs a model. Set ANTHROPIC_API_KEY. "
                       "(Detection, tracking, memory and the digest run without one; "
                       "only rule authoring and VLM-verified rules need a key.)")
    import anthropic  # deferred, optional dep
    client = anthropic.Anthropic()
    msg = client.messages.create(
        model=CONFIG.vlm.model,
        max_tokens=600,
        system=COMPILE_SYSTEM,
        tools=[COMPILE_TOOL],
        tool_choice={"type": "tool", "name": "compiled_rule"},
        messages=[{"role": "user", "content": f"Compile this rule: {text!r}"}],
    )
    out = None
    for block in msg.content:
        if getattr(block, "type", None) == "tool_use":
            out = block.input
            break
    if out is None:
        return Refusal("infeasible", "Could not compile the rule.")
    return _apply_gates(out, text)


def _apply_gates(out: dict, text: str) -> "dict | Refusal":
    """The trust boundary (§9.2, §11), split out so it's testable without a key.
    Given the compiler's structured output, either refuse loudly or emit a spec."""
    if not out.get("feasible", False):
        return Refusal("infeasible", out.get("reason")
                       or "This rule asks for something the camera cannot reliably observe.")
    if out.get("targets_individual") and not identity_enabled():
        note = out.get("identity_note") or "This rule tracks a specific individual."
        return Refusal("needs_identity",
                       f"{note} That requires face identity, which is off by default "
                       "(IDENTITY_ENABLED=false). Enable it consciously to compile this.")

    zone = (out.get("trigger_zone") or "").strip()
    camera = (out.get("trigger_camera") or "").strip()
    verify = {"prompt": out["verify_prompt"]} if out.get("needs_verify") and out.get("verify_prompt") else None
    return {
        "name": out["name"],
        "source": text,
        "trigger": {"class": (out.get("trigger_class") or "").strip() or None,
                    "zone": zone or None,
                    "camera": camera or None},
        "gate": {"dwell_seconds": float(out.get("dwell_seconds") or 0)},
        "verify": verify,
        "cooldown": {"per_track_seconds": float(out.get("cooldown_seconds") or 300)},
        "needs_identity": bool(out.get("targets_individual")),
    }


# --- storage -----------------------------------------------------------------
def load_rules() -> list[dict]:
    if not RULES_PATH.exists():
        return []
    import yaml
    data = yaml.safe_load(RULES_PATH.read_text()) or {}
    return data.get("rules", [])


def save_rule(spec: dict) -> None:
    """Append, or replace an existing rule with the same name."""
    import yaml
    rules = [r for r in load_rules() if r.get("name") != spec["name"]]
    rules.append(spec)
    RULES_PATH.parent.mkdir(parents=True, exist_ok=True)
    RULES_PATH.write_text(yaml.safe_dump({"rules": rules}, sort_keys=False))


def set_rule_enabled(name: str, enabled: bool) -> bool:
    """Toggle a rule without deleting it (M5). It stays in rules.yaml; the engine
    just skips disabled rules. Takes effect on the next pipeline start, like edits.
    Raises KeyError if no such rule."""
    import yaml
    rules = load_rules()
    if not any(r.get("name") == name for r in rules):
        raise KeyError(name)
    for r in rules:
        if r.get("name") == name:
            r["enabled"] = enabled
    RULES_PATH.write_text(yaml.safe_dump({"rules": rules}, sort_keys=False))
    return enabled


# --- correction loop (§9.3 — M5): dismissed crops sharpen the verifier -------
def _crop_path(keyframe_path) -> str:
    """The person-crop saved alongside a rule keyframe (negative-example material)."""
    s = str(keyframe_path)
    return (s[:-4] if s.lower().endswith(".jpg") else s) + ".crop.jpg"


def dismissed_crops(store, rule_name: str, limit: int = 3) -> list[str]:
    """Crops of past fires of `rule_name` that an operator marked wrong in the digest.

    Links `digest:dismissed:rule:<rule>:<entity>:<int_ts>` corrections (written by
    digest.feedback) back to the fired event's saved crop. Newest `limit` only — the
    verifier gets a few counter-examples, not an unbounded prompt.
    """
    dismissed: set[tuple] = set()               # (rule, entity, int_ts)
    fired: dict[tuple, str] = {}                # (rule, entity, int_ts) -> keyframe path
    for o in store.observations():
        p = o["predicate"]
        if p.startswith("digest:dismissed:rule:"):
            parts = p[len("digest:dismissed:"):].split(":")   # rule, <rule>, <entity>, <ts>
            if len(parts) == 4:
                dismissed.add((parts[1], parts[2], parts[3]))
        elif o["origin"] == "rule" and p == f"rule_fired:{rule_name}" and o["source_ref"]:
            fired[(rule_name, o["subject_entity_id"], str(int(o["timestamp"])))] = o["source_ref"]

    hits: list[tuple[int, str]] = []
    for key in dismissed:
        if key[0] != rule_name:
            continue
        kf = fired.get(key)
        if not kf:
            continue
        cp = _crop_path(kf)
        if os.path.exists(cp):
            hits.append((int(key[2]), cp))
    hits.sort()                                 # oldest -> newest by fire time
    return [cp for _, cp in hits[-limit:]]


# --- engine (cascade filter) -------------------------------------------------
@dataclass
class FiredEvent:
    rule: str
    entity_id: str
    location_id: Optional[str]
    observation_id: str
    keyframe: str
    timestamp: float


class RuleEngine:
    """Stage 3-4 of the cascade. Fed already-resolved detections from run.py; the
    LLM never runs here. Verifier is a callable(frame, bbox, prompt)->yes/no/unclear
    (Describer.verify); None -> verify-rules simply don't fire (precision, §9.3)."""

    gap_reset_s = 5.0     # subject unseen this long -> restart its dwell clock
    verify_retry_s = 30.0  # after a non-'yes' verify, wait this long before re-asking
    # ponytail: fixed retry interval; make per-rule if a rule ever needs faster re-checks

    def __init__(self, store, rules: Optional[list[dict]] = None,
                 verifier: Optional[Callable[..., str]] = None,
                 frames_dir=FRAMES_DIR):
        self.store = store
        # disabled rules stay in rules.yaml but never fire (M5 enable/disable)
        self.rules = [r for r in (rules if rules is not None else load_rules())
                      if r.get("enabled", True)]
        self.verifier = verifier
        self.frames_dir = frames_dir
        # zone name -> location_id (rules reference zones by name; detections carry ids)
        self.zone_ids = {r["name"]: r["location_id"] for r in store.locations()}
        self._dwell: dict[tuple, float] = {}      # (rule, entity) -> first_seen_ts
        self._last_seen: dict[tuple, float] = {}
        self._cooldown: dict[tuple, float] = {}   # (rule, entity) -> last_fired_ts
        self._last_verify: dict[tuple, float] = {}  # (rule, entity) -> last non-yes ask
        # §9.2 principle applied at arm-time: a rule whose zone doesn't exist can
        # never fire. Surface it loudly rather than watching nothing in silence.
        self.zone_warnings = [(r["name"], r["trigger"]["zone"]) for r in self.rules
                              if r.get("trigger", {}).get("zone")
                              and r["trigger"]["zone"] not in self.zone_ids]
        # §9.3 correction moat: crops an operator dismissed become negative examples
        # fed into the verifier so the rule sharpens with use. Loaded at arm time —
        # a corrected rule gets sharper on the next pipeline start (same lifecycle as
        # rule edits). ponytail: refresh mid-run only if corrections need to land live.
        self._neg: dict[str, list] = {}
        for r in self.rules:
            if not r.get("verify"):
                continue
            imgs = [im for im in (cv2.imread(p) for p in dismissed_crops(store, r["name"]))
                    if im is not None]
            if imgs:
                self._neg[r["name"]] = imgs
                print(f"[rule] {r['name']}: {len(imgs)} correction example(s) loaded")

    def feed(self, resolved, frame_bgr, ts: float) -> list[FiredEvent]:
        """resolved: list with .entity_id, .det (Detection), .location_id."""
        fired: list[FiredEvent] = []
        for rd in resolved:
            for rule in self.rules:
                ev = self._check(rule, rd, frame_bgr, ts)
                if ev:
                    fired.append(ev)
        # restart dwell for subjects that dropped out of view
        for key, seen in list(self._last_seen.items()):
            if ts - seen > self.gap_reset_s:
                self._dwell.pop(key, None)
                self._last_seen.pop(key, None)
        return fired

    def _check(self, rule, rd, frame_bgr, ts) -> Optional[FiredEvent]:
        trig = rule.get("trigger", {})
        if trig.get("class") and rd.det.cls_name != trig["class"]:
            return None
        # M6: camera filter — a rule scoped to one camera ignores others
        rule_cam = trig.get("camera")
        if rule_cam and getattr(rd, 'camera_id', None) != rule_cam:
            return None
        key = (rule["name"], rd.entity_id)
        zone = trig.get("zone")
        if zone:
            lid = self.zone_ids.get(zone)
            if lid is None or rd.location_id != lid:
                self._dwell.pop(key, None)      # left the zone -> reset dwell
                return None

        self._last_seen[key] = ts
        first = self._dwell.setdefault(key, ts)
        dwell = float(rule.get("gate", {}).get("dwell_seconds", 0) or 0)
        if ts - first < dwell:
            return None

        cd = float(rule.get("cooldown", {}).get("per_track_seconds", 300) or 0)
        last = self._cooldown.get(key)
        if last is not None and ts - last < cd:
            return None

        verify = rule.get("verify")
        if verify:
            if self.verifier is None:
                return None                    # no key -> don't fire (precision)
            # don't re-spend a VLM call every frame on a subject that already
            # verified 'no' — the whole cascade exists to keep stage 4 rare (§9.1)
            asked = self._last_verify.get(key)
            if asked is not None and ts - asked < self.verify_retry_s:
                return None
            self._last_verify[key] = ts
            verdict = self.verifier(frame_bgr, rd.det.bbox, verify["prompt"],
                                    self._neg.get(rule["name"]))
            # observability: the verify stage is otherwise invisible — this line is
            # how you see "dwell passed, VLM was asked, it said no" vs "never got there"
            print(f"[rule] verify {rule['name']} entity={rd.entity_id[-6:]} -> {verdict}")
            if verdict != "yes":
                return None

        self._cooldown[key] = ts
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        kf = self.frames_dir / f"rule_{rule['name']}_{int(ts * 1000)}.jpg"
        cv2.imwrite(str(kf), frame_bgr)
        # verify-rules also save the person crop, so a later "not smoking" dismissal
        # has a tight negative example to feed back into the verifier (§9.3, M5).
        if verify:
            x1, y1, x2, y2 = rd.det.bbox
            crop = frame_bgr[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
            if crop.size:
                cv2.imwrite(_crop_path(kf), crop)
        oid = self.store.add_observation(
            rd.entity_id, f"rule_fired:{rule['name']}",
            location_id=rd.location_id, confidence=0.9,
            source_ref=str(kf), origin="rule", timestamp=ts)
        return FiredEvent(rule["name"], rd.entity_id, rd.location_id, oid, str(kf), ts)


# --- CLI ---------------------------------------------------------------------
def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="intelligence_os.rules")
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("add", help="compile an English rule and save it")
    a.add_argument("text")
    sub.add_parser("list", help="show compiled rules")
    args = p.parse_args(argv)

    if args.cmd == "list":
        import yaml
        rules = load_rules()
        if not rules:
            print("no rules yet — add one with: python -m intelligence_os.rules add \"...\"")
            return 0
        print(yaml.safe_dump({"rules": rules}, sort_keys=False))
        return 0

    result = compile_rule(args.text)
    if isinstance(result, Refusal):
        print(f"REFUSED ({result.reason}): {result.message}", file=sys.stderr)
        return 1
    import yaml
    print(yaml.safe_dump(result, sort_keys=False))
    save_rule(result)
    print(f"saved to {RULES_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
