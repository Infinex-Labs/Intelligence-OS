"""VLM trigger decision + describer (§E, §F, §8 — Phase 5).

The VLM is the most expensive stage; it fires on CHANGE, not on a clock. A static
scene costs zero VLM calls. When it does fire, it runs once on a settled keyframe
and returns a STRUCTURED scene description (not prose) so the output is diffable
and queryable. VLM outputs are written as observations with confidence < 1.0 and
origin='vlm' — never as fact (hallucination containment, NFR-4).

Both VLM roles (describer here, reasoner in distill.py) use Anthropic
claude-opus-4-8. With no SDK / API key, the system degrades to a no-op describer
(it simply doesn't add VLM observations) — it can never hallucinate when off.
"""
from __future__ import annotations

import base64
import json
import time
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from .config import CONFIG


# --- Trigger decision (§8) ---------------------------------------------------
@dataclass
class TriggerContext:
    motion_present: bool
    settled: bool                      # settled keyframe available
    new_entity: bool                   # a freshly minted entity this event
    scene_diff: bool                   # scene-state add/remove/change vs last
    interaction: bool                  # person-object interaction event
    scene_signature: str               # identity of the current situation


class TriggerDecider:
    """Implements the §8 decision + cooldown. Returns whether to spend a VLM call."""

    def __init__(self):
        self.cooldown = CONFIG.trigger.vlm_cooldown_seconds
        self.sensitivity = CONFIG.trigger.sensitivity
        self._last_call: dict[str, float] = {}   # scene_signature -> ts

    def _in_cooldown(self, sig: str, now: float) -> bool:
        last = self._last_call.get(sig)
        return last is not None and (now - last) < self.cooldown

    def should_fire(self, ctx: TriggerContext, now: Optional[float] = None) -> bool:
        now = now or time.time()
        # the meaningful-change conditions (§8). new_entity may be a latched
        # "change pending" flag carried until a clear keyframe arrives.
        meaningful = ctx.new_entity or ctx.scene_diff or ctx.interaction
        if not meaningful:
            return False
        if self.sensitivity == "lazy":
            # memory-builder bias: spend only on a CLEAR keyframe — motion just
            # settled, or the scene is currently still (§8 keyframe selection).
            if not (ctx.settled or not ctx.motion_present):
                return False
        # 'balanced'/'eager' fire as soon as the change is meaningful.
        if self._in_cooldown(ctx.scene_signature, now):
            return False
        self._last_call[ctx.scene_signature] = now
        return True


# --- Structured describer (§F) -----------------------------------------------
DESCRIBE_SCHEMA = {
    "name": "scene_description",
    "description": "Structured, diffable description of a single settled keyframe.",
    "input_schema": {
        "type": "object",
        "properties": {
            "locations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"},
                        "contents": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["location", "contents"],
                },
            },
            "people": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "ref": {"type": "string",
                                "description": "known entity id if provided, else a description"},
                        "state": {"type": "string",
                                  "description": "open-vocabulary observable state, e.g. sleeping, on phone"},
                        "context_analysis": {"type": "string",
                                  "description": "Optional: compare the person's current state/objects to their known context. e.g. 'using phone instead of usual laptop'"}
                    },
                    "required": ["state"],
                },
            },
            "objects": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "label": {"type": "string"},
                        "description": {"type": "string"},
                    },
                    "required": ["label"],
                },
            },
            "notable": {
                "type": "array",
                "items": {"type": "string"},
                "description": "any other observable, open-vocabulary states (spill, window open, ...)",
            },
        },
        "required": ["locations", "people", "objects", "notable"],
    },
}

SYSTEM_DESCRIBE = (
    "You are a context-aware perception observer for a memory system. "
    "You will receive a frame and a 'Known context' JSON describing the entities present. "
    "Describe what is visibly happening in the frame. "
    "COMPARE what you see against the known context (e.g. if someone is using a different object, acting unusually, or if their current state matches known habits). "
    "Use open vocabulary for observable states. If unsure, omit. "
    "Return your answer by calling the scene_description tool."
)


@dataclass
class SceneDescription:
    raw: dict
    model: str
    def states(self) -> list[tuple[str, str]]:
        """Flatten to (subject_ref, predicate) observations for memory write."""
        out = []
        for p in self.raw.get("people", []):
            if p.get("state"):
                out.append((p.get("ref", "person"), f"state:{p['state']}"))
            if p.get("context_analysis"):
                out.append((p.get("ref", "person"), f"context_analysis:{p['context_analysis']}"))
        for n in self.raw.get("notable", []):
            out.append(("scene", n))
        return out


# --- Crop verifier (cascade stage 4, §9.1) -----------------------------------
VERIFY_SCHEMA = {
    "name": "verdict",
    "description": "Answer a single yes/no question about the cropped image.",
    "input_schema": {
        "type": "object",
        "properties": {"answer": {"type": "string", "enum": ["yes", "no", "unclear"]}},
        "required": ["answer"],
    },
}


class Describer:
    """Calls Anthropic with a forced tool to get structured JSON. No key -> None."""

    def __init__(self):
        self.cfg = CONFIG.vlm
        self._client = None

    def _ensure(self):
        if self._client is None:
            import anthropic  # deferred; optional dependency
            self._client = anthropic.Anthropic()
        return self._client

    @staticmethod
    def _encode(frame_bgr: np.ndarray) -> str:
        ok, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return base64.b64encode(buf.tobytes()).decode("ascii")

    def describe(self, frame_bgr: np.ndarray,
                context: Optional[dict] = None) -> Optional[SceneDescription]:
        if not self.cfg.enabled:
            return None
        client = self._ensure()
        ctx_txt = ""
        if context:
            ctx_txt = "\nKnown context (may be incomplete): " + json.dumps(context)
        msg = client.messages.create(
            model=self.cfg.model,
            max_tokens=self.cfg.max_tokens,
            system=SYSTEM_DESCRIBE,
            tools=[DESCRIBE_SCHEMA],
            tool_choice={"type": "tool", "name": "scene_description"},
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {
                        "type": "base64", "media_type": "image/jpeg",
                        "data": self._encode(frame_bgr)}},
                    {"type": "text", "text": "Describe this frame." + ctx_txt},
                ],
            }],
        )
        for block in msg.content:
            if getattr(block, "type", None) == "tool_use":
                return SceneDescription(block.input, self.cfg.model)
        return None

    def verify(self, frame_bgr: np.ndarray, bbox: tuple[int, int, int, int],
               prompt: str, negatives: Optional[list] = None) -> str:
        """Cascade stage-4 verifier: crop the bbox, ask one yes/no question.
        Returns 'yes' | 'no' | 'unclear'. No key -> 'unclear' (never a false
        positive; a rule that can't verify simply doesn't fire, §9.3).

        `negatives`: crops of past fires a human marked wrong (§9.3 correction moat).
        Prepended as counter-examples so the rule sharpens with use."""
        if not self.cfg.enabled:
            return "unclear"
        x1, y1, x2, y2 = bbox
        crop = frame_bgr[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
        if crop.size == 0:
            return "unclear"
        client = self._ensure()
        content: list = []
        neg_imgs = [n for n in (negatives or []) if n is not None and getattr(n, "size", 0)]
        for neg in neg_imgs[:3]:
            content.append({"type": "image", "source": {
                "type": "base64", "media_type": "image/jpeg", "data": self._encode(neg)}})
        if content:
            content.append({"type": "text", "text":
                "The image(s) above are past cases a human reviewer marked as NOT a "
                "match for the question — negative examples. Do not answer 'yes' for "
                "cases like those. Now judge the following image:"})
        content.append({"type": "image", "source": {
            "type": "base64", "media_type": "image/jpeg", "data": self._encode(crop)}})
        content.append({"type": "text", "text": prompt})
        msg = client.messages.create(
            model=self.cfg.model,
            max_tokens=100,
            system=("You verify one yes/no question about the cropped image region. "
                    "Answer 'unclear' if you genuinely cannot tell. Precision matters "
                    "more than recall — an alert lands on a specific person. "
                    "Answer by calling the verdict tool."),
            tools=[VERIFY_SCHEMA],
            tool_choice={"type": "tool", "name": "verdict"},
            messages=[{"role": "user", "content": content}],
        )
        for block in msg.content:
            if getattr(block, "type", None) == "tool_use":
                return block.input.get("answer", "unclear")
        return "unclear"
