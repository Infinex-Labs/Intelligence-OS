"""Central configuration for the Perceptual Memory System POC.

Every tunable the spec calls out as "tunable" or "config" lives here so the
trigger sensitivity / identity threshold can be tweaked without touching logic.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# --- Paths -------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent


def _load_dotenv() -> None:
    # ponytail: 6-line .env reader instead of the python-dotenv dependency.
    env = ROOT / ".env"
    if not env.exists():
        return
    for line in env.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_dotenv()
DATA_DIR = Path(os.environ.get("INTELLIGENCE_OS_DATA", ROOT / "data"))
DB_PATH = Path(os.environ.get("INTELLIGENCE_OS_DB", DATA_DIR / "memory.db"))
FRAMES_DIR = DATA_DIR / "frames"   # retained keyframes/crops for audit (NFR retention)


def retained_keyframe(source_ref) -> "str | None":
    """Basename of a still-on-disk keyframe, or None if retention reclaimed it.

    Retention (§11) deletes raw frames after `raw_retention_days` but keeps the
    observations — the claim outlives the picture, on purpose. So an old row's
    `source_ref` names a file that is gone, and anything that hands that name to
    a client produces a broken image. Callers check here before advertising it.

    None means retention worked, not that something failed.
    """
    if not source_ref:
        return None
    name = os.path.basename(str(source_ref))
    return name if (FRAMES_DIR / name).exists() else None


def resolve_yolo_weights() -> str:
    """§10 model weights: fetch on first run, never bake in. Precedence:
    INTELLIGENCE_OS_YOLO env > a local checkout copy (dev keeps it, gitignored) >
    a bare model name so Ultralytics downloads + caches it (its own progress bar)
    on first run. A fresh clone thus fetches rather than crashing on a missing file."""
    env = os.environ.get("INTELLIGENCE_OS_YOLO")
    if env:
        return env
    name = "yolo26s.pt"
    local = ROOT.parent / name
    return str(local) if local.exists() else name


@dataclass
class IdentityConfig:
    # Face match threshold (cosine similarity). Tunable per §C / FR-6.
    # buffalo_l ArcFace embeddings: ~0.5 is a common operating point; tune on
    # real people in real lighting (Phase-1 acceptance).
    face_match_threshold: float = 0.45
    # Cap on stored signatures per entity to capture pose/lighting variation (§C).
    max_signatures_per_entity: int = 12
    # Only add a new signature if it is at least this *different* from existing
    # ones (1 - cos), so we keep variety instead of N near-duplicates.
    signature_novelty_min: float = 0.10
    # Minimum detector confidence for a face to be used for identity.
    min_face_det_score: float = 0.55
    # FR-ST-3: off by default. Without it entities are track IDs (Person 01) and
    # nearly every rule still works; turning it on is a consent decision.
    enabled: bool = False


@dataclass
class MotionConfig:
    # Fraction of pixels that must change for "motion present".
    min_motion_fraction: float = 0.002
    # MOG2 history / threshold.
    mog2_history: int = 200
    mog2_var_threshold: float = 32.0
    # Pixel-intensity delta to count a pixel as "changed" in frame-diff fallback.
    diff_threshold: int = 25


@dataclass
class TriggerConfig:
    # VLM cooldown per scene-signature, seconds (§8 dampener).
    vlm_cooldown_seconds: float = 30.0
    # Observation cooldown so continuous presence doesn't spam memory (§G).
    observation_cooldown_seconds: float = 10.0
    # Frames of stillness that mark a "settled" keyframe (§8 keyframe selection).
    settle_frames: int = 3
    # Sensitivity: 'eager' | 'balanced' | 'lazy'. Spec biases toward 'lazy'
    # for a memory-builder (scene-diff is the safety net) (§8 tradeoff).
    sensitivity: str = "lazy"


@dataclass
class DetectConfig:
    yolo_weights: str = field(default_factory=resolve_yolo_weights)
    conf: float = 0.35
    iou: float = 0.5
    # COCO classes we treat as "objects of interest" (tracked classes, §B).
    # Everything else relies on the VLM describer for open-vocabulary naming.
    object_classes: list[str] = field(default_factory=lambda: [
        "chair", "laptop", "cell phone", "bottle", "cup", "book",
        "backpack", "handbag", "potted plant", "tv", "couch", "bed",
    ])
    tracker_cfg: str = "bytetrack.yaml"


@dataclass
class DistillConfig:
    # Weight update: bounded additive, scaled by observation confidence (§7).
    weight_increment: float = 0.15
    weight_cap: float = 1.0
    # Exponential decay half-life (days) for un-reinforced edges (§7).
    decay_half_life_days: float = 14.0
    # candidate -> confirmed threshold (§7).
    confirm_weight: float = 0.6
    # Live distillation cadence (seconds). Spec says "nightly"; for a live UI we
    # run it on a short timer so the world-graph fills in visibly. 0 = off.
    # ponytail: fixed interval; switch to change-triggered if a pass gets slow.
    live_interval_s: float = 60.0


@dataclass
class VLMConfig:
    # Anthropic model used for BOTH the describer [F] and reasoner [H] roles.
    model: str = "claude-opus-4-8"
    max_tokens: int = 1500
    # Falls back to a no-op/template describer when the SDK/key is absent.
    enabled: bool = bool(os.environ.get("ANTHROPIC_API_KEY"))


@dataclass
class Config:
    identity: IdentityConfig = field(default_factory=IdentityConfig)
    motion: MotionConfig = field(default_factory=MotionConfig)
    trigger: TriggerConfig = field(default_factory=TriggerConfig)
    detect: DetectConfig = field(default_factory=DetectConfig)
    distill: DistillConfig = field(default_factory=DistillConfig)
    vlm: VLMConfig = field(default_factory=VLMConfig)
    # Retention: drop raw frames/crops older than this many days (§11).
    raw_retention_days: int = 7


CONFIG = Config()

# --- App config (§10): version-controllable camera/zones, CLI flags override ---
APP_CONFIG_PATH = Path(os.environ.get("INTELLIGENCE_OS_CONFIG", ROOT.parent / "config.yaml"))


def load_app_config() -> dict:
    """Camera / zones / sensitivity from config.yaml (§10). Lets the app boot with
    no CLI args (the ten-minute path). A missing file just means 'use flags/defaults'."""
    if not APP_CONFIG_PATH.exists():
        return {}
    import yaml   # deferred; already a dependency (rules.py)
    return yaml.safe_load(APP_CONFIG_PATH.read_text()) or {}


def update_app_config(**kv) -> dict:
    """M6: persist settings into config.yaml (§10) and apply what can apply now.
    Camera changes need a restart; retention and identity take effect immediately."""
    import yaml
    cfg = load_app_config()
    cfg.update(kv)
    cfg = {k: v for k, v in cfg.items() if v is not None}   # None means "drop the key"
    APP_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    APP_CONFIG_PATH.write_text(yaml.safe_dump(cfg, sort_keys=False))
    apply_app_config(cfg)
    return cfg


def apply_app_config(cfg: dict | None = None) -> None:
    """config.yaml wins over the dataclass defaults for the settings the UI owns."""
    cfg = load_app_config() if cfg is None else cfg
    if "retention_days" in cfg:
        CONFIG.raw_retention_days = int(cfg["retention_days"])
    if "face_matching" in cfg:
        CONFIG.identity.enabled = bool(cfg["face_matching"])


def resolve_cameras(app_config: dict | None = None) -> list[dict]:
    """M6: normalize config into a list of {name, source} camera dicts.

    Three levels, first wins:
      1. cameras: [{name: front_door, source: "rtsp://..."}, ...]   # new multi-cam
      2. camera: 0 | "/path/..." | "rtsp://..."                     # old single-cam sugar
      3. (nothing) → [{name: "default", source: 0}]                 # webcam 0 fallback

    Every camera gets a unique name; duplicates are an error at boot, not a silent
    overwrite mid-run.
    """
    cfg = app_config if app_config is not None else load_app_config()

    if "cameras" in cfg and isinstance(cfg["cameras"], list) and cfg["cameras"]:
        cams = []
        for i, entry in enumerate(cfg["cameras"]):
            if isinstance(entry, dict) and "source" in entry:
                name = str(entry.get("name", f"cam_{i}")).strip()
                src = entry["source"]
                if isinstance(src, bool):
                    src = 0
                cams.append({"name": name, "source": src})
            else:
                # bare value (int or string) — auto-name
                src = entry if not isinstance(entry, bool) else 0
                cams.append({"name": f"cam_{i}", "source": src})
        # enforce unique names
        names = [c["name"] for c in cams]
        if len(names) != len(set(names)):
            dupes = [n for n in names if names.count(n) > 1]
            raise ValueError(f"Duplicate camera names in config: {set(dupes)}")
        return cams

    # backward compat: singular camera: key → one-element list
    if "camera" in cfg:
        src = cfg["camera"]
        if isinstance(src, bool):
            src = 0
        return [{"name": "default", "source": src}]

    # nothing at all → webcam 0
    return [{"name": "default", "source": 0}]


def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)


apply_app_config()   # settings saved by the UI survive a restart
