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


# --- Detection vocabulary ----------------------------------------------------
# The 80 COCO classes the shipped YOLO weights carry, in model order. Kept here
# rather than read off the model so `object_classes` can be validated at config
# load time, before ultralytics is imported (and without a GPU spin-up).
COCO_CLASSES: tuple[str, ...] = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
)

# Kind of thing each class is. This is NOT `entities.type` — that column is
# CHECK-constrained to person|object and widening it means rebuilding the table.
# Kind is the behavioural discriminator: it picks the relation verb the distiller
# may assert, and decides whether cross-day appearance re-ID is honest for the
# class (see ANIMATE_KINDS below).
_ANIMALS = frozenset({"bird", "cat", "dog", "horse", "sheep", "cow",
                      "elephant", "bear", "zebra", "giraffe"})
_VEHICLES = frozenset({"bicycle", "car", "motorcycle", "airplane", "bus",
                       "train", "truck", "boat"})

# Kinds that move under their own power (or someone else's). The appearance
# signature is an HS colour histogram, which separates a red chair from a blue
# one but NOT one black dog from another, and its cross-day use rests on an
# assumption stated in detect.py: objects are static. That assumption is false
# for these, so re-matching them across restarts would merge distinct subjects
# into one entity — asserting an identity nothing observed. We mint instead.
ANIMATE_KINDS: frozenset[str] = frozenset({"person", "animal", "vehicle"})


def kind_for_class(cls_name: str) -> str:
    """person | animal | vehicle | object for a COCO class name."""
    if cls_name == "person":
        return "person"
    if cls_name in _ANIMALS:
        return "animal"
    if cls_name in _VEHICLES:
        return "vehicle"
    return "object"


def normalize_object_classes(names) -> tuple[list[str], list[str]]:
    """(kept, rejected) for a user-supplied class list.

    A class the model does not carry can never be detected, so a typo like 'dogs'
    would silently watch nothing. Same failure the rule compiler refuses loudly
    for (§9.2) — the caller surfaces `rejected` rather than dropping it.
    """
    kept, rejected, seen = [], [], set()
    for raw in names or []:
        n = str(raw).strip().lower()
        if not n or n in seen:
            continue
        seen.add(n)
        (kept if n in COCO_CLASSES else rejected).append(n)
    return kept, rejected


@dataclass
class DetectConfig:
    yolo_weights: str = field(default_factory=resolve_yolo_weights)
    conf: float = 0.35
    iou: float = 0.5
    # COCO classes we treat as "objects of interest" (tracked classes, §B).
    # Everything else relies on the VLM describer for open-vocabulary naming.
    # Overridable from config.yaml (`object_classes:`) — see apply_app_config.
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
    # How far back a mining pass reads (days; 0 = all of history). A nightly job
    # that scans everything gets slower every night forever, and the extra it
    # reads is history whose edges it already mined. Measured back from the
    # newest observation rather than the wall clock, so a memory that was idle
    # over the weekend still mines the week it actually has.
    mine_window_days: float = 120.0


@dataclass
class SemanticConfig:
    """Meaning-based retrieval (search plan Phase 6).

    Optional in the same sense `insightface` is: with no local embedding model
    installed the layer produces nothing and search is exactly Phase 2's lexical
    behaviour. Nothing here downgrades an answer that already worked — semantic
    hits are FUSED with the lexical ones, never substituted for them.
    """
    # The master switch, separate from whether the model is installed. It exists
    # so a deployment that dislikes what this layer surfaces can turn it off in
    # config rather than uninstalling a package or rolling back a release.
    #
    # The env var is how "does this still work without the dependency?" gets
    # answered on a machine that HAS the dependency. That question needs a real
    # answer on every run rather than once, on a laptop, by uninstalling things
    # — which is a check nobody repeats.
    enabled: bool = not os.environ.get("INTELLIGENCE_OS_NO_SEMANTIC")
    # Named, not just loaded: the name is stored beside every vector, so a model
    # swap is detectable rather than silently mixing two incompatible spaces in
    # one index.
    #
    # L12 rather than the smaller L6, and rather than a stronger retrieval model
    # like bge-small, because this index has to do something most benchmarks do
    # not measure: SAY NO. Measured on the eval corpus (docs/search-baseline.md):
    #
    #   model      worst true hit   best score for 'dog', which is not there
    #   L6                  0.276                                      0.248
    #   L12                 0.375                                      0.233
    #   bge-small           0.521                                      0.494
    #
    # L6 puts a real answer and pure noise 0.03 apart, so no floor separates
    # them. bge scores everything highly — it is trained to RANK, and a ranker
    # asked for an absolute yes/no has nothing to give. L12 leaves a gap wide
    # enough to put a threshold in, for ~120MB instead of ~90MB.
    model: str = "sentence-transformers/all-MiniLM-L12-v2"
    # Cosine below this is not a match, and this is the one score in the system
    # that is ALLOWED to gate. bm25 only ever orders results, because its
    # magnitude is relative to the corpus and says nothing on its own. Cosine is
    # absolute and comparable across queries, so without a floor every question
    # returns the nearest row in the memory whether or not anything answers it —
    # which is how a semantic layer invents evidence.
    #
    # 0.30 sits in the middle of the gap in the table above. It is a property of
    # the model, not of this corpus: change the model and re-measure, which is
    # what `model` being a stored column is for.
    min_similarity: float = 0.30
    # How many semantically-ranked rows enter the fusion.
    top_k: int = 200
    # RRF's rank offset, the constant from the original paper. Deliberately not
    # tuned: its job is to flatten the gap between rank 1 and rank 2 enough that
    # neither list can dominate the other off a single confident hit, and a
    # value fitted to this corpus would stop doing that on someone else's.
    rrf_k: float = 60.0
    # Rows about the same subject and predicate closer together than this are one
    # hit, not several. A settled scene re-described every few seconds otherwise
    # fills an answer with the same sentence, and the count reads as recurrence.
    collapse_seconds: float = 120.0
    # Rows handed to the encoder per call, and vectors held in memory per chunk
    # while scanning. Both are bounded on purpose: an embedding is 1.5KB and a
    # year of prose is not something to load in one list.
    batch_size: int = 128
    scan_chunk: int = 4096
    # Keyframes shown per entity. Chosen for time coverage rather than by taking
    # the first `n` — see `ask._diverse_keyframes`.
    max_keyframes: int = 4


@dataclass
class ReflectionConfig:
    """The relaxation ladder (search plan Phase 7).

    Empty used to be final, which made "I could not find it" and "it did not
    happen" the same reply — the worst failure this system has, because it is
    invisible. When a question comes back with nothing, the ladder loosens ONE
    constraint at a time, retries, and stops at the first result.

    Three rules keep it from relaxing its way into inventing an answer:

      it only ever runs on an EMPTY result, so a question that was answered is
      bit-for-bit unchanged;

      it only loosens constraints that RESOLVED to something memory knows — a
      zone that exists, a label that names a real subject. Dropping a filter
      that named nothing is not widening the search, it is abandoning it, and
      that is how "was Mallory here?" would come back with a photograph of
      somebody else;

      it never touches an exclusion. "Anyone except the courier" is a
      constraint on the answer, and widening is allowed to add candidates,
      never to overrule what was ruled out.

    Everything it does is recorded in the trace and disclosed in the prose.
    """
    # As with the semantic layer: an env kill switch, so "does this still
    # behave like Phase 6?" is answered on every run rather than by reasoning
    # about it. With this off, a dead end is a dead end again.
    enabled: bool = not os.environ.get("INTELLIGENCE_OS_NO_RELAX")
    # At most this many rungs are tried. A ladder with no top is a search that
    # eventually returns the whole table and calls it an answer.
    max_steps: int = 3
    # Window widening, per side. The multiplier is what makes "a bit either
    # side" mean something for a six-hour question; the cap is what stops it
    # meaning something absurd for a six-WEEK one. Without the cap, a question
    # about last month widens by a fortnight and answers about a different
    # month — and "was anyone there next week?" quietly reaches back into
    # everything ever recorded. It is the cap, not the multiplier, that makes
    # the true-negative guard hold by construction.
    widen_factor: float = 1.5
    widen_cap_s: float = 6 * 3600.0


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
    semantic: SemanticConfig = field(default_factory=SemanticConfig)
    reflect: ReflectionConfig = field(default_factory=ReflectionConfig)
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
    if "semantic_search" in cfg:
        # A kill switch that does not need a redeploy. Turning it off leaves the
        # stored vectors alone, so turning it back on costs nothing — the point
        # is to stop *reading* them, not to throw the index away.
        CONFIG.semantic.enabled = bool(cfg["semantic_search"])
    if "widen_empty_searches" in cfg:
        # Off means a question that matches nothing answers "nothing", full
        # stop. Some deployments want exactly that: a loosened answer is a
        # correct answer to a question nobody asked, and disclosure is only
        # worth something if somebody reads it.
        CONFIG.reflect.enabled = bool(cfg["widen_empty_searches"])
    if "object_classes" in cfg:
        # An empty/absent list means "keep the defaults"; an explicit list wins.
        # `person` is always detected and is not part of this list (detect.py adds
        # it), so silently drop it rather than let it look like a togglable class.
        kept, rejected = normalize_object_classes(cfg["object_classes"])
        if rejected:
            print(f"[config] ignoring unknown detection class(es): "
                  f"{', '.join(rejected)} — not in the model's 80 COCO classes")
        kept = [c for c in kept if c != "person"]
        if kept:
            CONFIG.detect.object_classes = kept


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
