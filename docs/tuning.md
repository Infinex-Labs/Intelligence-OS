# Tuning

Three things are worth tuning, in this order of consequence.

---

## 1. The identity threshold

`IdentityConfig.face_match_threshold`, default `0.45`.

This is the number the entire memory graph rests on. Face embeddings are
compared by cosine similarity; above the threshold, a face is judged to be
someone already in the gallery ("match"), below it, a new entity is created
("mint").

| Threshold too **low** | Threshold too **high** |
|---|---|
| Two different people are fused into one entity. | One person fragments into many entities. |
| The graph confidently attributes A's habits to B. | Nothing recurs often enough to become a habit. |
| **Worse failure** — it is silent and it corrupts history. | Annoying but visible, and fixable with `merge`. |

Given the choice, err high. A fragmented graph looks wrong and gets corrected; a
fused one looks fine and lies.

### How to actually pick it

Do not use the default and hope. Record a clip that contains a known number of
people — say four colleagues, a minute, in the lighting you actually have — and
sweep:

```bash
python -m intelligence_os.phase1 tune --video enroll_clip.mp4
```

Illustrative output (abridged — the real sweep runs 0.20 → 0.60 in 0.05 steps):

```
[tune] 218 face embeddings extracted.
[tune] threshold -> #entities formed (pick the one matching #people you know are in the clip):
  threshold 0.20 ->   1 entities
  threshold 0.25 ->   1 entities
  threshold 0.35 ->   3 entities
  threshold 0.45 ->   4 entities     ← four people were in the clip
  threshold 0.55 ->   9 entities
  threshold 0.60 ->  17 entities
```

The sweep replays the same online match-or-mint logic the pipeline uses, so the
count genuinely predicts how many entities that threshold would form. Pick the
value that matches the truth, and prefer the **higher** end of a plateau.

### Then verify separation

```bash
python -m intelligence_os.phase1 run --webcam 0        # accumulate for a while
python -m intelligence_os.phase1 entities
python -m intelligence_os.phase1 separation
```

`separation` prints a cross-entity cosine matrix. **Distinct people should be
low.** Two entities sitting at 0.5 against each other are one bad frame away
from being fused — either your threshold is too low or you need more gallery
variety for those two.

### Related knobs

- `max_signatures_per_entity` (12) — the gallery is capped per person.
- `signature_novelty_min` (0.10) — a new vector is only stored if it differs
  from existing ones by at least this much, so you keep pose and lighting
  variety rather than twelve near-identical frontal shots. **Variety in the
  gallery matters more than volume.**
- `min_face_det_score` (0.55) — faces below this are not used for identity at
  all. Raise it if a busy scene is producing junk embeddings.

### Retuning

Redo this when the lighting changes materially (a new camera, seasonal daylight,
new IR illuminators at night), or when you add people who look alike. A
threshold tuned in July daylight is not a threshold for December.

---

## 2. VLM cost

The VLM is the only stage that costs money, and two settings bound it.

`TriggerConfig.sensitivity` — how eagerly stage E decides the scene changed
enough to describe:

| | Behaviour | Use when |
|---|---|---|
| `lazy` | Describes only clear changes. The default, and the right bias for a long-running memory builder — scene-diff is the safety net. | Weeks of unattended running |
| `balanced` | The web app's default: responsive enough for a live demo. | Watching the dashboard |
| `eager` | Describes readily. | Debugging the describer itself |

`TriggerConfig.vlm_cooldown_seconds` (30s) is a hard floor between calls per
scene signature. This is your ceiling: with a 30-second cooldown, one camera
cannot exceed 120 calls an hour no matter what happens in front of it. Raise it
first if the bill surprises you.

`observation_cooldown_seconds` (10s) is the equivalent dampener for memory
writes — it stops a person standing still from writing an identical row every
frame.

**Everything except description works with no key at all.** If cost is the
concern, running with no `ANTHROPIC_API_KEY` is a legitimate configuration, not
a degraded one.

---

## 3. Motion and detection

`MotionConfig.min_motion_fraction` (0.002) is the fraction of pixels that must
change before a frame is considered worth processing:

- **Raise it** when a swaying tree, a flickering light, or rain holds the gate
  permanently open — you will see constant activity and high CPU with nothing
  detected.
- **Lower it** when slow or distant movement is being missed entirely.

`DetectConfig.conf` (0.35) is the YOLO confidence floor. Raising it reduces
phantom detections at the cost of missing partially-occluded people. It is a
blunter instrument than the motion gate — try the gate first.

`DetectConfig.object_classes` is the list of COCO classes tracked as objects of
interest. Anything not in it relies on the VLM for naming. Adding classes here
is one of the easiest useful contributions to the project.

---

## Habit decay

`DistillConfig.decay_half_life_days` (14) governs how fast an un-reinforced edge
fades, and `confirm_weight` (0.6) is where a `candidate` becomes `confirmed`.

Shorten the half-life for a space whose patterns change fast (a shared office);
lengthen it where a habit is genuinely weekly rather than daily, or a real
pattern will decay between reinforcements and never confirm.

You can watch this directly — see
[`examples/01_memory_graph`](../examples/01_memory_graph/), which prints weights
and lifecycle status as habits form.
