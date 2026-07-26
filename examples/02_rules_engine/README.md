# Example 2 — The rule cascade

**No camera, no API key.** Five seconds.

```bash
pip install -r requirements-dev.txt
python examples/02_rules_engine/run.py
```

## What it shows

A rule here is not an `if` statement over model output. It is a cascade, and it
has a trust boundary in front of it.

### The compiler refuses things, loudly

You write rules in English and one LLM call compiles them into a spec. That call
needs an API key, so this example feeds the compiler's *output* straight into
`_apply_gates` — the same trust boundary the real compiler passes through, and
the same thing the acceptance tests exercise.

Two refusals, both deliberate:

| You ask for | Response | Why |
|---|---|---|
| "alert me if someone looks stressed" | `Refusal(infeasible)` | The camera cannot reliably observe it. A rule that fires on a guess is worse than no rule: it trains you to ignore alerts. |
| "tell me when Raj leaves" | `Refusal(needs_identity)` | It targets a named individual, and face identity is **off by default**. Biometric matching gets turned on consciously, not as a side effect of writing a rule. |

A refusal explains itself and names the missing capability. It never silently
degrades into a rule that watches nothing.

### The cascade gates the expensive stages

```
zone match  →  dwell gate  →  [optional VLM verify]  →  per-subject cooldown  →  alert
   free          free              costs money              free
```

The script feeds a person standing in the loading bay at t+0, +10, +29, +31 and
+45 seconds against a 30-second dwell gate. It fires **once**, at t+31. Walking
through the frame does not trip it; standing there does. Then a 300-second
per-subject cooldown means one alert per subject, not one per frame.

If a rule needs a VLM check ("is that person carrying a package?"), the verify
step only runs on the handful of frames that already passed the free gates. With
no verifier configured, verify-rules simply don't fire — precision over recall.

### An alert is a graph object, not a log line

A fired rule writes an observation with `origin='rule'` and a keyframe. So it
appears in the timeline, it can be dismissed by an operator, and — this is the
interesting part — **a dismissal becomes a negative example** fed back into the
verifier, so the rule sharpens with use from the next pipeline start.

## Writing rules for real

With `ANTHROPIC_API_KEY` set:

```bash
python -m intelligence_os.rules add "tell me if a vehicle stops at the gate for more than a minute"
python -m intelligence_os.rules list
```

Compiled rules live in `intelligence_os/data/rules.yaml`, can be enabled and
disabled individually, and are also editable from the Rules pane in the
dashboard.

**A rule whose zone doesn't exist can never fire.** The engine warns about that
at arm time rather than watching nothing in silence — if you see a zone warning
on startup, draw the zone or fix the name.
