# Example 1 — The memory graph

**No camera, no API key, no model weights.** Three packages and five seconds.

```bash
pip install -r requirements-dev.txt
python examples/01_memory_graph/run.py
```

## What it shows

The script fabricates three mornings of observations for two people and a
laptop — exactly the rows the pipeline would have written downstream of the
camera — and then does everything that happens *after* that point.

**1. The transcript.** Every observation is `subject → predicate → object` at a
location and a timestamp, carrying the keyframe it came from and the camera that
saw it. Predicates are open strings, not an enum, because the VLM says things a
fixed schema wouldn't have anticipated.

**2. Distillation.** `Distiller.run()` — the same pass
`python -m intelligence_os.distill` performs — turns repetition into durable
memory. Repeated presence in a place at the same hour on several days becomes a
`habit`; repeated proximity to an object becomes a `uses` relation; repeated
presence becomes `frequents`. Each edge carries a `weight` and a
`candidate → confirmed` lifecycle, and decays if it stops being reinforced.

**3. Provenance.** The part that matters. A habit is not an opinion the model
formed, it is a pointer to the specific observations and keyframes that produced
it. The script walks a claim back down to its evidence. **If you add a new kind
of claim to this system, it needs this property or it doesn't belong.**

**4. Correction.** The graph is wrong sometimes — bad lighting fragments one
person into two. `store.merge(src, dst)` fuses them and the evidence *moves*
rather than being discarded, so nothing is lost and the merge can be undone with
`split`.

**5. Deletion.** `cascade_delete` removes an entity's signatures, observations
and relations together. It is the privacy-removal path, it is meant to be
irreversible, and the script shows the graph genuinely emptying out.

## Things worth noticing in the output

- The visitor has no name — entities are **anonymous by default** and only get a
  label when an operator gives them one.
- The habit is `present_around_09h`, in **UTC**: `mine_habits` buckets by
  UTC hour-of-day, so the printed times are UTC too.
- The habit lands as `candidate`, not `confirmed`, on three days of evidence.
  Confidence is earned, not assumed.

## Try changing

- Seed a fourth and fifth day and watch the habit's weight cross into
  `confirmed`.
- Seed nothing for a week (`NOW - 10 * DAY`) and call `store.decay_relations()`
  to watch an un-reinforced habit fade — `DistillConfig.decay_half_life_days`
  controls how fast.
- Give the second person a `label` and re-run: the merge in step 4 becomes a
  much more consequential operation, which is exactly why it needs `split`.
