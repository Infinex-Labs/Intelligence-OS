# Examples

Four worked examples, ordered so that the ones needing the least hardware come
first. Every one of them runs against a throwaway database in `/tmp` — none of
them touch your real memory graph.

| | Example | Needs | Time |
|---|---|---|---|
| 1 | [The memory graph](01_memory_graph/) — observations → habits → provenance → correction → deletion | Nothing but `requirements-dev.txt` | 5 s |
| 2 | [The rule cascade](02_rules_engine/) — why rules get refused, and how the dwell gate works | Nothing but `requirements-dev.txt` | 5 s |
| 3 | [The video pipeline](03_video_pipeline/) — run the real cascade on a clip and watch the dashboard | Full `requirements.txt` (downloads YOLO) | 10 min |
| 4 | [Multiple cameras](04_multi_camera/) — one memory, several streams, per-camera zones | Two RTSP streams or two clips | 15 min |

Start with 1 even if you came for the cameras. It is the shortest path to
understanding what this project actually is: the vision stages are replaceable
glue, and the graph is the product.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt

python examples/01_memory_graph/run.py
python examples/02_rules_engine/run.py
```

Both print their reasoning as they go, and both end by telling you which
scratch directory they used.
