# Synthetic Tensile-Test Dataset: cyclic

Synthetic one-dimensional tensile-test experiment. The observable inputs are engineering strain and loading direction; the target is measured stress. The physical protocol is ordered: begin with elastic loading, continue into post-yield loading, unload from the high-strain state, then reload from the partially unloaded state. Later protocol stages are not physically collectable until their prerequisites have occurred, so data discovery should respect this experimental sequence rather than freely jumping to any hidden slice.

Files:
- `dataset.json`: full `DiscoveryDataset` with metadata, stage prerequisites, and observations.
- `observations.csv`: flat observation table for inspection/plotting.

Use with discovery CLI:

```bash
python src/world_model_breaker_cli.py --dataset-json /Users/mbuehler/LOCALCODES/BreakingTheWorld/data/tensile_cyclic/dataset.json --outdir demo_run
```

Stage transitions are physically ordered and continuity is enforced at branch handoffs.