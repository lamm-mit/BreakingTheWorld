# BreakingTheWorld

Agentic DAG world-model discovery with MDL scoring.

The discovery runner lives in `src/world_model_breaker_cli.py`. The notebook is
an interactive runner/visualizer around that library code. Synthetic data
generation is separated from discovery so future agents can inspect, modify, or
replace the oracle.

## What MDL Means

MDL is Minimum Description Length. The runner accepts a new world model only if
it gives a shorter total explanation of the revealed evidence:

```text
L_total = L_model + L_data
```

`L_model` is the bit cost of describing the DAG structure and fitted parameters.
`L_data` is the bit cost of the residual error left after the model predicts the
revealed data. This means a more complex hypothesis must earn its keep by
reducing residual error enough to lower the total bit count.

## What Was Implemented

- `src/discovery_data.py`: reusable `Observation` and `DiscoveryDataset` types plus JSON/CSV I/O.
- `src/tensile_test_oracle.py`: synthetic tensile-system simulators/oracles.
- `src/oracle_adapters.py`: live oracle protocol plus the Duffing oscillator simulator.
- `src/create_synthetic_dataset.py`: CLI for creating reusable dataset artifacts.
- `src/world_model_breaker_cli.py`: discovery CLI that can either generate a built-in synthetic dataset or load a saved `DiscoveryDataset` JSON.
- `visualize_discovery.ipynb`: thin notebook runner that calls the library discovery workflow.

## Synthetic Examples

### Cyclic Tensile Test

Physical sequence:

```text
elastic loading -> post-yield loading -> unloading -> reloading
```

Reloading starts continuously from the final unloading stress.

Generate:

```bash
python src/create_synthetic_dataset.py \
  --example cyclic \
  --outdir data/tensile_cyclic \
  --seed 7 \
  --noise-std 0.9
```

Run discovery from saved data:

```bash
python src/world_model_breaker_cli.py \
  --dataset-json data/tensile_cyclic/dataset.json \
  --outdir runs/tensile_cyclic \
  --no-llm
```

Or run directly from the built-in oracle:

```bash
python src/world_model_breaker_cli.py \
  --synthetic-example cyclic \
  --outdir runs/tensile_cyclic \
  --no-llm
```

### Live Duffing Oscillator Oracle

This example does not require creating a dataset first. The Breaker chooses the
next trajectory/action, then the live oracle integrates a hidden nonlinear
oscillator and returns newly revealed samples.

Hidden simulator family:

```text
dx/dt = v
dv/dt = -k*x - c*v - alpha*x^3 + force(t)
```

The Builder sees only collected samples with observables `x`, `v`, `force`, and
`time`; the target is `dvdt`.

Offline smoke run:

```bash
python src/world_model_breaker_cli.py \
  --oracle duffing \
  --outdir runs/duffing_live_offline \
  --no-llm \
  --rounds 3 \
  --search-steps 320 \
  --search-restarts 10 \
  --noise-std 0.01
```

LLM-assisted run:

```bash
python src/world_model_breaker_cli.py \
  --oracle duffing \
  --outdir runs/duffing_live_llm \
  --model gpt-5.5 \
  --llm-builder \
  --rounds 3 \
  --search-steps 320 \
  --search-restarts 10 \
  --noise-std 0.01
```

## Real Protein Normal-Mode Example

This example uses real PDB structures. Each residue becomes one observation:
the inputs are C-alpha contact-network and normal-mode features, and the target
is the residue's experimental C-alpha B-factor normalized within that protein.
The physics calculation is a lightweight Gaussian Network Model (GNM) implemented
in this repo, so ProDy is not required for the first version.

Build only the staged dataset:

```bash
python src/protein_world_model_cli.py build \
  --outdir data/protein_flex
```

Build and run the DAG+MDL world-model breaker (offline, no LLM):

```bash
python src/protein_world_model_cli.py run \
  --outdir runs/protein_flex \
  --no-llm \
  --rounds 3 \
  --search-steps 260 \
  --search-restarts 6
```

LLM-assisted run (Builder reasons from experiment context and proposes
candidate DAG features; MDL remains the acceptance rule):

```bash
python src/protein_world_model_cli.py run \
  --outdir runs/protein_flex_llm \
  --llm-builder \
  --rounds 3 \
  --search-steps 260 \
  --search-restarts 6
```

Deep-reasoning run (higher LLM effort, more search restarts and patience):

```bash
python src/protein_world_model_cli.py run \
  --outdir runs/protein_flex_llm_deep \
  --llm-builder \
  --reasoning-effort high \
  --rounds 3 \
  --search-steps 400 \
  --search-restarts 12 \
  --search-patience 50
```

Use custom stages:

```bash
python src/protein_world_model_cli.py run \
  --stage compact=1ubq:A,1crn:A \
  --stage termini=1aar:A,2ci2:I \
  --stage hinge=4ake:A,1ake:A \
  --outdir runs/protein_custom \
  --no-llm
```

The staged discovery story is:

```text
compact proteins -> terminal/boundary failures -> hinge/domain-motion failures -> validation
```

The DAG is reused directly. It searches over physically meaningful protein
features such as `gnm_fluct_z`, `contact_degree_z`, `terminal_exposure`,
`hinge_score_z`, and residue-class indicators, accepting new feature structure
only when it lowers MDL.

### Tensile Fracture/Release Test

Physical sequence:

```text
elastic loading -> strain hardening -> fracture softening -> release
```

Fracture softening is continuous with hardening, and release is continuous with
the fractured residual stress.

Generate:

```bash
python src/create_synthetic_dataset.py \
  --example fracture \
  --outdir data/tensile_fracture \
  --seed 11 \
  --noise-std 0.7
```

Run discovery from saved data:

```bash
python src/world_model_breaker_cli.py \
  --dataset-json data/tensile_fracture/dataset.json \
  --outdir runs/tensile_fracture \
  --no-llm
```

Or run directly from the built-in oracle:

```bash
python src/world_model_breaker_cli.py \
  --synthetic-example fracture \
  --outdir runs/tensile_fracture \
  --no-llm
```

## LLM Run

With an API key, the Breaker can use the LLM to choose experiments and the
Builder can reason about the new evidence, use the experiment description and
its scientific world knowledge, propose latent/internal variables, make
falsifiable predictions, and seed symbolic DAG feature proposals:

```bash
export OPENAI_API_KEY=...
python src/world_model_breaker_cli.py \
  --dataset-json data/tensile_cyclic/dataset.json \
  --outdir runs/tensile_cyclic_llm \
  --model gpt-5.5 \
  --llm-builder \
  --search-steps 320 \
  --search-restarts 10
```

At startup, the runner prints and writes a pre-run dataset analysis. With LLM
enabled, this is generated from dataset metadata without hidden target values.
Offline runs use a deterministic summary. Disable the LLM version with:

```bash
--no-llm-dataset-analysis
```

## Discovery Semantics

After each collected slice, the Builder sees only the revealed evidence: the
current equation, residual summaries by revealed protocol slice, observable
descriptions, protocol descriptions, and the system description. With
`--llm-builder`, it writes a structured hypothesis record containing:

- Diagnosis of what the current model fails to explain.
- Physical interpretation using the known experiment context.
- Candidate latent/internal variables that may be missing from the current observables.
- Candidate symbolic DAG features expressible by the current feature alphabet.
- Falsifiable predictions for future evidence.

The Builder's reasoning is logged, but MDL is still the acceptance rule. A
proposal survives only if refitting it reduces `L_total`.

By default the Breaker uses:

```bash
--breaker-mode experimental
```

It sees only collectable protocol metadata and current-model predictions,
states a falsification hypothesis, then collects one oracle slice. The legacy
hidden-label selector is available with:

```bash
--breaker-mode oracle
```

The default:

```bash
--collection-policy physical
```

enforces dataset-defined `stage_prerequisites`. Use this for physically ordered
experiments. The old adversarial behavior is:

```bash
--collection-policy unconstrained
```

## Artifacts

Each run writes:

```text
<outdir>/
  config.json
  metrics.csv
  report.md
  run_summary.json
  world_model_iter_*.json
  frame_iter_*.png
  evolution.gif
  agent_logs/
  paper_figures/
    mdl_trajectory.svg
    discovery_timeline.svg
    model_evolution.svg
    dag_evolution.svg
```

## Custom Systems

For other systems, construct a `DiscoveryDataset` with your own:

- `Observation` records.
- `stage_order`.
- `initial_stage_ids`.
- `stage_prerequisites`.
- `system_description`.
- `observable_descriptions`.
- `target_description`.
- `protocol_descriptions`.

Then either pass it directly:

```python
runner = DiscoveryRunner(args, dataset=my_dataset)
```

or write it with `write_dataset_json(...)` and use:

```bash
python src/world_model_breaker_cli.py --dataset-json path/to/dataset.json --outdir runs/my_system
```

For live simulators or lab-backed data sources, implement an oracle with a
`collect_stage(stage_id, ...)` method that returns `Observation` rows. For the
new live path, implement the `OracleAdapter` protocol in `src/oracle_adapters.py`:

- `dataset_shell()` returns system/protocol metadata without observations.
- `var_space()` returns the observable variables available to the world model.
- `preview_stage(stage_id, ...)` returns observable inputs for planning without hidden targets.
- `collect_stage(stage_id, ...)` runs the simulator/experiment and reveals targets.

The saved-dataset route remains useful for reproducible snapshots. The live
oracle route is the preferred abstraction for open-ended active discovery.
