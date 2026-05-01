# BreakingTheWorld

Agentic DAG world-model discovery with MDL scoring. Code companion to the paper
*"Why We Must Break the World"* (Buehler, 2026).

## Motivation

Discovery requires breaking the current world model to build the next one. This
repository implements a concrete version of that idea: two agents -- the
**Breaker** and the **Builder** -- collaborate adversarially over a shared
symbolic world model. The Breaker designs experiments to falsify the current
model. The Builder revises the model to explain the newly revealed evidence.
Every revision must pass a quantitative gate: Minimum Description Length (MDL).

The system works on real scientific data (protein structures, crystallographic
B-factors) and on synthetic benchmarks (tensile mechanics, nonlinear
oscillators). The key design choice is that the world model is an explicit,
interpretable equation -- not a black box -- and new structure is accepted only
when it earns its complexity cost in bits.

## Architecture

### The Breaker-Builder Loop

Each discovery run proceeds in iterations:

1. **Iteration 0 (Build)**: The Builder fits an initial DAG equation on the
   first revealed data stage.
2. **Iteration 1..N (Break + Rebuild)**: The Breaker selects the next
   experiment/data stage most likely to falsify the current model, states a
   hypothesis, and collects the data. The Builder then searches for a revised
   DAG on all revealed data. The revision is accepted only if it lowers total
   MDL bits.

With `--llm-builder`, the Builder LLM reasons about the system physics,
diagnoses residual patterns, proposes candidate latent variables, and seeds
symbolic DAG features. Without LLM (`--no-llm`), the system uses deterministic
heuristic seeds. In both cases, a stochastic hill-climb explores further
structural edits, and **MDL is the sole acceptance criterion**.

### The DAG World Model

The world model is a linear combination of features:

```text
y_hat = beta_0 + sum_i beta_i * f_i(x)
```

Each feature `f_i` is a product of 1-4 primitive factors drawn from:

| Factor | Form | Applies to |
|---|---|---|
| `Ident(v)` | `v` | continuous variables |
| `Pow(v, k)` | `v^k` (k=1..4) | continuous variables |
| `IndEq(v, a)` | `1 if v==a else 0` | discrete variables |
| `IndLE(v, t)` | `1 if v<=t else 0` | continuous variables |
| `ReLU(v, t)` | `max(v-t, 0)` | continuous variables |

This factor algebra is expressive enough to represent piecewise-linear models,
polynomial terms, regime splits, and interactions, while remaining compact
enough that MDL can meaningfully score structure against fit.

### MDL Scoring

MDL (Minimum Description Length) is the principled tradeoff between parsimony
and fit. Total description length is:

```text
L_total = L_model + L_data
```

- `L_model`: bit cost of the DAG structure (number of features, factor types,
  variable selections, thresholds) plus fitted coefficients.
- `L_data`: bit cost of the residual error under a Gaussian coding scheme.

A more complex model has higher `L_model` but can reduce `L_data`. The model is
accepted only when the net `L_total` decreases. This prevents overfitting
without an external validation set.

### Stages and Oracles

Data are organized into **stages** -- physically ordered experimental regimes
(e.g., elastic loading before unloading, compact proteins before hinge proteins).
Stages have `prerequisites` that enforce the physical collection order. The
Breaker can only request stages whose prerequisites have been collected.

An **oracle** is the data source. It can be:
- A **saved dataset** (`--dataset-json`): all observations pre-generated.
- A **live oracle** (`--oracle duffing`): the simulator runs on demand and
  reveals targets only when the Breaker collects a stage.

### Search Parameters

The symbolic DAG search is a stochastic hill-climb with these controls:

| Flag | Default | Meaning |
|---|---|---|
| `--search-steps` | 160/260 | Max DAG edit proposals per hill-climb run |
| `--search-restarts` | 3/6 | Independent hill-climbs per iteration; best is kept |
| `--search-patience` | 30/35 | Stop early after this many consecutive rejections |
| `--rounds` | 3 | Breaker collect + Builder rebuild cycles after iteration 0 |
| `--reasoning-effort` | medium | LLM reasoning depth (low/medium/high) |

More restarts and steps give the search a better chance of finding the global
MDL minimum at the cost of wall-clock time.

## What Was Implemented

- `src/discovery_data.py`: reusable `Observation` and `DiscoveryDataset` types plus JSON/CSV I/O.
- `src/dag_model.py`: DAG world model with factor algebra, least-squares fitting, and MDL scoring.
- `src/dag_search.py`: stochastic hill-climb over DAG structures with add/remove/swap/perturb operators.
- `src/world_model_breaker_cli.py`: main discovery CLI with Breaker/Builder agents, rendering, and artifact generation.
- `src/protein_nma_oracle.py`: real-protein GNM feature pipeline (PDB parsing, Kirchhoff matrix, normal modes, B-factor normalization).
- `src/protein_world_model_cli.py`: protein-specific CLI that builds datasets from PDB structures and runs discovery with auto-generated reports and figures.
- `src/tensile_test_oracle.py`: synthetic tensile-system simulators (cyclic and fracture).
- `src/oracle_adapters.py`: live oracle protocol plus the Duffing oscillator simulator.
- `src/create_synthetic_dataset.py`: CLI for creating reusable dataset snapshots.
- `src/create_protein_flex_dataset.py`: standalone CLI for building protein flexibility datasets.

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

This is the primary demonstration for the paper. It runs the full
build-break-rebuild loop on real scientific data: protein structures downloaded
from the Protein Data Bank.

**Physics pipeline**: Each protein chain is represented as a C-alpha contact
network. A Gaussian Network Model (GNM) -- the Kirchhoff matrix of the contact
graph -- is diagonalized to extract normal-mode fluctuation features. These
encode how much each residue is predicted to move based purely on contact
topology and elastic-network mechanics. The target is the residue's experimental
crystallographic B-factor (thermal displacement), z-scored within each chain.

**Staged discovery story**: The system starts with compact single-domain
proteins where GNM should work well, then reveals proteins with flexible termini
(probing boundary effects), hinge/domain-motion proteins (probing collective
modes), and finally a mixed validation set. At each stage, the current DAG
equation is challenged and revised only if MDL supports the added structure.

The GNM implementation is self-contained in this repo (no ProDy dependency).
PDB files are cached locally in `data/pdb_cache/`.

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
  config.json              # full CLI args and dataset metadata
  metrics.csv              # quantitative trajectory (RMSE, R2, MDL bits per iteration)
  report.md                # narrative report with equations, breaks, and interpretations
  run_summary.json         # machine-readable summary of the full run
  world_model_iter_*.json  # DAG snapshot at each iteration (structure, coefficients, bits)
  frame_iter_*.png         # multi-panel visualization of each iteration (see below)
  evolution.gif            # animated sequence of all iteration frames
  agent_logs/              # raw LLM request/response logs and search traces
  paper_figures/
    mdl_trajectory.svg     # MDL budget (L_model + L_data) across iterations
    discovery_timeline.svg # Breaker collection sequence with break types
    model_evolution.svg    # data + model fit panels side by side
    dag_evolution.svg      # DAG graph structure at each iteration
```

The protein CLI additionally generates:

```text
  protein_world_model_detailed_report.md   # interpretive report with feature explanations
  protein_world_model_report.tex           # LaTeX report with figures and tables
  protein_world_model_report.pdf           # compiled PDF (if pdflatex is available)
  protein_integrated_discovery_figure.pdf  # single summary figure for the paper
  report_assets/                           # per-protein 3D structure visualizations
```

### Iteration Frame Panels (`frame_iter_*.png`)

Each frame contains six panels that together provide a full audit of one
discovery iteration:

- **Top-left (scatter)**: Predicted vs. experimental target. For proteins this
  is DAG prediction vs. B-factor z, colored by revealed stage, with an inset
  RMSE-by-stage bar chart. The equation and MDL summary are overlaid.
- **Top-right (graph)**: The DAG structure as a directed graph. Blue = observable
  inputs, orange = primitive factors, green = product features, purple = target.
  Green-bordered nodes were added in this iteration.
- **Middle-right (stacked bar)**: MDL budget across all iterations so far.
  Blue = L_model (structure + coefficients), orange = L_data (residual error).
- **Bottom-left (trace)**: Inner hill-climb search trace. Green dots = accepted
  proposals, red x's = rejected. Shows convergence behavior.
- **Bottom-right (text)**: Break detection status, Breaker hypothesis,
  collection request, search convergence info, and Builder/Breaker rationales.

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
