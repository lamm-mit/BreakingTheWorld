#!/usr/bin/env python3
"""
world_model_breaker_cli.py

Agentic world-model discovery over an open DAG space with MDL scoring.

What it does
------------
- Generates a mechanics-inspired oracle dataset with four staged regimes
  (elastic, post-yield, unloading, reloading).
- Maintains a persistent DAG world model: a linear combination of features,
  where each feature is a product of primitive factors over the observable
  variables. Any DAG over the factor alphabet is reachable.
- Two LLM agents (or deterministic heuristics in --no-llm mode) drive the loop:
    Breaker  proposes the next experiment/data slice to collect from observable
             metadata only, plus a falsification hypothesis for the current model.
    Builder  does not invent structure directly - it supervises an inner
             stochastic hill-climb that proposes DAG edits, refits by lstsq,
             and accepts moves that reduce total_bits = L_model + L_data.
- Complexity and fit are both expressed in bits, so the parsimony/fit tradeoff
  is principled (Minimum Description Length) rather than hand-weighted.

Quickstart
----------
    pip install openai numpy matplotlib networkx pandas pillow
    export OPENAI_API_KEY=...
    python src/world_model_breaker_cli.py --outdir demo_run

    python src/world_model_breaker_cli.py --outdir demo_offline --no-llm
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import textwrap
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.markers import MarkerStyle  # type: ignore[import]
import networkx as nx

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dag_model import DAGModel, FACTOR_TYPES, Factor, Feature, K_MAX, VarSpec
from dag_search import SearchTrace, hill_climb
from discovery_data import (
    DiscoveryDataset,
    Observation,
    observation_inputs,
    observation_target,
    read_dataset_json,
)
from oracle_adapters import DuffingOscillatorOracle, OracleAdapter
from tensile_test_oracle import synthetic_discovery_dataset

try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore


# -----------------------------
# Utility functions
# -----------------------------

def to_pretty_json(obj: Any) -> str:
    return json.dumps(obj, indent=2, sort_keys=True, default=_json_default)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Not JSON serialisable: {type(obj).__name__}")


def safe_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def safe_write_json(path: Path, obj: Any) -> None:
    safe_write_text(path, to_pretty_json(obj))


def try_make_gif(frame_paths: Sequence[Path], out_path: Path, duration_ms: int = 900) -> bool:
    try:
        from PIL import Image
    except Exception:
        return False
    images = [Image.open(p) for p in frame_paths if p.exists()]
    if not images:
        return False
    images[0].save(out_path, save_all=True, append_images=images[1:], duration=duration_ms, loop=0)
    return True


def progress_print(enabled: bool, message: str = "") -> None:
    if enabled:
        print(message, flush=True)


MDL_EXPLANATION = (
    "MDL means Minimum Description Length: prefer the model that gives the "
    "shortest total explanation of the revealed evidence. L_model is the bit "
    "cost of describing the DAG structure and fitted parameters; L_data is the "
    "bit cost of the remaining residual error. A new hypothesis is accepted "
    "only when L_total = L_model + L_data decreases, so extra complexity must "
    "pay for itself by explaining the data better."
)


def obs_to_X(obs: Sequence[Observation]) -> Dict[str, np.ndarray]:
    names: List[str] = []
    for o in obs:
        for name in observation_inputs(o):
            if name not in names:
                names.append(name)
    X: Dict[str, np.ndarray] = {}
    for name in names:
        vals = [observation_inputs(o).get(name, 0.0) for o in obs]
        X[name] = np.asarray(vals)
    return X


def var_space_for(
    obs: Sequence[Observation],
    observable_types: Optional[Dict[str, str]] = None,
) -> Tuple[VarSpec, ...]:
    observable_types = observable_types or {}
    X = obs_to_X(obs)
    specs: List[VarSpec] = []
    for name, values in X.items():
        requested = observable_types.get(name)
        if requested == "discrete":
            uniq = tuple(sorted({int(v) for v in values}))
            specs.append(VarSpec(name=name, kind="discrete", values=uniq or (0,)))
            continue
        if requested == "continuous":
            vals = values.astype(float)
            specs.append(VarSpec(name=name, kind="continuous", lo=float(np.min(vals)), hi=float(np.max(vals))))
            continue
        try:
            vals = values.astype(float)
            unique = sorted({float(v) for v in vals})
            if len(unique) <= 12 and all(abs(v - round(v)) < 1e-9 for v in unique):
                specs.append(VarSpec(name=name, kind="discrete", values=tuple(int(round(v)) for v in unique)))
            else:
                specs.append(VarSpec(name=name, kind="continuous", lo=float(np.min(vals)), hi=float(np.max(vals))))
        except (TypeError, ValueError):
            encoded = {v: i for i, v in enumerate(sorted({str(v) for v in values}))}
            specs.append(VarSpec(name=name, kind="discrete", values=tuple(encoded.values())))
    return tuple(specs)


def obs_to_XY(obs: Sequence[Observation]) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    return obs_to_X(obs), np.asarray([observation_target(o) for o in obs], dtype=float)


def initial_dag(var_space: Tuple[VarSpec, ...], target: str = "stress") -> DAGModel:
    """Minimal seed model: intercept only. Search will immediately add Ident(strain)."""
    return DAGModel(features=[], var_space=var_space, target=target)


# -----------------------------
# World model record (for artefacts)
# -----------------------------

@dataclass
class WorldModelRecord:
    version: int
    equation_lines: List[str]
    features: List[Dict[str, Any]]
    bits: Dict[str, float]
    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]
    revealed_slice: Optional[str]
    break_detected: bool
    break_type: str
    break_bits_gain: float
    breaker_hypothesis: str = ""
    collection_request: str = ""
    revealed_n_points: int = 0
    breaker_rationale: str = ""
    builder_rationale: str = ""
    builder_hypothesis: Dict[str, Any] = field(default_factory=dict)
    search_trace: List[Dict[str, Any]] = field(default_factory=list)
    search_summary: Dict[str, Any] = field(default_factory=dict)


def parse_proposed_feature(
    factors_json: Sequence[Dict[str, Any]],
    var_space: Sequence[VarSpec],
) -> Optional[Feature]:
    """Validate and construct a Feature from an LLM JSON proposal.

    Returns None if any factor is malformed (unknown kind, unknown variable,
    wrong var kind for the factor kind, or out-of-range parameter). This keeps
    Builder proposals hermetic: invalid ones are silently dropped rather than
    corrupting the hill-climb.
    """
    if not factors_json:
        return None
    if all(f.get("kind") == "Const" for f in factors_json):
        return None
    var_by_name = {v.name: v for v in var_space}
    parsed: List[Factor] = []
    for f in factors_json:
        kind = f.get("kind")
        if kind not in FACTOR_TYPES:
            return None
        if kind == "Const":
            parsed.append(Factor(kind="Const"))
            continue
        var = f.get("var")
        if not isinstance(var, str) or var not in var_by_name:
            return None
        v = var_by_name[var]
        if kind == "Ident":
            if v.kind != "continuous":
                return None
            parsed.append(Factor(kind="Ident", var=var))
        elif kind == "Pow":
            k = f.get("k")
            if not isinstance(k, int) or not (1 <= k <= K_MAX) or v.kind != "continuous":
                return None
            parsed.append(Factor(kind="Pow", var=var, k=int(k)))
        elif kind == "IndEq":
            a = f.get("a")
            if a is None or v.kind != "discrete" or int(a) not in v.values:
                return None
            parsed.append(Factor(kind="IndEq", var=var, a=int(a)))
        elif kind in ("IndLE", "ReLU"):
            t = f.get("threshold")
            if t is None or v.kind != "continuous":
                return None
            t = max(v.lo, min(v.hi, float(t)))
            parsed.append(Factor(kind=kind, var=var, threshold=t))
        else:
            return None
    return Feature.make(parsed)


def feature_dict(feat: Feature) -> Dict[str, Any]:
    return {
        "label": feat.label(),
        "factors": [
            {
                "kind": f.kind, "var": f.var, "k": f.k, "a": f.a,
                "threshold": (None if f.threshold is None else float(f.threshold)),
            }
            for f in feat.factors
        ],
    }


def record_from_model(
    version: int, model: DAGModel, bits: Dict[str, float],
    trace: SearchTrace, revealed_slice: Optional[str],
    break_detected: bool, break_type: str, break_bits_gain: float,
    breaker_rationale: str, builder_rationale: str,
    breaker_hypothesis: str = "", collection_request: str = "",
    revealed_n_points: int = 0,
    builder_hypothesis: Optional[Dict[str, Any]] = None,
) -> WorldModelRecord:
    nodes, edges = model.graph_spec()
    trace_dicts = [
        {
            "step": s.step, "operator": s.operator, "accepted": s.accepted,
            "bits_before": s.bits_before, "bits_after": s.bits_after,
            "l_model": s.l_model, "l_data": s.l_data,
            "k_features": s.k_features, "description": s.description,
        }
        for s in trace.steps
    ]
    return WorldModelRecord(
        version=version,
        equation_lines=model.equation_lines(),
        features=[feature_dict(f) for f in model.features],
        bits=bits,
        nodes=nodes, edges=edges,
        revealed_slice=revealed_slice,
        break_detected=break_detected,
        break_type=break_type,
        break_bits_gain=break_bits_gain,
        breaker_hypothesis=breaker_hypothesis,
        collection_request=collection_request,
        revealed_n_points=revealed_n_points,
        breaker_rationale=breaker_rationale,
        builder_rationale=builder_rationale,
        builder_hypothesis=builder_hypothesis or {},
        search_trace=trace_dicts,
        search_summary={
            "accepted_count": sum(1 for s in trace.steps if s.accepted),
            "rejected_count": sum(1 for s in trace.steps if not s.accepted),
            "total_steps": len(trace.steps),
            "converged": trace.converged,
            "stop_reason": trace.stop_reason,
            "final_bits": trace.best_bits_history[-1] if trace.best_bits_history else None,
            "starting_bits": trace.best_bits_history[0] if trace.best_bits_history else None,
        },
    )


# -----------------------------
# OpenAI Responses API agent
# -----------------------------

class StructuredAgent:
    def __init__(self, model: str, reasoning_effort: str = "medium",
                 temperature: Optional[float] = None, store: bool = False,
                 enabled: bool = True) -> None:
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.temperature = temperature
        self.store = store
        self.enabled = enabled
        self.client = None
        if enabled:
            if OpenAI is None:
                raise RuntimeError("openai package is not installed. Run: pip install --upgrade openai")
            self.client = OpenAI()

    def call_json(self, name: str, system_prompt: str,
                  user_payload: Dict[str, Any], schema: Dict[str, Any],
                  log_path: Optional[Path] = None) -> Dict[str, Any]:
        if not self.enabled:
            raise RuntimeError("StructuredAgent.call_json invoked while LLM use is disabled.")
        assert self.client is not None
        request = {
            "model": self.model,
            "store": self.store,
            "reasoning": {"effort": self.reasoning_effort},
            "input": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": to_pretty_json(user_payload)},
            ],
            "text": {"format": {"type": "json_schema", "name": name, "strict": True, "schema": schema}},
        }
        if self.temperature is not None:
            request["temperature"] = self.temperature
        raw = self.client.responses.create(**request)
        text = getattr(raw, "output_text", None)
        if not text:
            raise RuntimeError(f"Agent {name} returned no output_text.")
        parsed = json.loads(text)
        if log_path is not None:
            safe_write_json(log_path, {"request": request, "response_text": text, "parsed": parsed})
        return parsed


# -----------------------------
# Discovery runner
# -----------------------------

class DiscoveryRunner:
    def __init__(
        self,
        args: argparse.Namespace,
        dataset: Optional[DiscoveryDataset] = None,
        oracle: Optional[OracleAdapter] = None,
    ) -> None:
        self.args = args
        self.outdir = Path(args.outdir).resolve()
        self.outdir.mkdir(parents=True, exist_ok=True)
        (self.outdir / "agent_logs").mkdir(exist_ok=True)
        random.seed(args.seed)
        np.random.seed(args.seed)
        self.rng = random.Random(args.seed)
        self.oracle: Optional[OracleAdapter] = oracle
        if self.oracle is None and getattr(args, "oracle", "none") == "duffing":
            self.oracle = DuffingOscillatorOracle(seed=args.seed, noise_std=args.noise_std)

        if self.oracle is not None:
            self.dataset = self.oracle.dataset_shell()
        elif dataset is not None:
            self.dataset = dataset
        elif getattr(args, "dataset_json", None):
            self.dataset = read_dataset_json(Path(args.dataset_json))
        else:
            self.dataset = synthetic_discovery_dataset(
                seed=args.seed,
                noise_std=args.noise_std,
                example=getattr(args, "synthetic_example", "cyclic"),
            )
        self.obs = list(self.dataset.observations)
        self.stage_order = list(self.dataset.stage_order)
        self.stage_prerequisites = {
            sid: list(prereqs) for sid, prereqs in self.dataset.stage_prerequisites.items()
        }
        self.stage_labels = {o.stage_id: o.stage_label for o in self.obs}
        self.stage_to_obs: Dict[str, List[Observation]] = {sid: [] for sid in self.stage_order}
        self.stage_preview_obs: Dict[str, List[Observation]] = {sid: [] for sid in self.stage_order}

        if self.oracle is not None:
            idx = 0
            t = 0
            for sid in self.stage_order:
                self.stage_preview_obs[sid] = self.oracle.preview_stage(sid, idx_start=idx, t_start=t)
            for sid in self.dataset.initial_stage_ids:
                collected = self.oracle.collect_stage(sid, idx_start=idx, t_start=t)
                self.stage_to_obs[sid] = collected
                self.obs.extend(collected)
                self.stage_labels[sid] = collected[0].stage_label if collected else sid
                idx += len(collected)
                t += len(collected)
            self.next_idx = idx
            self.next_t = t
        else:
            for o in self.obs:
                if o.stage_id not in self.stage_to_obs:
                    self.stage_order.append(o.stage_id)
                    self.stage_to_obs[o.stage_id] = []
                    self.stage_preview_obs[o.stage_id] = []
                    self.stage_labels[o.stage_id] = o.stage_label
                self.stage_to_obs[o.stage_id].append(o)
            for sid in self.stage_order:
                self.stage_preview_obs[sid] = list(self.stage_to_obs.get(sid, []))
            self.next_idx = len(self.obs)
            self.next_t = len(self.obs)
        for sid in self.stage_order:
            if sid not in self.stage_labels:
                preview = self.stage_preview_obs.get(sid, [])
                self.stage_labels[sid] = preview[0].stage_label if preview else sid

        self.seen_stage_ids: List[str] = list(self.dataset.initial_stage_ids)
        self.remaining_stage_ids: List[str] = [s for s in self.stage_order if s not in self.seen_stage_ids]
        if self.oracle is not None:
            self.var_space = tuple(self.oracle.var_space())
        else:
            protocol_obs = [o for sid in self.stage_order for o in self.stage_preview_obs.get(sid, [])]
            self.var_space = var_space_for(protocol_obs or self.obs, self.dataset.observable_types)
        self.target_name = self.dataset.target_name or "stress"
        self.primary_x = self._primary_x_variable()

        llm_enabled = (not args.no_llm) and bool(os.environ.get("OPENAI_API_KEY"))
        self.agent = StructuredAgent(
            model=args.model, reasoning_effort=args.reasoning_effort,
            temperature=None, store=False, enabled=llm_enabled,
        ) if llm_enabled else None
        self.llm_enabled = llm_enabled

        self.history: List[WorldModelRecord] = []
        self.models: List[DAGModel] = []
        self.iteration_rows: List[Dict[str, Any]] = []
        self.frame_paths: List[Path] = []
        self.collection_history: List[Dict[str, Any]] = []
        self.dataset_analysis: Dict[str, Any] = {}
        self.verbose = bool(getattr(args, "verbose", True))

        safe_write_json(self.outdir / "config.json", {
            "args": vars(args),
            "llm_enabled": self.llm_enabled,
            "breaker_mode": getattr(args, "breaker_mode", "experimental"),
            "collection_policy": getattr(args, "collection_policy", "physical"),
            "stage_order": self.stage_order,
            "initial_stage_ids": self.seen_stage_ids,
            "stage_prerequisites": self.stage_prerequisites,
            "system_description": self.dataset.system_description,
            "observable_descriptions": self.dataset.observable_descriptions,
            "target_description": self.dataset.target_description,
            "target_name": self.target_name,
            "observable_types": self.dataset.observable_types,
            "protocol_descriptions": self.dataset.protocol_descriptions,
            "mdl_explanation": MDL_EXPLANATION,
            "data_source": "live_oracle" if self.oracle is not None else "dataset",
        })
        progress_print(self.verbose, f"Output directory: {self.outdir}")
        progress_print(self.verbose, f"LLM enabled: {self.llm_enabled}")
        progress_print(self.verbose, f"Breaker mode: {getattr(args, 'breaker_mode', 'experimental')}")
        progress_print(self.verbose, f"Collection policy: {getattr(args, 'collection_policy', 'physical')}")
        progress_print(self.verbose, f"Data source: {'live oracle' if self.oracle is not None else 'dataset'}")
        progress_print(self.verbose, f"Initial revealed slice: {', '.join(self.seen_stage_ids)}")
        progress_print(self.verbose, f"MDL: {MDL_EXPLANATION}")
        self.dataset_analysis = self.analyze_dataset_context()
        self.print_dataset_analysis()

    # --- data helpers -----
    def _primary_x_variable(self) -> str:
        preferred = ["strain", "x", "time"]
        continuous = [v.name for v in self.var_space if v.kind == "continuous"]
        for name in preferred:
            if name in continuous:
                return name
        return continuous[0] if continuous else self.var_space[0].name

    def seen_observations(self) -> List[Observation]:
        out: List[Observation] = []
        for sid in self.seen_stage_ids:
            out.extend(self.stage_to_obs.get(sid, []))
        return out

    def collectable_stage_ids(self) -> List[str]:
        if getattr(self.args, "collection_policy", "physical") == "unconstrained":
            return list(self.remaining_stage_ids)
        seen = set(self.seen_stage_ids)
        out: List[str] = []
        for sid in self.remaining_stage_ids:
            prereqs = self.stage_prerequisites.get(sid, [])
            if all(prereq in seen for prereq in prereqs):
                out.append(sid)
        return out

    def dataset_profile(self, include_hidden_targets: bool = False) -> Dict[str, Any]:
        profile: Dict[str, Any] = {
            "system_description": self.dataset.system_description,
            "observable_descriptions": self.dataset.observable_descriptions,
            "target_description": self.dataset.target_description,
            "stage_order": self.stage_order,
            "initial_stage_ids": self.seen_stage_ids,
            "stage_prerequisites": self.stage_prerequisites,
            "protocol_descriptions": self.dataset.protocol_descriptions,
            "stages": [],
        }
        for sid in self.stage_order:
            obs = self.stage_preview_obs.get(sid) or self.stage_to_obs.get(sid, [])
            Xs = obs_to_X(obs) if obs else {}
            var_kinds = {v.name: v.kind for v in self.var_space}
            observable_ranges: Dict[str, Any] = {}
            for name, values in Xs.items():
                if var_kinds.get(name) == "discrete":
                    observable_ranges[name] = sorted({int(v) for v in values})
                    continue
                try:
                    vals = values.astype(float)
                    observable_ranges[name] = [float(np.min(vals)), float(np.max(vals))]
                except (TypeError, ValueError):
                    observable_ranges[name] = sorted({str(v) for v in values})
            primary_vals = Xs.get(self.primary_x, np.asarray([], dtype=float))
            primary_range = (
                [float(np.min(primary_vals.astype(float))), float(np.max(primary_vals.astype(float)))]
                if len(primary_vals) else [None, None]
            )
            direction_values = sorted({int(v) for v in Xs.get("direction", [])}) if "direction" in Xs else []
            stage: Dict[str, Any] = {
                "stage_id": sid,
                "stage_label": self.stage_labels.get(sid, sid),
                "protocol_description": self.dataset.protocol_descriptions.get(sid, ""),
                "n_points": len(obs),
                "primary_observable": self.primary_x,
                "primary_observable_range": primary_range,
                "strain_range": primary_range,
                "observable_ranges": observable_ranges,
                "direction_values": direction_values,
                "prerequisites": self.stage_prerequisites.get(sid, []),
                "initially_revealed": sid in self.seen_stage_ids,
            }
            if include_hidden_targets and self.stage_to_obs.get(sid):
                stresses = np.asarray([observation_target(o) for o in self.stage_to_obs.get(sid, [])], dtype=float)
                stage["target_summary"] = {
                    "stress_range": [float(np.min(stresses)), float(np.max(stresses))],
                    "stress_mean": float(np.mean(stresses)),
                    "stress_std": float(np.std(stresses)),
                }
            profile["stages"].append(stage)
        return profile

    def analyze_dataset_context(self) -> Dict[str, Any]:
        profile = self.dataset_profile(include_hidden_targets=False)
        fallback = self.heuristic_dataset_analysis(profile)
        if not getattr(self.args, "llm_dataset_analysis", True):
            return fallback
        if not (self.llm_enabled and self.agent is not None):
            return fallback

        schema = {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "system_summary": {"type": "string"},
                "observables": {"type": "array", "items": {"type": "string"}},
                "protocol_logic": {"type": "array", "items": {"type": "string"}},
                "likely_world_model_challenges": {"type": "array", "items": {"type": "string"}},
                "discovery_expectations": {"type": "array", "items": {"type": "string"}},
                "caveats": {"type": "array", "items": {"type": "string"}},
            },
            "required": [
                "title", "system_summary", "observables", "protocol_logic",
                "likely_world_model_challenges", "discovery_expectations", "caveats",
            ],
            "additionalProperties": False,
        }
        system_prompt = textwrap.dedent("""
            You are a scientific run narrator. Analyze the dataset and experimental
            protocol before discovery begins. Explain what system is being studied,
            what the variables mean, what data can be collected, and what model
            failures are scientifically plausible. Do not inspect or infer hidden
            target measurements beyond the protocol metadata provided.
            Return only valid JSON matching the schema.
        """).strip()
        payload = {
            "task": "Produce a concise but technically detailed pre-run analysis for the user.",
            "dataset_profile_without_hidden_targets": profile,
            "constraints": [
                "Do not claim knowledge of unrevealed stress values.",
                "Use protocol descriptions and prerequisites to explain collection order.",
                "Focus on what a world-model discovery run is studying and what it may learn.",
            ],
        }
        try:
            return self.agent.call_json(
                name="dataset_context_analysis",
                system_prompt=system_prompt,
                user_payload=payload,
                schema=schema,
                log_path=self.outdir / "agent_logs" / "dataset_context_analysis.json",
            )
        except Exception:
            traceback.print_exc()
            return fallback

    def heuristic_dataset_analysis(self, profile: Dict[str, Any]) -> Dict[str, Any]:
        stage_lines = []
        for stage in profile["stages"]:
            prereqs = stage.get("prerequisites") or []
            prereq_text = f" after {', '.join(prereqs)}" if prereqs else " initially collectable"
            obs_ranges = stage.get("observable_ranges", {})
            range_text = ", ".join(
                f"{name} {rng[0]:.4g}-{rng[1]:.4g}" if isinstance(rng, list) and len(rng) == 2 else f"{name} {rng}"
                for name, rng in obs_ranges.items()
            )
            direction_text = (
                f", direction {stage['direction_values']}" if stage.get("direction_values") else ""
            )
            stage_lines.append(
                f"{stage['stage_label']}: {stage['n_points']} points, "
                f"{range_text}{direction_text}; {stage.get('protocol_description', '')} "
                f"Prerequisites: {prereq_text}."
            )
        observables = [
            f"{name}: {desc}" for name, desc in profile.get("observable_descriptions", {}).items()
        ]
        if profile.get("target_description"):
            observables.append(f"target: {profile['target_description']}")
        return {
            "title": "Pre-run dataset analysis",
            "system_summary": profile.get("system_description") or "Staged discovery dataset.",
            "observables": observables,
            "protocol_logic": stage_lines,
            "likely_world_model_challenges": [
                "A compact DAG must trade structural complexity against residual error using MDL bits.",
                (
                    "Stage transitions may require boundary, collective-mode, nonlinear, "
                    "or regime-specific DAG features."
                    if self.target_name == "bfactor_z"
                    else "Stage transitions may require new factors such as nonlinear strain terms, thresholds, or branch indicators."
                ),
                "Physical prerequisites constrain what can be collected next.",
            ],
            "discovery_expectations": [
                "The initial model is fit only on initially revealed data.",
                "Each iteration collects one protocol slice and rebuilds the model on all revealed observations.",
                "A successful rebuild lowers total description length relative to refitting the previous DAG.",
            ],
            "caveats": [
                "This pre-run analysis uses protocol metadata, not unrevealed target measurements.",
            ],
        }

    def print_dataset_analysis(self) -> None:
        if not self.verbose or not self.dataset_analysis:
            return
        progress_print(self.verbose)
        progress_print(self.verbose, f"Dataset analysis: {self.dataset_analysis.get('title', 'Pre-run dataset analysis')}")
        summary = self.dataset_analysis.get("system_summary", "")
        if summary:
            progress_print(self.verbose, f"  {summary}")
        for item in self.dataset_analysis.get("protocol_logic", [])[:6]:
            progress_print(self.verbose, f"  Protocol: {item}")
        for item in self.dataset_analysis.get("likely_world_model_challenges", [])[:4]:
            progress_print(self.verbose, f"  Challenge: {item}")

    # --- breaker agent / heuristic -----
    def experiment_candidates(self, current: DAGModel) -> List[Dict[str, Any]]:
        """Candidate experiments the Breaker may run, without looking at hidden y.

        The Breaker is allowed to know the experimental protocol it can request
        (strain range, loading direction, number of samples, and stage label).
        It is not allowed to inspect the unrevealed stress measurements. The
        score is therefore an experiment-design prior: novelty, extrapolation,
        and model sensitivity under the requested inputs.
        """
        seen_X = obs_to_X(self.seen_observations())
        x_seen = seen_X[self.primary_x].astype(float)
        x_min_seen, x_max_seen = float(np.min(x_seen)), float(np.max(x_seen))
        x_center_seen = float(np.mean(x_seen))
        x_span_seen = max(1e-9, x_max_seen - x_min_seen)
        seen_dirs = set(int(v) for v in seen_X.get("direction", np.asarray([], dtype=int)))
        seen_labels = {self.stage_labels[sid].lower() for sid in self.seen_stage_ids}
        summaries: List[Dict[str, Any]] = []
        for sid in self.collectable_stage_ids():
            obs = self.stage_preview_obs.get(sid) or self.stage_to_obs[sid]
            X = obs_to_X(obs)
            yhat = current.evaluate(X)
            primary_values = X[self.primary_x].astype(float)
            strain_min = float(np.min(primary_values))
            strain_max = float(np.max(primary_values))
            outside_support = float(np.mean((primary_values < x_min_seen) | (primary_values > x_max_seen)))
            slice_dirs = {int(d) for d in X.get("direction", np.asarray([], dtype=int))}
            direction_novelty = 1.0 if (slice_dirs - seen_dirs) else 0.0
            label = self.stage_labels[sid]
            label_lower = label.lower()
            protocol_novelty = 0.0 if label_lower in seen_labels else 1.0
            path_dependence_probe = 1.0 if any(term in label_lower for term in ("unload", "reload")) else 0.0
            high_strain_probe = float(max(0.0, strain_max - x_max_seen) / x_span_seen)
            center_shift = float(abs(0.5 * (strain_min + strain_max) - x_center_seen) / x_span_seen)
            pred_span = float(np.ptp(yhat))
            pred_scale = float(abs(np.mean(yhat)) + 1.0)
            model_sensitivity = min(2.0, pred_span / pred_scale)
            experiment_score = (
                1.10 * direction_novelty
                + 0.75 * outside_support
                + 0.45 * protocol_novelty
                + 0.35 * path_dependence_probe
                + 0.30 * high_strain_probe
                + 0.20 * center_shift
                + 0.15 * model_sensitivity
            )
            hypothesis = self.heuristic_hypothesis_for_candidate(
                sid=sid,
                label=label,
                strain_range=(strain_min, strain_max),
                direction_values=sorted(slice_dirs),
                direction_novelty=direction_novelty,
                outside_support=outside_support,
                current=current,
            )
            observable_ranges: Dict[str, Any] = {}
            var_kinds = {v.name: v.kind for v in self.var_space}
            for name, values in X.items():
                if var_kinds.get(name) == "discrete":
                    observable_ranges[name] = sorted({int(v) for v in values})
                    continue
                try:
                    vals = values.astype(float)
                    observable_ranges[name] = [float(np.min(vals)), float(np.max(vals))]
                except (TypeError, ValueError):
                    observable_ranges[name] = sorted({str(v) for v in values})
            summaries.append({
                "slice_id": sid,
                "slice_label": label,
                "protocol_description": self.dataset.protocol_descriptions.get(sid, ""),
                "prerequisites": self.stage_prerequisites.get(sid, []),
                "n_points": len(obs),
                "primary_observable": self.primary_x,
                "strain_range": [strain_min, strain_max],
                "primary_observable_range": [strain_min, strain_max],
                "observable_ranges": observable_ranges,
                "direction_values": sorted(slice_dirs),
                "predicted_stress_range": [float(np.min(yhat)), float(np.max(yhat))],
                "predicted_target_range": [float(np.min(yhat)), float(np.max(yhat))],
                "predicted_target_span": pred_span,
                "predicted_stress_span": pred_span,
                "outside_seen_support_fraction": outside_support,
                "direction_novelty": direction_novelty,
                "protocol_novelty": protocol_novelty,
                "path_dependence_probe": path_dependence_probe,
                "model_sensitivity": model_sensitivity,
                "experiment_score": experiment_score,
                "hypothesis": hypothesis,
            })
        summaries.sort(key=lambda s: s["experiment_score"], reverse=True)
        return summaries

    def slice_break_summaries(self, current: DAGModel) -> List[Dict[str, Any]]:
        """Compatibility wrapper.

        In the default experimental mode this returns no-hidden-label experiment
        candidates. Set --breaker-mode oracle to use the old hidden-y scoring.
        """
        if self.oracle is not None:
            return self.experiment_candidates(current)
        if getattr(self.args, "breaker_mode", "experimental") == "oracle":
            return self.oracle_slice_break_summaries(current)
        return self.experiment_candidates(current)

    def oracle_slice_break_summaries(self, current: DAGModel) -> List[Dict[str, Any]]:
        """Legacy Breaker candidate scoring that peeks at unrevealed y values."""
        seen_X = obs_to_X(self.seen_observations())
        x_seen = seen_X[self.primary_x].astype(float)
        x_min_seen, x_max_seen = float(np.min(x_seen)), float(np.max(x_seen))
        seen_dirs = set(int(v) for v in seen_X.get("direction", np.asarray([], dtype=int)))
        summaries: List[Dict[str, Any]] = []
        for sid in self.collectable_stage_ids():
            obs = self.stage_to_obs[sid]
            X, y = obs_to_XY(obs)
            yhat = current.evaluate(X)
            rmse = float(np.sqrt(np.mean((y - yhat) ** 2)))
            tol = 0.05 * max(1.0, float(np.ptp(y)))
            unexplained = float(np.mean(np.abs(y - yhat) > tol))
            primary_values = X[self.primary_x].astype(float)
            outside_support = float(np.mean((primary_values < x_min_seen) | (primary_values > x_max_seen)))
            slice_dirs = {int(d) for d in X.get("direction", np.asarray([], dtype=int))}
            direction_novelty = 1.0 if (slice_dirs - seen_dirs) else 0.0
            break_score = 0.55 * (rmse / (abs(float(np.mean(y))) + 1e-6)) \
                + 0.9 * unexplained + 0.35 * outside_support + 0.45 * direction_novelty
            summaries.append({
                "slice_id": sid,
                "slice_label": self.stage_labels[sid],
                "n_points": len(obs),
                "primary_observable": self.primary_x,
                "strain_range": [float(np.min(primary_values)), float(np.max(primary_values))],
                "primary_observable_range": [float(np.min(primary_values)), float(np.max(primary_values))],
                "direction_values": sorted(slice_dirs),
                "predicted_rmse": rmse,
                "predicted_unexplained_fraction": unexplained,
                "outside_seen_support_fraction": outside_support,
                "direction_novelty": direction_novelty,
                "break_score": break_score,
                "experiment_score": break_score,
                "hypothesis": "Oracle scoring: hidden labels are used to estimate model failure directly.",
            })
        summaries.sort(key=lambda s: s["break_score"], reverse=True)
        return summaries

    def heuristic_hypothesis_for_candidate(
        self, sid: str, label: str, strain_range: Tuple[float, float],
        direction_values: Sequence[int], direction_novelty: float,
        outside_support: float, current: DAGModel,
    ) -> str:
        label_lower = label.lower()
        if self.target_name == "dvdt" or "oscillator" in self.dataset.system_description.lower():
            if "large" in label_lower:
                return (
                    "A larger-amplitude trajectory may expose nonlinear restoring forces: "
                    "the current acceleration law may need higher powers of displacement."
                )
            if "velocity" in label_lower:
                return (
                    "A high-velocity trajectory may expose damping: the acceleration law "
                    "may need velocity-dependent terms."
                )
            if "forced" in label_lower:
                return (
                    "A driven trajectory may expose external forcing: the acceleration law "
                    "may need a force input term or force-state interaction."
                )
            return "A new trajectory may reveal missing terms in the governing acceleration law."
        if self.target_name == "bfactor_z":
            if "termin" in label_lower or "flex" in label_lower:
                return (
                    "Proteins with exposed or flexible termini may break a pure GNM-fluctuation "
                    "model, requiring explicit boundary or disorder-proxy features."
                )
            if "hinge" in label_lower or "domain" in label_lower:
                return (
                    "Hinge/domain-motion proteins may expose collective-mode structure: the "
                    "world model may need slow-mode participation or hinge-shape features."
                )
            if "validation" in label_lower or "mixed" in label_lower:
                return (
                    "A mixed protein stage tests whether the revised flexibility ontology "
                    "generalizes beyond the regimes that forced earlier revisions."
                )
            if outside_support:
                return (
                    "This protein stage extends the observed feature support and may reveal "
                    "where the current residue-flexibility DAG extrapolates poorly."
                )
            if not current.features:
                return "The intercept-only model should fail on structured residue B-factor data."
            return "This protein stage probes a distinct flexibility regime and may reveal missing structure."
        if "unload" in label_lower:
            return (
                "Reverse loading may expose path dependence: the current model may need "
                "a direction-specific offset or slope rather than one stress-strain curve."
            )
        if "reload" in label_lower:
            return (
                "Reloading may expose hysteresis: stress at the same strain may differ "
                "from first loading, requiring a regime split or direction/strain interaction."
            )
        if "post" in label_lower or strain_range[1] > 0.02:
            return (
                "Higher-strain loading may expose yield or hardening: the current model "
                "may need a thresholded or nonlinear strain feature."
            )
        if direction_novelty:
            return "A new direction value may require a new observed-variable branch in the DAG."
        if outside_support:
            return "Inputs outside the observed strain support may reveal extrapolation failure."
        if not current.features:
            return "The intercept-only model should fail on any structured stress-strain experiment."
        return "This protocol probes a distinct part of input space and may reveal missing structure."

    def heuristic_breaker_choice(self, summaries: List[Dict[str, Any]]) -> Dict[str, Any]:
        choice = summaries[0]
        score = choice.get("experiment_score", choice.get("break_score", 0.0))
        def format_range(name: str, rng: Any) -> str:
            if isinstance(rng, list) and len(rng) == 2 and all(isinstance(v, (int, float)) for v in rng):
                return f"{name}={rng[0]:.4g}..{rng[1]:.4g}"
            return f"{name}={rng}"
        observable_text = ", ".join(
            format_range(name, rng)
            for name, rng in choice.get("observable_ranges", {}).items()
        )
        direction_text = (
            f" and direction values {choice['direction_values']}"
            if choice.get("direction_values") and "direction" not in choice.get("observable_ranges", {}) else ""
        )
        return {
            "selected_slice": choice["slice_id"],
            "hypothesis": choice.get("hypothesis", "The selected experiment may falsify the current model."),
            "collection_request": (
                f"Collect {choice['n_points']} observations for {choice['slice_label']} "
                f"over {observable_text or (self.primary_x + ' range')} "
                f"{direction_text}."
            ),
            "expected_failure_mode": choice.get("hypothesis", "largest experimental break pressure"),
            "rationale": (
                f"Heuristic pick: `{choice['slice_id']}` has the highest experiment score "
                f"({score:.3f}) among remaining collectable protocols. The score uses only "
                "observable protocol metadata and current-model predictions, not hidden "
                f"{self.target_name} measurements."
            ),
        }

    def breaker_choice(self, current: DAGModel, summaries: List[Dict[str, Any]], iteration: int) -> Dict[str, Any]:
        if self.llm_enabled and self.agent is not None:
            schema = {
                "type": "object",
                "properties": {
                    "selected_slice": {"type": "string", "enum": [s["slice_id"] for s in summaries]},
                    "hypothesis": {"type": "string"},
                    "collection_request": {"type": "string"},
                    "expected_failure_mode": {"type": "string"},
                    "rationale": {"type": "string"},
                },
                "required": ["selected_slice", "hypothesis", "collection_request", "expected_failure_mode", "rationale"],
                "additionalProperties": False,
            }
            payload = {
                "task": (
                    "Choose the next experiment/data slice to collect in order to falsify "
                    "or stress-test the current world model. The candidate list contains "
                    "only observable protocol metadata and current-model predictions; it "
                    "does not contain unrevealed stress measurements."
                ),
                "current_world_model": {
                    "equations": current.equation_lines(),
                    "features": [feature_dict(f) for f in current.features],
                    "bits": current.total_bits(*obs_to_XY(self.seen_observations())),
                },
                "system_context": {
                    "system_description": self.dataset.system_description,
                    "observable_descriptions": self.dataset.observable_descriptions,
                    "target_description": self.dataset.target_description,
                    "stage_prerequisites": self.stage_prerequisites,
                    "collection_policy": getattr(self.args, "collection_policy", "physical"),
                },
                "candidate_experiments": summaries,
                "decision_principles": [
                    "State a concrete hypothesis about how the current model might fail.",
                    "Prefer a minimally sufficient experiment that tests hidden regimes, extrapolation, or path dependence.",
                    "Do not claim to know unrevealed residuals or measured stresses before collection.",
                    "Respect the system context and physical collection policy; do not propose a protocol whose prerequisites are not met.",
                    "The selected slice will be collected from the oracle only after your choice.",
                ],
            }
            system_prompt = textwrap.dedent("""
                You are the Breaker agent in a scientific world-model discovery loop.
                Design the next data-collection experiment most likely to falsify the current
                DAG world model. You may use the available experiment protocols and the current
                model predictions, but you do not know the unrevealed measured stresses.
                Return only valid JSON matching the schema.
            """).strip()
            try:
                return self.agent.call_json(
                    name=f"breaker_choice_iter_{iteration:02d}",
                    system_prompt=system_prompt, user_payload=payload, schema=schema,
                    log_path=self.outdir / "agent_logs" / f"breaker_choice_iter_{iteration:02d}.json",
                )
            except Exception:
                traceback.print_exc()
        return self.heuristic_breaker_choice(summaries)

    # --- builder agent / search -----
    def heuristic_builder_hypothesis(
        self, current: DAGModel, iteration: int,
    ) -> Dict[str, Any]:
        X, y = obs_to_XY(self.seen_observations())
        yhat = current.evaluate(X)
        residual = y - yhat
        stage_rmse = []
        for sid in self.seen_stage_ids:
            obs_slice = self.stage_to_obs.get(sid, [])
            if not obs_slice:
                continue
            Xs, ys = obs_to_XY(obs_slice)
            rs = ys - current.evaluate(Xs)
            stage_rmse.append((sid, self.stage_labels[sid], float(np.sqrt(np.mean(rs ** 2)))))
        stage_rmse.sort(key=lambda item: item[2], reverse=True)
        worst = stage_rmse[0] if stage_rmse else ("", "", float("nan"))
        return {
            "mode": "heuristic",
            "diagnosis": (
                f"Revealed data contain {len(y)} observations. Current-model RMSE is "
                f"{float(np.sqrt(np.mean(residual ** 2))):.4f}; the largest slice RMSE "
                f"is on `{worst[0]}` ({worst[1]})."
            ),
            "physical_interpretation": (
                "Offline mode does not use external world knowledge. The search will test "
                "compact symbolic DAG edits such as nonlinear observable terms, thresholds, "
                "discrete branches, and interactions."
            ),
            "candidate_latent_variables": [
                "unobserved regime/state variable if residuals cluster by protocol slice",
                "path-history variable if loading and unloading/reloading disagree",
            ],
            "proposed_features": [],
            "falsifiable_predictions": [
                "If a proposed structure is useful, it must reduce L_total after refitting.",
                "If residuals remain clustered by slice, the observable state is likely missing a protocol or path-history variable.",
            ],
            "mdl_acceptance_rule": MDL_EXPLANATION,
        }

    def heuristic_seed_features(self) -> List[Feature]:
        """Deterministic first-order seeds for no-LLM runs.

        The stochastic DAG search can discover these unaided, but real datasets
        with many observables make the initial search unnecessarily fragile.
        These seeds are only proposals; MDL still rejects any feature that does
        not earn its complexity cost after refitting.
        """
        by_name = {v.name: v for v in self.var_space}
        seeds: List[Feature] = []

        def add_ident(name: str) -> None:
            spec = by_name.get(name)
            if spec is not None and spec.kind == "continuous":
                seeds.append(Feature.make([Factor(kind="Ident", var=name)]))

        def add_pow(name: str, k: int) -> None:
            spec = by_name.get(name)
            if spec is not None and spec.kind == "continuous":
                seeds.append(Feature.make([Factor(kind="Pow", var=name, k=k)]))

        def add_indicator(name: str, value: int) -> None:
            spec = by_name.get(name)
            if spec is not None and spec.kind == "discrete" and value in spec.values:
                seeds.append(Feature.make([Factor(kind="IndEq", var=name, a=value)]))

        target = self.target_name.lower()
        if target == "bfactor_z":
            for name in [
                "gnm_fluct_z",
                "gnm_fluct_log_z",
                "contact_degree_z",
                "terminal_exposure",
                "mode1_abs_z",
                "hinge_score_z",
                "chain_break_proximity",
                "res_index_norm",
            ]:
                add_ident(name)
            for name in ["is_terminal", "is_gly", "is_pro", "is_hydrophobic", "is_charged", "is_polar"]:
                add_indicator(name, 1)
        elif target == "dvdt":
            for name in ["x", "v", "force", "time"]:
                add_ident(name)
            add_pow("x", 3)
            add_pow("v", 3)
        else:
            for name in ["strain", "x", "time"]:
                add_ident(name)

        unique: Dict[Tuple, Feature] = {}
        for feat in seeds:
            unique[feat.canonical_key()] = feat
        return list(unique.values())

    def builder_hypothesis_and_feature_proposals(
        self, current: DAGModel, iteration: int,
    ) -> Tuple[List[Feature], Dict[str, Any]]:
        """Ask the Builder LLM for a physics hypothesis plus candidate features.

        Returns candidate Features plus a richer reasoning record. The hill-climb
        front-loads proposed features as add_feature trials; only those that
        strictly reduce L_total after lstsq refit are accepted, so MDL remains
        the quantitative judge.
        """
        fallback = self.heuristic_builder_hypothesis(current, iteration)
        if not getattr(self.args, "llm_builder", False):
            return self.heuristic_seed_features(), fallback
        if not (self.llm_enabled and self.agent is not None):
            return self.heuristic_seed_features(), fallback

        X, y = obs_to_XY(self.seen_observations())
        bits_report = current.total_bits(X, y)

        slice_summaries: List[Dict[str, Any]] = []
        for sid in self.seen_stage_ids:
            obs_slice = self.stage_to_obs.get(sid, [])
            if not obs_slice:
                continue
            Xs, ys = obs_to_XY(obs_slice)
            yhat_s = current.evaluate(Xs)
            r = ys - yhat_s
            primary_values = Xs[self.primary_x].astype(float)
            slice_summaries.append({
                "slice_id": sid,
                "slice_label": self.stage_labels[sid],
                "n_points": len(obs_slice),
                "primary_observable": self.primary_x,
                "strain_range": [float(np.min(primary_values)), float(np.max(primary_values))],
                "primary_observable_range": [float(np.min(primary_values)), float(np.max(primary_values))],
                "direction_values": sorted({int(d) for d in Xs.get("direction", [])}),
                "rmse": float(np.sqrt(np.mean(r ** 2))),
                "mean_resid": float(np.mean(r)),
                "std_resid": float(np.std(r)),
                "max_abs_resid": float(np.max(np.abs(r))),
            })

        var_space_desc = [
            {
                "name": v.name,
                "kind": v.kind,
                "values": list(v.values) if v.kind == "discrete" else [],
                "lo": float(v.lo) if v.kind == "continuous" else 0.0,
                "hi": float(v.hi) if v.kind == "continuous" else 0.0,
            }
            for v in self.var_space
        ]

        factor_alphabet = [
            "Const -> 1 (no variable, no parameter)",
            "Ident(var): var (continuous var only)",
            f"Pow(var, k): var**k, k integer in [1, {K_MAX}] (continuous var only)",
            "IndEq(var, a): 1 if var==a else 0 (discrete var only; a must be in var.values)",
            "IndLE(var, threshold): 1 if var<=threshold else 0 (continuous var only)",
            "ReLU(var, threshold): max(var - threshold, 0) (continuous var only)",
        ]

        payload = {
            "task": (
                "Diagnose the current world-model failure after the newly revealed evidence, "
                "reason about the underlying system using the provided experiment context and "
                "your general scientific knowledge, then propose candidate symbolic features "
                "to add to the DAG world model. Each feature is a product of 1-4 primitive "
                "factors from the alphabet. Features will be tried in order; each is accepted "
                "only if it strictly reduces total MDL bits (L_model + L_data) after lstsq refit."
            ),
            "current_world_model": {
                "equations": current.equation_lines(),
                "existing_features": [feature_dict(f) for f in current.features],
                "bits": bits_report,
            },
            "system_context": {
                "system_description": self.dataset.system_description,
                "observable_descriptions": self.dataset.observable_descriptions,
                "target_description": self.dataset.target_description,
                "protocol_descriptions": {
                    sid: self.dataset.protocol_descriptions.get(sid, "")
                    for sid in self.seen_stage_ids
                },
            },
            "residuals_by_revealed_slice": slice_summaries,
            "var_space": var_space_desc,
            "factor_alphabet": factor_alphabet,
            "mdl_acceptance_rule": MDL_EXPLANATION,
            "guidance": [
                "You may use the system and protocol descriptions to bring in domain knowledge.",
                "Do not claim access to unrevealed future measurements.",
                "Explain the physical idea first, then encode only what the current factor alphabet can represent.",
                "Target slices with the largest RMSE or systematic mean_resid != 0.",
                "If a latent/internal variable would help but is not observable yet, name it explicitly.",
                "Prefer structurally distinct proposals (not near-duplicates of existing features).",
                "Use thresholds inside the variable's [lo, hi] range.",
                "Return 4-6 candidate features, ordered best-first.",
            ],
        }

        factor_schema = {
            "type": "object",
            "properties": {
                "kind": {"type": "string", "enum": list(FACTOR_TYPES)},
                "var": {"type": ["string", "null"]},
                "k": {"type": ["integer", "null"]},
                "a": {"type": ["integer", "null"]},
                "threshold": {"type": ["number", "null"]},
            },
            "required": ["kind", "var", "k", "a", "threshold"],
            "additionalProperties": False,
        }
        schema = {
            "type": "object",
            "properties": {
                "diagnosis": {"type": "string"},
                "physical_interpretation": {"type": "string"},
                "candidate_latent_variables": {"type": "array", "items": {"type": "string"}},
                "proposed_features": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "factors": {"type": "array", "items": factor_schema},
                            "rationale": {"type": "string"},
                        },
                        "required": ["factors", "rationale"],
                        "additionalProperties": False,
                    },
                },
                "falsifiable_predictions": {"type": "array", "items": {"type": "string"}},
                "overall_rationale": {"type": "string"},
            },
            "required": [
                "diagnosis", "physical_interpretation", "candidate_latent_variables",
                "proposed_features", "falsifiable_predictions", "overall_rationale",
            ],
            "additionalProperties": False,
        }

        system_prompt = textwrap.dedent("""
            You are the Builder agent in a scientific world-model discovery loop. Given the
            current DAG model, revealed evidence, residuals, and experiment context, reason
            about what physical mechanism may be missing from the world model. You may use
            your general scientific knowledge of the described system, but you must not claim
            access to unrevealed measurements. Translate the reasoning into candidate DAG
            features that can be tested by MDL. Return only valid JSON matching the schema.
        """).strip()

        try:
            response = self.agent.call_json(
                name=f"builder_proposals_iter_{iteration:02d}",
                system_prompt=system_prompt,
                user_payload=payload,
                schema=schema,
                log_path=self.outdir / "agent_logs" / f"builder_proposals_iter_{iteration:02d}.json",
            )
        except Exception:
            traceback.print_exc()
            return [], fallback

        features: List[Feature] = []
        proposed_feature_records: List[Dict[str, Any]] = []
        for item in response.get("proposed_features", []):
            feat = parse_proposed_feature(item.get("factors", []), self.var_space)
            if feat is not None:
                features.append(feat)
                proposed_feature_records.append({
                    "label": feat.label(),
                    "rationale": item.get("rationale", ""),
                    "factors": feature_dict(feat)["factors"],
                })
        hypothesis = {
            "mode": "llm",
            "diagnosis": response.get("diagnosis", ""),
            "physical_interpretation": response.get("physical_interpretation", ""),
            "candidate_latent_variables": response.get("candidate_latent_variables", []),
            "proposed_features": proposed_feature_records,
            "falsifiable_predictions": response.get("falsifiable_predictions", []),
            "overall_rationale": response.get("overall_rationale", ""),
            "mdl_acceptance_rule": MDL_EXPLANATION,
        }
        return features, hypothesis

    def run_inner_search(self, start: DAGModel, iteration: int, label: str) -> Tuple[DAGModel, SearchTrace, Dict[str, Any]]:
        X, y = obs_to_XY(self.seen_observations())
        seed_features, builder_hypothesis = self.builder_hypothesis_and_feature_proposals(start, iteration)
        diagnosis = builder_hypothesis.get("diagnosis", "")
        if diagnosis:
            progress_print(self.verbose, f"  Builder diagnosis: {diagnosis}")
        interpretation = builder_hypothesis.get("physical_interpretation", "")
        if interpretation:
            progress_print(self.verbose, f"  Builder physical interpretation: {interpretation}")
        latent = builder_hypothesis.get("candidate_latent_variables", [])
        if latent:
            progress_print(self.verbose, f"  Builder candidate latent variables: {', '.join(latent[:4])}")
        if seed_features:
            progress_print(self.verbose, f"  Builder LLM seeds: {', '.join(f.label() for f in seed_features)}")
        starts: List[DAGModel] = [start.clone()]
        for _ in range(max(0, self.args.search_restarts - 1)):
            starts.append(initial_dag(self.var_space, self.target_name))

        best_model: Optional[DAGModel] = None
        best_bits = float("inf")
        best_trace: Optional[SearchTrace] = None
        for i, seed_model in enumerate(starts):
            child_rng = random.Random(self.rng.randrange(2**30))
            progress_print(self.verbose, f"  Search start {i + 1}/{len(starts)}...")
            model, trace = hill_climb(
                seed_model, X, y, child_rng,
                max_steps=self.args.search_steps,
                patience=self.args.search_patience,
                seed_features=seed_features,
            )
            final_bits = trace.best_bits_history[-1] if trace.best_bits_history else float("inf")
            if final_bits < best_bits:
                best_bits = final_bits
                best_model = model
                best_trace = trace
                progress_print(
                    self.verbose,
                    f"    new best: L_total={best_bits:.2f} bits, "
                    f"accepted={sum(1 for s in trace.steps if s.accepted)}/{len(trace.steps)}",
                )
            safe_write_json(
                self.outdir / "agent_logs" / f"search_trace_iter_{iteration:02d}_{label}_start{i}.json",
                {
                    "steps": [
                        {
                            "step": s.step, "operator": s.operator, "accepted": s.accepted,
                            "bits_before": s.bits_before, "bits_after": s.bits_after,
                            "l_model": s.l_model, "l_data": s.l_data,
                            "description": s.description,
                        } for s in trace.steps
                    ],
                    "converged": trace.converged,
                    "stop_reason": trace.stop_reason,
                    "final_bits": final_bits,
                },
            )
        assert best_model is not None and best_trace is not None
        return best_model, best_trace, builder_hypothesis

    def classify_break(self, prev_model: DAGModel, new_model: DAGModel) -> str:
        prev_keys = set(prev_model.canonical_key())
        new_keys = set(new_model.canonical_key())
        added = new_keys - prev_keys
        removed = prev_keys - new_keys
        if not added and not removed:
            return "parameter_update"

        def touches_direction(feats: set) -> bool:
            for feat_key in feats:
                for item in feat_key:
                    if item[1] == "direction":
                        return True
            return False

        def has_threshold(feats: set) -> bool:
            for feat_key in feats:
                for item in feat_key:
                    if item[0] in ("IndLE", "ReLU"):
                        return True
            return False

        if touches_direction(added) and not touches_direction(prev_keys):
            return "new_observed_variable"
        if has_threshold(added) and not has_threshold(prev_keys):
            return "regime_split"
        return "ontology_break"

    # --- loop entry points -----
    def create_initial_world_model(self) -> DAGModel:
        progress_print(self.verbose)
        progress_print(self.verbose, "Iteration 0: build initial world model")
        progress_print(self.verbose, f"  Revealed data: {self.stage_labels[self.seen_stage_ids[0]]}")
        start = initial_dag(self.var_space, self.target_name)
        model, trace, builder_hypothesis = self.run_inner_search(start, iteration=0, label="initial")
        X, y = obs_to_XY(self.seen_observations())
        bits = model.total_bits(X, y)
        record = record_from_model(
            version=0, model=model, bits=bits, trace=trace,
            revealed_slice=None, break_detected=False, break_type="none",
            break_bits_gain=0.0,
            breaker_rationale="",
            builder_rationale=(
                f"Inner search starting from intercept-only; converged at {bits['L_total']:.1f} bits "
                f"({trace.stop_reason})."
            ),
            builder_hypothesis=builder_hypothesis,
        )
        self.history.append(record)
        self.models.append(model)
        self.write_world_model_json(record)
        self.render_iteration_frame(model, record, newly_revealed_slice=None)
        self.iteration_rows.append(self._row_from_record(record))
        self.print_record_progress(record)
        return model

    def collect_experiment(self, break_choice: Dict[str, Any], summaries: Sequence[Dict[str, Any]],
                           iteration: int) -> Tuple[str, List[Observation]]:
        """Reveal the Breaker-selected experiment from the oracle.

        This is the only point where hidden measurements become visible. Before
        this call the Breaker sees protocols and model predictions, not stress y.
        """
        selected = break_choice["selected_slice"]
        collectable = self.collectable_stage_ids()
        if selected not in collectable:
            selected = summaries[0]["slice_id"]
        self.remaining_stage_ids.remove(selected)
        self.seen_stage_ids.append(selected)
        if self.oracle is not None:
            collected = self.oracle.collect_stage(selected, idx_start=self.next_idx, t_start=self.next_t)
            self.next_idx += len(collected)
            self.next_t += len(collected)
            self.stage_to_obs[selected] = collected
            self.obs.extend(collected)
            if collected:
                self.stage_labels[selected] = collected[0].stage_label
        else:
            collected = list(self.stage_to_obs[selected])
        summary = next((s for s in summaries if s["slice_id"] == selected), {})
        event = {
            "iteration": iteration,
            "selected_slice": selected,
            "slice_label": self.stage_labels[selected],
            "n_points": len(collected),
            "strain_range": summary.get("strain_range"),
            "direction_values": summary.get("direction_values"),
            "hypothesis": break_choice.get("hypothesis", ""),
            "collection_request": break_choice.get("collection_request", ""),
            "expected_failure_mode": break_choice.get("expected_failure_mode", ""),
            "rationale": break_choice.get("rationale", ""),
        }
        self.collection_history.append(event)
        safe_write_json(self.outdir / "agent_logs" / f"collection_iter_{iteration:02d}.json", event)
        progress_print(self.verbose, f"  Breaker collected: {selected} ({self.stage_labels[selected]})")
        progress_print(self.verbose, f"  Hypothesis: {event['hypothesis'] or 'n/a'}")
        progress_print(self.verbose, f"  Collection request: {event['collection_request'] or 'n/a'}")
        return selected, collected

    def run_discovery_step(self, current: DAGModel, iteration: int) -> Tuple[DAGModel, WorldModelRecord, List[Dict[str, Any]], Dict[str, Any]]:
        """Run one Breaker collection plus Builder rebuild step."""
        progress_print(self.verbose)
        progress_print(self.verbose, f"Iteration {iteration}: Breaker designs next collection")
        summaries = self.slice_break_summaries(current)
        if not summaries:
            raise RuntimeError("No remaining experiments to collect.")
        top = summaries[0]
        score = top.get("experiment_score", top.get("break_score", 0.0))
        progress_print(
            self.verbose,
            f"  Top candidate: {top['slice_id']} ({top['slice_label']}), score={score:.3f}",
        )
        break_choice = self.breaker_choice(current, summaries, iteration)
        selected, collected = self.collect_experiment(break_choice, summaries, iteration)

        X, y = obs_to_XY(self.seen_observations())
        prev_model = current
        prev_refit = prev_model.clone()
        prev_refit.fit(X, y)
        prev_bits = prev_refit.total_bits(X, y)

        new_model, trace, builder_hypothesis = self.run_inner_search(current, iteration=iteration, label="rebuild")
        new_bits = new_model.total_bits(X, y)

        break_bits_gain = prev_bits["L_total"] - new_bits["L_total"]
        break_type = self.classify_break(prev_model, new_model)
        break_detected = (break_type not in ("parameter_update", "none")) and break_bits_gain > 0.5

        record = record_from_model(
            version=iteration, model=new_model, bits=new_bits, trace=trace,
            revealed_slice=selected, break_detected=break_detected,
            break_type=break_type, break_bits_gain=break_bits_gain,
            breaker_rationale=break_choice.get("rationale", ""),
            builder_rationale=(
                f"Inner hill-climb accepted {sum(1 for s in trace.steps if s.accepted)} "
                f"proposals over {len(trace.steps)} steps; {trace.stop_reason}. "
                f"Total bits: {prev_bits['L_total']:.1f} -> {new_bits['L_total']:.1f} "
                f"(gain {break_bits_gain:+.1f})."
            ),
            breaker_hypothesis=break_choice.get("hypothesis", ""),
            collection_request=break_choice.get("collection_request", ""),
            revealed_n_points=len(collected),
            builder_hypothesis=builder_hypothesis,
        )
        self.history.append(record)
        self.models.append(new_model)
        self.write_world_model_json(record)
        self.render_iteration_frame(new_model, record, newly_revealed_slice=selected)
        self.iteration_rows.append(self._row_from_record(record))
        self.print_record_progress(record)
        return new_model, record, summaries, break_choice

    def finalize_outputs(self) -> None:
        progress_print(self.verbose)
        progress_print(self.verbose, "Finalizing outputs...")
        self.write_metrics_csv()
        self.write_report()
        self.write_summary_json()
        self.write_paper_figures()
        try_make_gif(self.frame_paths, self.outdir / "evolution.gif")
        progress_print(self.verbose, f"  Wrote metrics: {self.outdir / 'metrics.csv'}")
        progress_print(self.verbose, f"  Wrote report: {self.outdir / 'report.md'}")
        progress_print(self.verbose, f"  Wrote summary: {self.outdir / 'run_summary.json'}")
        if getattr(self.args, "paper_figures", True):
            progress_print(self.verbose, f"  Wrote SVG figures: {self.outdir / 'paper_figures'}")

    def print_record_progress(self, record: WorldModelRecord) -> None:
        accepted = record.search_summary.get("accepted_count", 0)
        total_steps = record.search_summary.get("total_steps", 0)
        progress_print(
            self.verbose,
            f"  Builder search: accepted={accepted}/{total_steps}, "
            f"converged={record.search_summary.get('converged')}",
        )
        progress_print(
            self.verbose,
            f"  MDL: L_model={record.bits['L_model']:.1f}, "
            f"L_data={record.bits['L_data']:.1f}, "
            f"L_total={record.bits['L_total']:.1f}, "
            f"gain={record.break_bits_gain:+.1f}",
        )
        progress_print(
            self.verbose,
            f"  Fit: RMSE={record.bits['rmse']:.4f}, R2={record.bits['r2']:.4f}, "
            f"features={record.bits['k_features']}",
        )
        progress_print(self.verbose, f"  Equation: {record.equation_lines[0]}")

    def run(self) -> None:
        current = self.create_initial_world_model()
        max_rounds = min(self.args.rounds, len(self.remaining_stage_ids))
        for iteration in range(1, max_rounds + 1):
            if not self.remaining_stage_ids or not self.collectable_stage_ids():
                break
            current, _, _, _ = self.run_discovery_step(current, iteration)

        self.finalize_outputs()

    # --- artefacts -----
    def _row_from_record(self, r: WorldModelRecord) -> Dict[str, Any]:
        return {
            "iteration": r.version,
            "revealed_slice": r.revealed_slice if r.revealed_slice else "initial",
            "k_features": r.bits.get("k_features"),
            "L_model_bits": round(r.bits["L_model"], 3),
            "L_data_bits": round(r.bits["L_data"], 3),
            "L_total_bits": round(r.bits["L_total"], 3),
            "rmse": round(r.bits["rmse"], 4),
            "r2": round(r.bits["r2"], 4),
            "break_detected": r.break_detected,
            "break_type": r.break_type,
            "break_bits_gain": round(r.break_bits_gain, 3),
            "revealed_n_points": r.revealed_n_points,
            "search_accepted": r.search_summary["accepted_count"],
            "search_steps": r.search_summary["total_steps"],
            "search_converged": r.search_summary["converged"],
        }

    def write_world_model_json(self, record: WorldModelRecord) -> None:
        safe_write_json(self.outdir / f"world_model_iter_{record.version:02d}.json", asdict(record))

    def write_metrics_csv(self) -> None:
        path = self.outdir / "metrics.csv"
        if not self.iteration_rows:
            return
        fieldnames = list(self.iteration_rows[0].keys())
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in self.iteration_rows:
                writer.writerow(row)

    def write_summary_json(self) -> None:
        safe_write_json(self.outdir / "run_summary.json", {
            "llm_enabled": self.llm_enabled,
            "breaker_mode": getattr(self.args, "breaker_mode", "experimental"),
            "system_description": self.dataset.system_description,
            "observable_descriptions": self.dataset.observable_descriptions,
            "target_description": self.dataset.target_description,
            "target_name": self.target_name,
            "observable_types": self.dataset.observable_types,
            "protocol_descriptions": self.dataset.protocol_descriptions,
            "stage_prerequisites": self.stage_prerequisites,
            "mdl_explanation": MDL_EXPLANATION,
            "dataset_analysis": self.dataset_analysis,
            "history": [asdict(r) for r in self.history],
            "iterations": self.iteration_rows,
            "collection_history": self.collection_history,
            "seen_stage_ids": self.seen_stage_ids,
            "remaining_stage_ids": self.remaining_stage_ids,
        })

    def protein_model_interpretation(self, record: WorldModelRecord) -> List[str]:
        if self.target_name != "bfactor_z":
            return []
        labels = [fd.get("label", "") for fd in record.features]
        lines: List[str] = []
        if not labels:
            return [
                "The current protein world model is still an intercept-only baseline. "
                "It has not yet committed to a physical explanation of residue flexibility."
            ]
        if labels == ["gnm_fluct_z"] or "gnm_fluct_z" in labels and len(labels) == 1:
            lines.append(
                "The model's ontology is the classical elastic-network view: residues "
                "with larger Gaussian Network Model fluctuations tend to have larger "
                "experimental B-factors. This means the first accepted explanation is "
                "global contact-network mechanics, not residue identity or a black-box fit."
            )
        if "terminal_exposure" in labels:
            lines.append(
                "The model has added an explicit boundary term. In this ontology, the "
                "uniform elastic network is not sufficient: chain-end exposure carries "
                "additional explanatory power for experimental flexibility."
            )
        if "mode1_abs_z" in labels:
            lines.append(
                "The model has added participation in the slowest GNM mode. This shifts "
                "the explanation from total local fluctuation alone toward collective, "
                "domain-scale motion: residues moving strongly in the first global mode "
                "are treated as systematically more flexible."
            )
        if "mode1_abs_z^2" in labels:
            lines.append(
                "The model has made slow-mode participation nonlinear. Squaring the "
                "first-mode amplitude treats both positive and negative displacements "
                "as equivalent participation in the same collective motion."
            )
        if "gnm_fluct_log_z" in labels:
            lines.append(
                "The model uses log-transformed GNM fluctuation, which compresses very "
                "large predicted fluctuations. This is a more robust flexibility scale "
                "when the dataset includes proteins with extreme terminal or loop motion."
            )
        if any("contact_degree" in label for label in labels):
            lines.append(
                "The model uses local contact density, so part of the explanation is "
                "packing: highly connected residues are constrained, while weakly packed "
                "residues are easier to move."
            )
        if any(label.startswith("[is_") or "[is_" in label for label in labels):
            lines.append(
                "The model has introduced residue-class information. That means the "
                "purely mechanical C-alpha network is being supplemented by local chemical "
                "context, though MDL only keeps such terms if they pay for themselves."
            )
        if not lines:
            lines.append(
                "The accepted features are protein-derived DAG features. Interpret this "
                "iteration by reading the equation as a compact physical hypothesis for "
                "normalized experimental B-factor."
            )
        if record.break_detected:
            lines.append(
                f"The newly revealed data forced a structural model change (`{record.break_type}`), "
                f"reducing total description length by {record.break_bits_gain:.1f} bits relative "
                "to refitting the previous DAG on the same revealed observations."
            )
        return lines

    def write_report(self) -> None:
        lines: List[str] = ["# Agentic World-Model Discovery Report (DAG + MDL)\n"]
        lines.append(f"LLM enabled: **{self.llm_enabled}**\n")
        lines.append("## What MDL means")
        lines.append(MDL_EXPLANATION)
        lines.append("")
        if self.dataset.system_description:
            lines.append("## System context")
            lines.append(self.dataset.system_description)
            lines.append("")
        if self.dataset_analysis:
            lines.append("## Pre-run dataset analysis")
            lines.append(f"**{self.dataset_analysis.get('title', 'Dataset analysis')}**")
            lines.append("")
            if self.dataset_analysis.get("system_summary"):
                lines.append(self.dataset_analysis["system_summary"])
                lines.append("")
            sections = [
                ("Observables", "observables"),
                ("Protocol logic", "protocol_logic"),
                ("Likely world-model challenges", "likely_world_model_challenges"),
                ("Discovery expectations", "discovery_expectations"),
                ("Caveats", "caveats"),
            ]
            for title, key in sections:
                items = self.dataset_analysis.get(key, [])
                if items:
                    lines.append(f"### {title}")
                    for item in items:
                        lines.append(f"- {item}")
                    lines.append("")
        for r in self.history:
            lines.append(f"## Iteration {r.version}")
            if r.revealed_slice:
                lines.append(f"- Revealed slice: `{r.revealed_slice}` ({self.stage_labels[r.revealed_slice]})")
                lines.append(f"- Breaker hypothesis: {r.breaker_hypothesis or 'n/a'}")
                lines.append(f"- Collection request: {r.collection_request or 'n/a'}")
                lines.append(f"- Collected points: {r.revealed_n_points}")
            lines.append(f"- Features: {r.bits['k_features']}")
            lines.append(f"- L_model: {r.bits['L_model']:.1f} bits | L_data: {r.bits['L_data']:.1f} bits | L_total: {r.bits['L_total']:.1f} bits")
            lines.append(f"- RMSE: {r.bits['rmse']:.4f} | R^2: {r.bits['r2']:.4f}")
            lines.append(f"- Break detected: {r.break_detected} (`{r.break_type}`, gain {r.break_bits_gain:+.1f} bits)")
            lines.append("- Equations:")
            for eq in r.equation_lines:
                lines.append(f"  - `{eq}`")
            lines.append("- Features (DAG leaves -> product):")
            for fd in r.features:
                lines.append(f"  - `{fd['label']}`")
            interpretation = self.protein_model_interpretation(r)
            if interpretation:
                lines.append("- Model interpretation:")
                for item in interpretation:
                    lines.append(f"  - {item}")
            if r.builder_hypothesis:
                lines.append("- Builder diagnosis: " + str(r.builder_hypothesis.get("diagnosis", "n/a")))
                lines.append("- Builder physical interpretation: " + str(r.builder_hypothesis.get("physical_interpretation", "n/a")))
                latent = r.builder_hypothesis.get("candidate_latent_variables", [])
                if latent:
                    lines.append("- Builder candidate latent/internal variables:")
                    for item in latent:
                        lines.append(f"  - {item}")
                predictions = r.builder_hypothesis.get("falsifiable_predictions", [])
                if predictions:
                    lines.append("- Builder falsifiable predictions:")
                    for item in predictions:
                        lines.append(f"  - {item}")
                proposed = r.builder_hypothesis.get("proposed_features", [])
                if proposed:
                    lines.append("- Builder proposed features before MDL selection:")
                    for item in proposed:
                        lines.append(f"  - `{item.get('label', '?')}`: {item.get('rationale', '')}")
            lines.append(f"- Builder rationale: {r.builder_rationale}")
            if r.breaker_rationale:
                lines.append(f"- Breaker rationale: {r.breaker_rationale}")
            lines.append("")
        safe_write_text(self.outdir / "report.md", "\n".join(lines))

    # --- rendering -----
    def render_iteration_frame(self, model: DAGModel, record: WorldModelRecord,
                               newly_revealed_slice: Optional[str]) -> None:
        frame_path = self.outdir / f"frame_iter_{record.version:02d}.png"
        prev = self.history[-2] if len(self.history) >= 2 and self.history[-1].version == record.version else None

        fig = plt.figure(figsize=(16, 10), constrained_layout=True)
        gs = gridspec.GridSpec(3, 2, figure=fig, width_ratios=[1.25, 1.0], height_ratios=[1.0, 0.55, 0.55])
        ax_data = fig.add_subplot(gs[0:2, 0])
        ax_graph = fig.add_subplot(gs[0, 1])
        ax_bits = fig.add_subplot(gs[1, 1])
        ax_search = fig.add_subplot(gs[2, 0])
        ax_info = fig.add_subplot(gs[2, 1])

        # Data panel
        all_obs = self.obs
        seen = self.seen_observations()
        seen_ids = {o.idx for o in seen}
        new_ids = {o.idx for o in self.stage_to_obs[newly_revealed_slice]} if newly_revealed_slice else set()
        hidden = [o for o in all_obs if o.idx not in seen_ids]
        seen_old = [o for o in seen if o.idx not in new_ids]
        new_obs = [o for o in seen if o.idx in new_ids]
        if self.target_name == "bfactor_z" and seen:
            X_seen, y_seen = obs_to_XY(seen)
            y_pred = model.evaluate(X_seen)
            stage_colors = plt.get_cmap("tab10")
            sid_to_color = {sid: stage_colors(i % 10) for i, sid in enumerate(self.seen_stage_ids)}
            for sid in self.seen_stage_ids:
                idxs = [i for i, obs in enumerate(seen) if obs.stage_id == sid]
                if not idxs:
                    continue
                ax_data.scatter(
                    y_seen[idxs], y_pred[idxs],
                    s=26,
                    alpha=0.78,
                    color=sid_to_color[sid],
                    edgecolors="white",
                    linewidths=0.35,
                    label=self.stage_labels.get(sid, sid),
                )
            lim_lo = float(np.nanmin([np.min(y_seen), np.min(y_pred)]))
            lim_hi = float(np.nanmax([np.max(y_seen), np.max(y_pred)]))
            pad = 0.08 * max(1e-6, lim_hi - lim_lo)
            ax_data.plot([lim_lo - pad, lim_hi + pad], [lim_lo - pad, lim_hi + pad],
                         color="#111827", linewidth=1.2, linestyle="--", label="ideal")
            ax_data.set_xlim(lim_lo - pad, lim_hi + pad)
            ax_data.set_ylim(lim_lo - pad, lim_hi + pad)
            ax_data.set_aspect("equal", adjustable="box")
            ax_data.set_xlabel("experimental B-factor z")
            ax_data.set_ylabel("DAG prediction")
            residual = y_seen - y_pred
            inset = ax_data.inset_axes([0.60, 0.08, 0.36, 0.30])
            stage_rmse = []
            for sid in self.seen_stage_ids:
                idxs = [i for i, obs in enumerate(seen) if obs.stage_id == sid]
                if not idxs:
                    continue
                rmse = float(np.sqrt(np.mean(residual[idxs] ** 2)))
                stage_rmse.append((self.stage_labels.get(sid, sid), rmse, sid_to_color[sid]))
            inset.bar(
                range(len(stage_rmse)),
                [item[1] for item in stage_rmse],
                color=[item[2] for item in stage_rmse],
                edgecolor="#1f2937",
            )
            inset.set_title("RMSE by revealed stage", fontsize=7.5)
            inset.set_xticks(range(len(stage_rmse)))
            inset.set_xticklabels([f"S{i}" for i in range(len(stage_rmse))], fontsize=7)
            inset.tick_params(axis="y", labelsize=7)
            inset.grid(alpha=0.2, axis="y")
            ax_data.legend(fontsize=7, loc="upper left", framealpha=0.92)
            ax_data.set_title(
                f"Iteration {record.version}: predicted vs experimental residue flexibility"
            )
        else:
            x_of = lambda rows: [float(observation_inputs(o).get(self.primary_x, o.strain)) for o in rows]
            y_of = lambda rows: [observation_target(o) for o in rows]
            if hidden:
                ax_data.scatter(x_of(hidden), y_of(hidden), s=16, alpha=0.3, marker=MarkerStyle("x"), label="unrevealed")
            if seen_old:
                ax_data.scatter(x_of(seen_old), y_of(seen_old), s=22, alpha=0.9, label="revealed")
            if new_obs:
                ax_data.scatter(x_of(new_obs), y_of(new_obs), s=50, alpha=1.0,
                                linewidths=1.2, edgecolors="black", label="breaker revealed")

            if seen:
                X_seen, y_seen = obs_to_XY(seen)
                y_pred = model.evaluate(X_seen)
                x_seen = X_seen[self.primary_x].astype(float)
                ax_data.vlines(
                    x_seen, y_seen, y_pred, color="#111", alpha=0.18,
                    linewidth=0.8, label="residuals on revealed observations",
                )
                ax_data.scatter(
                    x_seen, y_pred, s=34, facecolors="none", edgecolors="#111",
                    linewidths=1.1, marker="D", label="world-model predictions",
                    zorder=4,
                )

            ax_data.set_title(f"Iteration {record.version}: world model evaluated on revealed observations")
            ax_data.set_xlabel(self.primary_x)
            ax_data.set_ylabel(self.target_name)
            ax_data.legend(fontsize=8, loc="best")

        if new_obs and self.target_name == "bfactor_z":
            ax_data.text(
                0.02, 0.08,
                f"Breaker revealed: {self.stage_labels.get(newly_revealed_slice, newly_revealed_slice)}\n"
                f"n={len(new_obs)} residues",
                transform=ax_data.transAxes,
                fontsize=8.0,
                va="bottom",
                bbox={"boxstyle": "round", "fc": "#fff7ed", "ec": "#f97316", "alpha": 0.94},
            )
        eq_text = "\n".join(model.equation_lines()[:1])
        bits = record.bits
        ax_data.text(
            0.02, 0.98,
            f"Features: {bits['k_features']}\n"
            f"L_model: {bits['L_model']:.1f} bits\n"
            f"L_data:  {bits['L_data']:.1f} bits\n"
            f"L_total: {bits['L_total']:.1f} bits\n"
            f"RMSE: {bits['rmse']:.4f} | R^2: {bits['r2']:.4f}\n\n"
            f"{eq_text}",
            transform=ax_data.transAxes, va="top", ha="left", fontsize=8.5,
            bbox={"boxstyle": "round", "fc": "white", "alpha": 0.9},
        )

        # DAG panel
        self.draw_world_graph(ax_graph, record, prev)

        # Bits budget panel: stacked bar across iterations
        iters = [r.version for r in self.history]
        l_models = [r.bits["L_model"] for r in self.history]
        l_datas = [r.bits["L_data"] for r in self.history]
        ax_bits.bar(iters, l_models, color="#6baed6", edgecolor="#222", label="L_model")
        ax_bits.bar(iters, l_datas, bottom=l_models, color="#fd8d3c", edgecolor="#222", label="L_data")
        ax_bits.set_title("MDL budget (bits)")
        ax_bits.set_xlabel("iteration"); ax_bits.set_ylabel("bits")
        ax_bits.legend(fontsize=8, loc="upper right")
        ax_bits.grid(alpha=0.2, axis="y")

        # Inner-search trace
        if record.search_trace:
            xs = [s["step"] for s in record.search_trace]
            bits_after = [s["bits_after"] if s["bits_after"] == s["bits_after"] else np.nan for s in record.search_trace]
            accepts = [s["accepted"] for s in record.search_trace]
            ax_search.plot(xs, bits_after, color="#888", linewidth=1.0, alpha=0.5, label="proposal bits")
            acc_xs = [s["step"] for s in record.search_trace if s["accepted"]]
            acc_ys = [s["bits_after"] for s in record.search_trace if s["accepted"]]
            ax_search.scatter(acc_xs, acc_ys, s=30, color="#2ca02c", label="accepted", zorder=3)
            rej_xs = [s["step"] for s, a in zip(record.search_trace, accepts) if not a and s["bits_after"] == s["bits_after"]]
            rej_ys = [s["bits_after"] for s, a in zip(record.search_trace, accepts) if not a and s["bits_after"] == s["bits_after"]]
            ax_search.scatter(rej_xs, rej_ys, s=14, color="#d62728", alpha=0.7, marker=MarkerStyle("x"), label="rejected")
            ax_search.set_title(f"Inner hill-climb trace ({record.search_summary['accepted_count']} accepted / {record.search_summary['total_steps']} steps)")
            ax_search.set_xlabel("inner step"); ax_search.set_ylabel("total bits")
            ax_search.legend(fontsize=8, loc="upper right"); ax_search.grid(alpha=0.25)
        else:
            ax_search.text(0.5, 0.5, "no inner steps", ha="center", va="center"); ax_search.axis("off")

        # Info panel
        ax_info.axis("off")
        ax_info.text(
            0.02, 0.98,
            f"Break detected: {record.break_detected} ({record.break_type})\n"
            f"Break bits gain vs previous DAG refit: {record.break_bits_gain:+.2f}\n"
            f"Breaker hypothesis: {record.breaker_hypothesis or 'n/a'}\n"
            f"Collection request: {record.collection_request or 'n/a'}\n"
            f"Search converged: {record.search_summary['converged']}\n"
            f"Stop reason: {record.search_summary['stop_reason']}\n\n"
            f"Builder rationale:\n{record.builder_rationale}\n\n"
            f"Breaker rationale:\n{record.breaker_rationale or 'n/a'}",
            transform=ax_info.transAxes, va="top", ha="left", fontsize=8.5, wrap=True,
        )

        fig.suptitle("Breaking and expanding the DAG world model (MDL-scored)", fontsize=15)
        fig.savefig(frame_path, dpi=160)
        plt.close(fig)
        self.frame_paths.append(frame_path)

    def draw_world_graph(self, ax: plt.Axes, record: WorldModelRecord, prev: Optional[WorldModelRecord]) -> None:
        G = nx.DiGraph()
        for n in record.nodes:
            G.add_node(n["id"], **n)
        for e in record.edges:
            G.add_edge(e["source"], e["target"], **e)

        layer_of = {"observable": 0, "factor": 1, "feature": 2, "target": 3}
        layers: Dict[str, int] = {}
        for n in record.nodes:
            layers[n["id"]] = layer_of.get(n.get("kind", "observable"), 1)
        groups: Dict[int, List[str]] = {}
        for nid, layer in layers.items():
            groups.setdefault(layer, []).append(nid)
        pos: Dict[str, Tuple[float, float]] = {}
        for layer, node_ids in groups.items():
            node_ids = sorted(node_ids)
            ys = np.linspace(0.92, 0.08, len(node_ids)) if len(node_ids) > 1 else np.array([0.5])
            for nid, y in zip(node_ids, ys):
                pos[nid] = (layer, float(y))

        prev_ids = set() if prev is None else {n["id"] for n in prev.nodes}
        curr_ids = {n["id"] for n in record.nodes}
        added_nodes = curr_ids - prev_ids

        color_map = {"observable": "#a6cee3", "factor": "#fdbf6f", "feature": "#b2df8a", "target": "#cab2d6"}
        node_colors = [color_map.get(G.nodes[n].get("kind", "observable"), "#dddddd") for n in G.nodes()]
        edge_colors = ["#2e7d32" if n in added_nodes else "#444444" for n in G.nodes()]
        linewidths = [2.4 if n in added_nodes else 1.1 for n in G.nodes()]
        labels = {n: G.nodes[n].get("label", n) for n in G.nodes()}

        nx.draw_networkx_nodes(G, pos, ax=ax, node_size=1300, node_color=node_colors,
                               edgecolors=edge_colors, linewidths=linewidths)
        nx.draw_networkx_labels(G, pos, labels=labels, ax=ax, font_size=7)
        nx.draw_networkx_edges(G, pos, ax=ax, arrows=True, arrowsize=14, width=1.1, alpha=0.8)
        ax.set_title("World-model DAG  (inputs -> factors -> features -> target)")
        ax.set_xlim(-0.45, 3.45)
        ax.set_ylim(-0.18, 1.12)
        ax.axis("off")
        ax.text(0.02, 0.02,
                "Blue=observed  Orange=factor  Green=feature  Purple=target\nGreen border=added this iteration",
                transform=ax.transAxes, va="bottom", ha="left", fontsize=7.5,
                bbox={"boxstyle": "round", "fc": "white", "alpha": 0.9})

    def write_paper_figures(self) -> None:
        if not getattr(self.args, "paper_figures", True):
            return
        if not self.history:
            return
        figdir = self.outdir / "paper_figures"
        figdir.mkdir(parents=True, exist_ok=True)
        self.render_paper_mdl_trajectory(figdir / "mdl_trajectory.svg")
        self.render_paper_discovery_timeline(figdir / "discovery_timeline.svg")
        self.render_paper_model_evolution(figdir / "model_evolution.svg")
        self.render_paper_dag_evolution(figdir / "dag_evolution.svg")

    def _revealed_stage_ids_through(self, version: int) -> List[str]:
        revealed = list(self.dataset.initial_stage_ids)
        for r in self.history:
            if r.version == 0:
                continue
            if r.version > version:
                break
            if r.revealed_slice and r.revealed_slice not in revealed:
                revealed.append(r.revealed_slice)
        return revealed

    def _observations_for_stage_ids(self, stage_ids: Sequence[str]) -> List[Observation]:
        out: List[Observation] = []
        for sid in stage_ids:
            out.extend(self.stage_to_obs.get(sid, []))
        return out

    def render_paper_mdl_trajectory(self, out_path: Path) -> None:
        iters = [r.version for r in self.history]
        l_model = [r.bits["L_model"] for r in self.history]
        l_data = [r.bits["L_data"] for r in self.history]
        l_total = [r.bits["L_total"] for r in self.history]
        rmse = [r.bits["rmse"] for r in self.history]

        fig, (ax_bits, ax_fit) = plt.subplots(1, 2, figsize=(10.5, 3.8), constrained_layout=True)
        ax_bits.bar(iters, l_model, color="#356d8f", edgecolor="#222", label="model bits")
        ax_bits.bar(iters, l_data, bottom=l_model, color="#e08b46", edgecolor="#222", label="data bits")
        ax_bits.plot(iters, l_total, color="#111", marker="o", linewidth=1.8, label="total bits")
        ax_bits.set_title("MDL budget")
        ax_bits.set_xlabel("Discovery iteration")
        ax_bits.set_ylabel("Description length (bits)")
        ax_bits.legend(frameon=False, fontsize=8)
        ax_bits.grid(alpha=0.22, axis="y")

        ax_fit.plot(iters, rmse, marker="s", color="#2f7d32", linewidth=2)
        for r in self.history[1:]:
            ax_fit.annotate(
                f"{r.break_bits_gain:+.1f} bits",
                xy=(r.version, r.bits["rmse"]),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
                fontsize=7,
            )
        ax_fit.set_title("Fit after each collected slice")
        ax_fit.set_xlabel("Discovery iteration")
        ax_fit.set_ylabel("RMSE on revealed data")
        ax_fit.grid(alpha=0.22)

        fig.savefig(out_path, format="svg")
        plt.close(fig)

    def render_paper_discovery_timeline(self, out_path: Path) -> None:
        fig, ax = plt.subplots(figsize=(10.5, 3.2))
        fig.subplots_adjust(left=0.03, right=0.98, top=0.92, bottom=0.10)
        ax.axis("off")
        xs = np.linspace(0.08, 0.92, max(1, len(self.history)))
        y = 0.58
        for i, r in enumerate(self.history):
            x = xs[i]
            color = "#1f77b4" if i == 0 else "#2ca02c" if r.break_detected else "#999999"
            ax.scatter([x], [y], s=520, color=color, edgecolor="#222", zorder=3)
            ax.text(x, y, str(r.version), ha="center", va="center", color="white", weight="bold", fontsize=10)
            if i > 0:
                ax.plot([xs[i - 1], x], [y, y], color="#555", linewidth=1.5, zorder=1)
            label = (
                "initial\n" + self.stage_labels.get(self.seen_stage_ids[0], self.seen_stage_ids[0])
                if r.version == 0 else self.stage_labels.get(str(r.revealed_slice), str(r.revealed_slice))
            )
            ax.text(x, y - 0.18, label, ha="center", va="top", fontsize=8)
            if r.version > 0:
                ax.text(
                    x, y + 0.16,
                    f"{r.break_type}\n{r.break_bits_gain:+.1f} bits",
                    ha="center", va="bottom", fontsize=8,
                )
        ax.text(0.02, 0.95, "Breaker collection sequence", transform=ax.transAxes,
                ha="left", va="top", fontsize=12, weight="bold")
        ax.text(0.02, 0.10,
                "Each step: Breaker states a falsification hypothesis, collects one protocol slice, Builder rebuilds the DAG by MDL search.",
                transform=ax.transAxes, ha="left", va="bottom", fontsize=8.5)
        fig.savefig(out_path, format="svg")
        plt.close(fig)

    def render_paper_model_evolution(self, out_path: Path) -> None:
        n = len(self.history)
        fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.5), sharey=True, constrained_layout=True)
        if n == 1:
            axes = np.asarray([axes])
        cmap = plt.get_cmap("tab10")
        stage_colors = {sid: cmap(i % 10) for i, sid in enumerate(self.stage_order)}
        for ax, record, model in zip(axes, self.history, self.models):
            revealed = set(self._revealed_stage_ids_through(record.version))
            revealed_obs = self._observations_for_stage_ids(list(revealed))
            for sid in self.stage_order:
                obs = self.stage_to_obs.get(sid, [])
                if not obs:
                    continue
                alpha = 0.88 if sid in revealed else 0.15
                marker = "o" if sid in revealed else "x"
                ax.scatter(
                    [float(observation_inputs(o).get(self.primary_x, o.strain)) for o in obs],
                    [observation_target(o) for o in obs],
                    s=14, alpha=alpha, marker=marker, color=stage_colors.get(sid, "#444"),
                    label=self.stage_labels[sid] if record.version == 0 else None,
                )
            if revealed_obs:
                X_seen, y_seen = obs_to_XY(revealed_obs)
                y_pred = model.evaluate(X_seen)
                x_seen = X_seen[self.primary_x].astype(float)
                ax.vlines(x_seen, y_seen, y_pred, color="#111", alpha=0.16, linewidth=0.55)
                ax.scatter(
                    x_seen, y_pred, s=18, facecolors="none",
                    edgecolors="#111", linewidths=0.8, marker="D",
                    label="model prediction" if record.version == 0 else None,
                    zorder=4,
                )
            title = f"iter {record.version}\nk={record.bits['k_features']}, RMSE={record.bits['rmse']:.2f}"
            ax.set_title(title, fontsize=9)
            ax.set_xlabel(self.primary_x)
            ax.grid(alpha=0.18)
        axes[0].set_ylabel(self.target_name)
        axes[0].legend(frameon=False, fontsize=6, loc="upper left")
        fig.suptitle("World-model fit as data are collected", fontsize=12)
        fig.savefig(out_path, format="svg")
        plt.close(fig)

    def render_paper_dag_evolution(self, out_path: Path) -> None:
        n = len(self.history)
        cols = min(2, max(1, n))
        rows = int(np.ceil(n / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(6.4 * cols, 4.3 * rows))
        fig.subplots_adjust(left=0.03, right=0.98, top=0.90, bottom=0.05, wspace=0.08, hspace=0.18)
        axes = np.asarray(axes).reshape(-1)
        for i, (ax, record) in enumerate(zip(axes, self.history)):
            prev = self.history[i - 1] if i > 0 else None
            self.draw_world_graph(ax, record, prev)
            ax.set_title(f"DAG iter {record.version}", fontsize=9)
        for ax in axes[n:]:
            ax.axis("off")
        fig.suptitle("DAG world-model evolution", fontsize=12)
        fig.savefig(out_path, format="svg")
        plt.close(fig)


# -----------------------------
# CLI
# -----------------------------

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Agentic DAG world-model discovery with MDL scoring.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--outdir", type=str, default="world_model_demo_run")
    p.add_argument("--model", type=str, default="gpt-5.5")
    p.add_argument("--reasoning-effort", type=str, default="medium", choices=["low", "medium", "high"])
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--noise-std", type=float, default=0.9)
    p.add_argument("--oracle", choices=["none", "duffing"], default="none",
                   help="Use a live oracle adapter instead of a precomputed dataset. "
                        "`duffing` simulates a nonlinear oscillator on demand.")
    p.add_argument("--synthetic-example", choices=["cyclic", "fracture"], default="cyclic",
                   help="Built-in synthetic dataset to use when --dataset-json is not supplied")
    p.add_argument("--dataset-json", type=str, default=None,
                   help="Load a DiscoveryDataset JSON file instead of generating the built-in tensile-test dataset")
    p.add_argument("--search-steps", type=int, default=160, help="Max proposals per inner hill-climb")
    p.add_argument("--search-patience", type=int, default=30, help="Stop after this many consecutive rejections")
    p.add_argument("--search-restarts", type=int, default=3, help="Number of independent hill-climb starts per iteration; best is kept")
    p.add_argument("--breaker-mode", choices=["experimental", "oracle"], default="experimental",
                   help="experimental designs data collection without hidden labels; oracle uses legacy hidden-y slice scoring")
    p.add_argument("--collection-policy", choices=["physical", "unconstrained"], default="physical",
                   help="physical enforces stage prerequisites; unconstrained lets Breaker choose any remaining slice")
    p.add_argument("--no-llm", action="store_true")
    p.add_argument("--llm-builder", action="store_true",
                   help="Ask the LLM Builder to reason from revealed evidence and experiment "
                        "context, propose physical hypotheses/latent variables/falsifiable "
                        "predictions, and seed candidate DAG features. Features are accepted "
                        "only if they strictly reduce total MDL bits.")
    p.add_argument("--no-llm-dataset-analysis", dest="llm_dataset_analysis", action="store_false",
                   help="Use deterministic pre-run dataset analysis even when LLM use is enabled.")
    p.add_argument("--no-paper-figures", dest="paper_figures", action="store_false",
                   help="Skip SVG overview figures in outdir/paper_figures.")
    p.add_argument("--quiet", dest="verbose", action="store_false",
                   help="Suppress live progress logging; artifacts are still written.")
    p.set_defaults(llm_dataset_analysis=True, paper_figures=True, verbose=True)
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    runner = DiscoveryRunner(args)
    runner.run()
    print(f"Wrote outputs to: {runner.outdir}")
    if not runner.llm_enabled and not args.no_llm:
        print("OPENAI_API_KEY was not found. The run used deterministic offline heuristics instead.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
