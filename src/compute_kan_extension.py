#!/usr/bin/env python3
"""
Post-hoc computation of left Kan extensions across all iteration
transitions in a Builder/Breaker discovery run.

For each transition t → t+1, extracts the schema categories S_t and
S_{t+1}, computes the inclusion functor u, evaluates the comma
categories (u ↓ A') for every object A' in S_{t+1}, and determines
which fibers are transportable, which are empty (the categorical
obstruction), and what residual content the discovery move must supply.

Produces high-quality SVG/PNG figures for the paper.

Usage:
    python src/compute_kan_extension.py
    python src/compute_kan_extension.py --run-dir runs/protein_flex_llm_deep/discovery
"""
from __future__ import annotations

import argparse
import json
import math
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np

# MDL constants (must match dag_model.py)
FACTOR_TYPE_BITS = math.log2(6)   # 6 factor types
THRESHOLD_BITS = 8
COEFFICIENT_BITS = 16
N_OBSERVABLE_VARS = 14            # protein dataset has 14 observables


# =====================================================================
# Schema category representation
# =====================================================================

@dataclass
class SchemaObject:
    """An object (artifact type) in the schema category."""
    name: str
    kind: str          # "base_physics", "observable", "intermediate", "target", "parameter"
    description: str = ""
    fiber_size: int = 0  # |I_t(A)| — number of artifacts of this type


@dataclass
class SchemaMorphism:
    """A morphism (typed operation) in the schema category."""
    name: str
    source: str        # source object name
    target: str        # target object name (for multi-input, use tuple key)
    sources: Tuple[str, ...] = ()  # for multi-input morphisms
    description: str = ""
    bits_cost: float = 0.0  # MDL cost of this morphism


@dataclass
class SchemaCategory:
    """A schema category S_b with objects, morphisms, and fiber contents."""
    label: str
    objects: Dict[str, SchemaObject] = field(default_factory=dict)
    morphisms: Dict[str, SchemaMorphism] = field(default_factory=dict)

    def object_names(self) -> Set[str]:
        return set(self.objects.keys())

    def add_obj(self, name: str, kind: str, desc: str = "", fiber: int = 0):
        self.objects[name] = SchemaObject(name, kind, desc, fiber)

    def add_mor(self, name: str, source: str, target: str,
                sources: Tuple[str, ...] = (), desc: str = "", bits: float = 0.0):
        self.morphisms[name] = SchemaMorphism(name, source, target, sources, desc, bits)


@dataclass
class KanResult:
    """Result of the Kan extension computation for one object."""
    object_name: str
    status: str          # "shared", "new_generator_reachable",
                         # "new_composite_reachable", "new_isolated", "retracted"
    comma_objects: List[str]  # objects in the generator-level comma category
    composite_path: str       # how this type is reachable via composites (if any)
    transported_size: int    # |Lan_u I_t(A')|
    actual_size: int         # |I'_{t+1}(A')|
    residual_size: int       # |R(A')|
    bits_cost: float = 0.0   # structural bits for this object's residual


@dataclass
class TransitionResult:
    """Full Kan extension analysis for one transition."""
    iteration_from: int
    iteration_to: int
    break_type: str
    schema_old: SchemaCategory
    schema_new: SchemaCategory
    shared_objects: Set[str]
    new_objects: Set[str]
    retracted_objects: Set[str]
    kan_results: Dict[str, KanResult]
    total_discovery_bits: float
    structural_bits: float
    parameter_bits: float


# =====================================================================
# Build schema categories from run artifacts
# =====================================================================

def _base_physics_objects(n_chains: int, n_residues: int) -> List[Tuple[str, str, str, int]]:
    """Objects in the physics pipeline that are always present."""
    return [
        ("PDBChain", "base_physics", "Protein chain from PDB", n_chains),
        ("CalphaCoords", "base_physics", "C-alpha coordinate array", n_chains),
        ("ContactGraph", "base_physics", "C-alpha contact adjacency matrix (10 Å)", n_chains),
        ("KirchhoffMatrix", "base_physics", "GNM Kirchhoff/Laplacian matrix", n_chains),
        ("GNMSpectrum", "base_physics", "Eigenvalues + eigenvectors", n_chains),
        ("Compliance", "base_physics", "GNM mean-square fluctuation (pseudoinverse diagonal)", n_residues),
        ("ModeAmplitude", "base_physics", "Slowest eigenvector absolute displacement", n_residues),
        ("NormCompliance", "observable", "z-scored compliance (gnm_fluct_z)", n_residues),
        ("NormModeAmpl", "observable", "z-scored |mode1| (mode1_abs_z)", n_residues),
        ("BFactor", "target", "Experimental B-factor z-score (target)", n_residues),
    ]


def _base_physics_morphisms() -> List[Tuple[str, str, str, str]]:
    """Morphisms in the physics pipeline."""
    return [
        ("extract_ca", "PDBChain", "CalphaCoords", "Extract C-alpha trace"),
        ("build_contacts", "CalphaCoords", "ContactGraph", "Pairwise distance ≤ 10 Å"),
        ("kirchhoff", "ContactGraph", "KirchhoffMatrix", "Kirchhoff matrix construction"),
        ("diagonalize", "KirchhoffMatrix", "GNMSpectrum", "Eigendecomposition"),
        ("compliance", "GNMSpectrum", "Compliance", "Σ u²/λ (pseudoinverse diagonal)"),
        ("mode1_extract", "GNMSpectrum", "ModeAmplitude", "First nonzero eigenvector"),
        ("znorm_compliance", "Compliance", "NormCompliance", "Per-chain z-score"),
        ("abs_znorm_mode", "ModeAmplitude", "NormModeAmpl", "abs + per-chain z-score"),
    ]


# Chain counts per iteration (cumulative: 2 chains per stage)
STAGE_CHAINS = {0: 2, 1: 4, 2: 6, 3: 8}


def _n_residues(record: Dict) -> int:
    """Get accumulated residue count from the record's bits.n field."""
    return int(record.get("bits", {}).get("n", 0))


def build_schema_iter0(record: Dict) -> SchemaCategory:
    """Iter 0: bfactor_z = β · gnm_fluct_z (simple linear)."""
    sc = SchemaCategory("S₀ (iter 0)")
    n_c, n_r = STAGE_CHAINS[0], _n_residues(record)
    for name, kind, desc, fib in _base_physics_objects(n_c, n_r):
        sc.add_obj(name, kind, desc, fib)
    for name, src, tgt, desc in _base_physics_morphisms():
        sc.add_mor(name, src, tgt, desc=desc)
    # Explanatory structure: direct linear
    sc.add_mor("explain_linear", "NormCompliance", "BFactor",
               desc="y = α + β·gnm_fluct_z",
               bits=FACTOR_TYPE_BITS + math.log2(N_OBSERVABLE_VARS))
    sc.add_obj("Parameter_α₀", "parameter", "Intercept", 1)
    sc.add_obj("Parameter_β₀", "parameter", "Coefficient for gnm_fluct_z", 1)
    return sc


def build_schema_iter1(record: Dict) -> SchemaCategory:
    """Iter 1: terminal_exp × relu(gnm_fluct_z) + mode1_abs_z."""
    sc = SchemaCategory("S₁ (iter 1)")
    n_c, n_r = STAGE_CHAINS[1], _n_residues(record)
    for name, kind, desc, fib in _base_physics_objects(n_c, n_r):
        sc.add_obj(name, kind, desc, fib)
    for name, src, tgt, desc in _base_physics_morphisms():
        sc.add_mor(name, src, tgt, desc=desc)
    # Additional observable
    sc.add_obj("TerminalExposure", "observable",
               "Boundary/terminus proximity feature", n_r)
    sc.add_mor("compute_terminal", "PDBChain", "TerminalExposure",
               desc="1 - clip(nearest_terminus / (n/2))")
    # Intermediate types for feature 0: terminal_exp * relu(gnm_fluct_z)
    sc.add_obj("ReLUCompliance", "intermediate",
               "max(gnm_fluct_z - θ₁, 0)", n_r)
    sc.add_mor("relu_gate", "NormCompliance", "ReLUCompliance",
               desc="ReLU with threshold θ₁ = -1.26",
               bits=FACTOR_TYPE_BITS + math.log2(N_OBSERVABLE_VARS) + THRESHOLD_BITS)
    sc.add_obj("BoundaryProduct", "intermediate",
               "terminal_exp × relu(gnm_fluct_z)", n_r)
    sc.add_mor("boundary_multiply", "TerminalExposure", "BoundaryProduct",
               sources=("TerminalExposure", "ReLUCompliance"),
               desc="Product of terminal exp. and ReLU compliance")
    # Explanatory morphism
    sc.add_mor("explain_combined", "BoundaryProduct", "BFactor",
               sources=("BoundaryProduct", "NormModeAmpl"),
               desc="y = α + β₁·(term×relu) + β₂·mode1")
    # Parameters
    sc.add_obj("Parameter_α₁", "parameter", "Intercept", 1)
    sc.add_obj("Parameter_β₁", "parameter", "Coeff for boundary product", 1)
    sc.add_obj("Parameter_β₂", "parameter", "Coeff for mode1_abs_z", 1)
    sc.add_obj("Parameter_θ₁", "parameter", "ReLU threshold = -1.26", 1)
    return sc


def build_schema_iter2(record: Dict) -> SchemaCategory:
    """Iter 2: gnm_fluct_z + mode1_abs_z (additive, retracted terminal)."""
    sc = SchemaCategory("S₂ (iter 2)")
    n_c, n_r = STAGE_CHAINS[2], _n_residues(record)
    for name, kind, desc, fib in _base_physics_objects(n_c, n_r):
        sc.add_obj(name, kind, desc, fib)
    for name, src, tgt, desc in _base_physics_morphisms():
        sc.add_mor(name, src, tgt, desc=desc)
    # Explanatory structure: additive linear
    sc.add_mor("explain_additive", "NormCompliance", "BFactor",
               sources=("NormCompliance", "NormModeAmpl"),
               desc="y = α + β₁·gnm_fluct_z + β₂·mode1_abs_z")
    # Parameters
    sc.add_obj("Parameter_α₂", "parameter", "Intercept", 1)
    sc.add_obj("Parameter_β₃", "parameter", "Coeff for gnm_fluct_z", 1)
    sc.add_obj("Parameter_β₄", "parameter", "Coeff for mode1_abs_z", 1)
    return sc


def build_schema_iter3(record: Dict) -> SchemaCategory:
    """Iter 3: gnm_fluct_log_z × relu(mode1_abs_z + 2.27)."""
    sc = SchemaCategory("S₃ (iter 3)")
    n_c, n_r = STAGE_CHAINS[3], _n_residues(record)
    for name, kind, desc, fib in _base_physics_objects(n_c, n_r):
        sc.add_obj(name, kind, desc, fib)
    for name, src, tgt, desc in _base_physics_morphisms():
        sc.add_mor(name, src, tgt, desc=desc)
    # New intermediate types
    sc.add_obj("LogNormCompliance", "intermediate",
               "z-scored log-compliance (gnm_fluct_log_z)", n_r)
    sc.add_mor("log_znorm", "Compliance", "LogNormCompliance",
               desc="log(compliance) then z-score",
               bits=FACTOR_TYPE_BITS + math.log2(N_OBSERVABLE_VARS))
    sc.add_obj("ReLUModeAmpl", "intermediate",
               "max(mode1_abs_z + θ, 0)", n_r)
    sc.add_mor("relu_shift", "NormModeAmpl", "ReLUModeAmpl",
               desc="ReLU with shift θ = 2.27",
               bits=FACTOR_TYPE_BITS + math.log2(N_OBSERVABLE_VARS) + THRESHOLD_BITS)
    sc.add_obj("ModeConditionedCompliance", "intermediate",
               "log-compliance × ReLU(mode1)", n_r)
    sc.add_mor("multiply", "LogNormCompliance", "ModeConditionedCompliance",
               sources=("LogNormCompliance", "ReLUModeAmpl"),
               desc="Product feature: local compliance gated by collective mode")
    # Explanatory morphism
    sc.add_mor("explain_product", "ModeConditionedCompliance", "BFactor",
               desc="y = α + β · (logCompl × reluMode)")
    # Parameters
    sc.add_obj("Parameter_α₃", "parameter", "Intercept = -0.1332", 1)
    sc.add_obj("Parameter_β₅", "parameter", "Coeff = 0.2239", 1)
    sc.add_obj("Parameter_θ₂", "parameter", "ReLU shift = 2.2678", 1)
    return sc


SCHEMA_BUILDERS = [build_schema_iter0, build_schema_iter1,
                   build_schema_iter2, build_schema_iter3]


# =====================================================================
# Kan extension computation
# =====================================================================

def _generator_comma(
    obj_name: str,
    schema_new: SchemaCategory,
    shared: Set[str],
) -> List[str]:
    """Generator-level comma category: immediate morphisms from u(S_b) only.

    Returns old objects X with a single generating morphism u(X) → A'.
    Does NOT follow composites or multi-input product morphisms.
    """
    if obj_name in shared:
        return [obj_name]
    comma = []
    for mor in schema_new.morphisms.values():
        if mor.target == obj_name and not mor.sources:
            # Unary morphism with source in old schema
            if mor.source in shared:
                comma.append(mor.source)
    return comma


def _composite_reachability(
    schema_new: SchemaCategory,
    shared: Set[str],
) -> Dict[str, str]:
    """Transitive reachability through composites and multi-input morphisms.

    Returns a dict mapping each reachable new object to a human-readable
    path description.  Uses fixed-point iteration: start from shared
    objects, then iteratively add targets of morphisms whose sources are
    all reachable.
    """
    reachable: Dict[str, str] = {name: "shared (identity)" for name in shared}
    changed = True
    while changed:
        changed = False
        for mor in schema_new.morphisms.values():
            tgt = mor.target
            if tgt in reachable:
                continue
            # Determine all source objects for this morphism
            srcs = list(mor.sources) if mor.sources else [mor.source]
            if all(s in reachable for s in srcs):
                if len(srcs) == 1:
                    path = f"via {mor.name}: {srcs[0]} → {tgt}"
                else:
                    src_str = " × ".join(srcs)
                    path = f"via {mor.name}: ({src_str}) → {tgt}"
                reachable[tgt] = path
                changed = True
    return reachable


def compute_transition(
    iter_from: int,
    iter_to: int,
    schema_old: SchemaCategory,
    schema_new: SchemaCategory,
    record_old: Dict,
    record_new: Dict,
) -> TransitionResult:
    """Compute the full Kan extension analysis for one transition.

    Two levels of analysis are performed:
      1. **Generator-level** comma category — only immediate unary
         morphisms from u(S_b).  This is the 1-categorical shadow.
      2. **Composite reachability** — transitive closure through all
         morphisms in S_{b'}, including multi-input (product) morphisms.
         This is the multicategorical/hypergraph reading.

    An object whose generator-level comma category is empty but which
    is reachable via composites is labeled ``new_composite_reachable``:
    old evidence reaches it, but only through new intermediate structure
    (new morphisms, new product operations, or new parameters).
    """
    old_objs = schema_old.object_names()
    new_objs = schema_new.object_names()

    shared = old_objs & new_objs
    added = new_objs - old_objs
    retracted = old_objs - new_objs

    # Composite reachability (multicategorical reading)
    composite_reach = _composite_reachability(schema_new, shared)

    kan_results: Dict[str, KanResult] = {}

    for obj_name in sorted(new_objs):
        obj = schema_new.objects[obj_name]
        gen_comma = _generator_comma(obj_name, schema_new, shared)
        comp_path = composite_reach.get(obj_name, "")

        if obj_name in shared:
            old_fiber = schema_old.objects[obj_name].fiber_size
            new_fiber = obj.fiber_size
            kan_results[obj_name] = KanResult(
                object_name=obj_name,
                status="shared",
                comma_objects=gen_comma,
                composite_path="identity",
                transported_size=old_fiber,
                actual_size=new_fiber,
                residual_size=max(0, new_fiber - old_fiber),
            )
        elif gen_comma:
            # New but generator-reachable (non-empty 1-categorical comma)
            transported = sum(
                schema_old.objects[c].fiber_size
                for c in gen_comma if c in schema_old.objects
            )
            kan_results[obj_name] = KanResult(
                object_name=obj_name,
                status="new_generator_reachable",
                comma_objects=gen_comma,
                composite_path=comp_path,
                transported_size=transported,
                actual_size=obj.fiber_size,
                residual_size=max(0, obj.fiber_size - transported),
            )
        elif comp_path:
            # Generator-level comma is empty, but reachable via composites
            # (through new intermediate types or multi-input morphisms).
            # The Lan in the 1-categorical shadow is ∅; old evidence
            # reaches this type only through new structural morphisms.
            bits = 0.0
            for mor in schema_new.morphisms.values():
                if mor.target == obj_name:
                    bits += mor.bits_cost
            kan_results[obj_name] = KanResult(
                object_name=obj_name,
                status="new_composite_reachable",
                comma_objects=[],
                composite_path=comp_path,
                transported_size=0,
                actual_size=obj.fiber_size,
                residual_size=obj.fiber_size,
                bits_cost=bits,
            )
        else:
            # Truly isolated: unreachable even via composites
            bits = 0.0
            for mor in schema_new.morphisms.values():
                if mor.target == obj_name:
                    bits += mor.bits_cost
            kan_results[obj_name] = KanResult(
                object_name=obj_name,
                status="new_isolated",
                comma_objects=[],
                composite_path="",
                transported_size=0,
                actual_size=obj.fiber_size,
                residual_size=obj.fiber_size,
                bits_cost=bits,
            )

    # Track retracted objects
    for obj_name in sorted(retracted):
        obj = schema_old.objects[obj_name]
        kan_results[obj_name] = KanResult(
            object_name=obj_name,
            status="retracted",
            comma_objects=[],
            composite_path="",
            transported_size=obj.fiber_size,
            actual_size=0,
            residual_size=0,
        )

    # Discovery cost in bits
    bits_new = record_new.get("bits", {})
    bits_old = record_old.get("bits", {})

    # Structural bits: difference in L_model
    structural_bits = bits_new.get("L_model", 0) - bits_old.get("L_model", 0)

    # Parameter bits: new parameters × COEFFICIENT_BITS
    old_params = [o for o in old_objs if o.startswith("Parameter_")]
    new_params = [o for o in new_objs if o.startswith("Parameter_")]
    added_params = set(new_params) - set(old_params)
    parameter_bits = len(added_params) * COEFFICIENT_BITS

    # The total discovery cost from the MDL gate
    break_gain = record_new.get("break_bits_gain", 0)

    return TransitionResult(
        iteration_from=iter_from,
        iteration_to=iter_to,
        break_type=record_new.get("break_type", "none"),
        schema_old=schema_old,
        schema_new=schema_new,
        shared_objects=shared,
        new_objects=added,
        retracted_objects=retracted,
        kan_results=kan_results,
        total_discovery_bits=break_gain,
        structural_bits=max(0, structural_bits),
        parameter_bits=parameter_bits,
    )


# =====================================================================
# Figure rendering
# =====================================================================

_STATUS_COLORS = {
    "shared": "#d0e8f2",
    "new_generator_reachable": "#d4edda",
    "new_composite_reachable": "#fff3cd",
    "new_isolated": "#fde4c8",
    "retracted": "#f8d7da",
}
_STATUS_EDGE = {
    "shared": "#3a7ca5",
    "new_generator_reachable": "#2d8e2d",
    "new_composite_reachable": "#856404",
    "new_isolated": "#c48220",
    "retracted": "#c44040",
}
_STATUS_LABELS = {
    "shared": "Shared (transported)",
    "new_generator_reachable": "New, generator-reachable",
    "new_composite_reachable": "New, composite-reachable only",
    "new_isolated": "Isolated (Lan = ∅)",
    "retracted": "Retracted",
}


def render_transition_table(
    result: TransitionResult,
    out_path: Path,
    dpi: int = 300,
) -> None:
    """Render a single transition's Kan extension as a table figure."""
    # Sort results: shared first, then new_reachable, new_isolated, retracted
    order = {"shared": 0, "new_reachable": 1, "new_isolated": 2, "retracted": 3}
    items = sorted(result.kan_results.values(),
                   key=lambda r: (order.get(r.status, 9), r.object_name))

    n_rows = len(items)
    fig_height = max(3.5, 0.38 * n_rows + 2.0)
    fig, ax = plt.subplots(figsize=(12, fig_height))
    fig.patch.set_facecolor("white")
    ax.axis("off")

    # Title
    ax.set_title(
        f"Kan Extension: iter {result.iteration_from} → {result.iteration_to}  "
        f"({result.break_type.replace('_', ' ')})",
        fontsize=13, fontweight="bold", pad=15,
    )

    # Table headers
    headers = ["Object A'", "Status", "(u ↓ A') gen.", "|Lan_u I_t|",
               "|I'_{t+1}|", "|Residual|"]
    col_x = [0.02, 0.22, 0.42, 0.62, 0.74, 0.86]
    y_top = 0.92
    row_h = 0.038 if n_rows > 15 else 0.045

    # Header
    for j, hdr in enumerate(headers):
        ax.text(col_x[j], y_top, hdr,
                transform=ax.transAxes, fontsize=8, fontweight="bold",
                family="monospace", va="top")
    ax.plot([0.01, 0.99], [y_top - 0.015, y_top - 0.015],
            transform=ax.transAxes, color="#888", linewidth=0.5)

    # Rows
    for i, item in enumerate(items):
        y = y_top - 0.025 - (i + 1) * row_h
        fc = _STATUS_COLORS.get(item.status, "#eee")

        # Row background
        rect = mpatches.FancyBboxPatch(
            (0.01, y - row_h * 0.35), 0.98, row_h * 0.8,
            boxstyle="round,pad=0.003,rounding_size=0.005",
            facecolor=fc, edgecolor="none", alpha=0.5,
            transform=ax.transAxes,
        )
        ax.add_patch(rect)

        comma_str = ", ".join(item.comma_objects) if item.comma_objects else "∅"
        if len(comma_str) > 25:
            comma_str = comma_str[:22] + "..."

        vals = [
            item.object_name,
            item.status.replace("_", " "),
            comma_str,
            str(item.transported_size) if item.status != "retracted" else "—",
            str(item.actual_size) if item.status != "retracted" else "0",
            str(item.residual_size) if item.residual_size > 0 else "—",
        ]
        for j, val in enumerate(vals):
            ax.text(col_x[j], y, val,
                    transform=ax.transAxes, fontsize=7,
                    family="monospace", va="center")

    # Summary box at bottom
    y_summary = y_top - 0.025 - (n_rows + 2) * row_h
    summary_text = (
        f"Shared: {len(result.shared_objects)}  |  "
        f"New: {len(result.new_objects)}  |  "
        f"Retracted: {len(result.retracted_objects)}\n"
        f"MDL break gain: {result.total_discovery_bits:+.1f} bits  |  "
        f"ΔL_model: {result.structural_bits:.1f} bits  |  "
        f"New parameter bits: {result.parameter_bits:.0f} bits"
    )
    ax.text(0.5, max(0.02, y_summary), summary_text,
            transform=ax.transAxes, ha="center", va="top",
            fontsize=9, family="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#f0f4f8",
                      edgecolor="#b0c4de"))

    # Legend
    y_leg = max(0.01, y_summary - 0.06)
    for i, (status, label) in enumerate(_STATUS_LABELS.items()):
        lx = 0.05 + i * 0.24
        ax.add_patch(mpatches.Rectangle(
            (lx, y_leg), 0.015, 0.015,
            facecolor=_STATUS_COLORS[status],
            edgecolor=_STATUS_EDGE[status],
            linewidth=0.8, transform=ax.transAxes,
        ))
        ax.text(lx + 0.02, y_leg + 0.007, label,
                transform=ax.transAxes, fontsize=6, va="center")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    for suffix in [".svg", ".png"]:
        fig.savefig(out_path.with_suffix(suffix), dpi=dpi,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)


def render_overview(
    transitions: List[TransitionResult],
    records: List[Dict],
    out_path: Path,
    dpi: int = 300,
) -> None:
    """Render the combined regime-enlargement trajectory."""
    n_trans = len(transitions)
    fig = plt.figure(figsize=(16, 8))
    fig.patch.set_facecolor("white")

    gs = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[1.0, 0.6],
                           hspace=0.35)

    # --- Top: regime evolution timeline ---
    ax_top = fig.add_subplot(gs[0])
    ax_top.set_xlim(-0.5, 3.5)
    ax_top.set_ylim(-0.3, 1.2)
    ax_top.axis("off")
    ax_top.set_title("Regime Enlargement Trajectory — Schema Evolution",
                     fontsize=14, fontweight="bold", pad=10)

    iter_x = [0, 1, 2, 3]
    iter_colors = ["#3a7ca5", "#2d8e2d", "#c44e20", "#7b3fa0"]
    break_types = ["initial", "regime_split", "ontology_break", "regime_split"]

    for i, rec in enumerate(records):
        x = iter_x[i]
        bits = rec.get("bits", {})
        k = bits.get("k_features", 0)
        r2 = bits.get("r2", 0)
        eq = (rec.get("equation_lines") or [""])[0]
        short_eq = eq if len(eq) < 50 else eq[:47] + "..."

        # Main circle
        ax_top.scatter([x], [0.5], s=800, color=iter_colors[i],
                       edgecolors="#222", linewidths=2, zorder=3)
        ax_top.text(x, 0.5, str(i), ha="center", va="center",
                    color="white", fontweight="bold", fontsize=14, zorder=4)

        # Arrow to next
        if i < 3:
            ax_top.annotate(
                "", xy=(x + 0.7, 0.5), xytext=(x + 0.3, 0.5),
                arrowprops=dict(arrowstyle="-|>", color="#555", lw=2),
            )
            # Transition label
            tr = transitions[i]
            label = tr.break_type.replace("_", " ")
            n_new = len(tr.new_objects)
            n_ret = len(tr.retracted_objects)
            ax_top.text(x + 0.5, 0.65,
                        f"{label}\n+{n_new} new, −{n_ret} retracted",
                        ha="center", va="bottom", fontsize=7, color="#555")

        # Info below
        info = f"k={k}  R²={r2:.2f}\n{short_eq}"
        ax_top.text(x, 0.15, info, ha="center", va="top",
                    fontsize=6.5, family="monospace", color="#334155")

        # Schema size above
        n_obj = len(SCHEMA_BUILDERS[i](rec).objects)
        n_mor = len(SCHEMA_BUILDERS[i](rec).morphisms)
        ax_top.text(x, 0.85, f"|Obj|={n_obj}  |Mor|={n_mor}",
                    ha="center", va="bottom", fontsize=7, color="#666")

    # --- Bottom: discovery cost bar chart ---
    ax_bot = fig.add_subplot(gs[1])
    labels = [f"iter {t.iteration_from}→{t.iteration_to}" for t in transitions]
    gains = [t.total_discovery_bits for t in transitions]
    colors = [iter_colors[t.iteration_to] for t in transitions]

    bars = ax_bot.bar(range(n_trans), gains, color=colors, edgecolor="#333",
                      linewidth=0.8, alpha=0.85)
    for bar, gain, tr in zip(bars, gains, transitions):
        ax_bot.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f"+{gain:.1f} bits\n({tr.break_type.replace('_', ' ')})",
                    ha="center", va="bottom", fontsize=8, fontweight="bold")

    # Count objects with empty generator-level comma per transition
    for i, tr in enumerate(transitions):
        n_gen_empty = sum(1 for r in tr.kan_results.values()
                          if r.status in ("new_composite_reachable", "new_isolated"))
        n_truly_isolated = sum(1 for r in tr.kan_results.values()
                               if r.status == "new_isolated")
        if n_gen_empty > 0:
            label_parts = [f"{n_gen_empty} gen-empty"]
            if n_truly_isolated > 0 and n_truly_isolated < n_gen_empty:
                label_parts.append(f"({n_truly_isolated} isolated)")
            ax_bot.text(i, gains[i] * 0.5,
                        "\n".join(label_parts),
                        ha="center", va="center", fontsize=7,
                        color="white", fontweight="bold")

    ax_bot.set_xticks(range(n_trans))
    ax_bot.set_xticklabels(labels, fontsize=10)
    ax_bot.set_ylabel("MDL Break Gain (bits)", fontsize=11)
    ax_bot.set_title("MDL Break Gain per Transition", fontsize=12,
                     fontweight="bold")
    ax_bot.grid(alpha=0.2, axis="y")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    for suffix in [".svg", ".png"]:
        fig.savefig(out_path.with_suffix(suffix), dpi=dpi,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)


# =====================================================================
# Console summary
# =====================================================================

def print_summary(transitions: List[TransitionResult]) -> None:
    """Print the categorical summary for the paper."""
    print("\n" + "=" * 72)
    print("LEFT KAN EXTENSION ANALYSIS — ALL TRANSITIONS")
    print("=" * 72)

    for tr in transitions:
        print(f"\n--- Transition iter {tr.iteration_from} → {tr.iteration_to} "
              f"({tr.break_type.replace('_', ' ')}) ---")
        print(f"  Shared objects:    {len(tr.shared_objects)}")
        print(f"  New objects:       {len(tr.new_objects)}")
        print(f"  Retracted objects: {len(tr.retracted_objects)}")

        # List new objects with their comma category status
        for name in sorted(tr.new_objects):
            kr = tr.kan_results.get(name)
            if kr:
                comma_str = ", ".join(kr.comma_objects) if kr.comma_objects else "∅"
                comp = f"  comp: {kr.composite_path}" if kr.composite_path and not kr.comma_objects else ""
                transported = kr.transported_size
                residual = kr.residual_size
                print(f"    {name:35s}  gen(u↓A')={comma_str:20s}  "
                      f"Lan={transported:>5d}  actual={kr.actual_size:>5d}  "
                      f"residual={residual:>5d}  [{kr.status}]"
                      f"{comp}")

        # List retracted objects
        for name in sorted(tr.retracted_objects):
            kr = tr.kan_results.get(name)
            if kr:
                print(f"    {name:35s}  [RETRACTED]  "
                      f"old fiber={kr.transported_size}")

        print(f"\n  MDL break gain: {tr.total_discovery_bits:+.1f} bits")

    # Key findings
    print("\n" + "=" * 72)
    print("KEY CATEGORICAL FINDINGS (two-level analysis)")
    print("=" * 72)

    for tr in transitions:
        gen_reachable = [name for name, kr in tr.kan_results.items()
                         if kr.status == "new_generator_reachable"]
        comp_reachable = [name for name, kr in tr.kan_results.items()
                          if kr.status == "new_composite_reachable"]
        isolated = [name for name, kr in tr.kan_results.items()
                    if kr.status == "new_isolated"]
        retracted = sorted(tr.retracted_objects)

        if gen_reachable:
            print(f"\n  iter {tr.iteration_from}→{tr.iteration_to}: "
                  f"Generator-reachable new types (non-empty 1-cat comma): "
                  f"{', '.join(gen_reachable)}.")
            print(f"    → Old evidence transports via a single new unary morphism.")
        if comp_reachable:
            print(f"\n  iter {tr.iteration_from}→{tr.iteration_to}: "
                  f"Composite-reachable new types (empty generator comma, "
                  f"reachable via multicategorical composites): "
                  f"{', '.join(comp_reachable)}.")
            for name in comp_reachable:
                kr = tr.kan_results[name]
                print(f"    {name}: {kr.composite_path}")
            print(f"    → In the 1-categorical shadow, Lan_u I_t = ∅ "
                  f"at these types.")
            print(f"    → In the multicategorical reading, old evidence "
                  f"reaches them through NEW intermediate structure.")
        if isolated:
            print(f"\n  iter {tr.iteration_from}→{tr.iteration_to}: "
                  f"Truly isolated types (unreachable even via composites): "
                  f"{', '.join(isolated)}.")
            print(f"    → Lan_u I_t(A') = ∅ in both readings.")
        if retracted:
            print(f"\n  iter {tr.iteration_from}→{tr.iteration_to}: "
                  f"RETRACTED types: {', '.join(retracted)}.")
            print(f"    → These fibers are lost; the regime contracted.")


# =====================================================================
# CLI
# =====================================================================

def main() -> int:
    p = argparse.ArgumentParser(
        description="Compute left Kan extensions across all iteration transitions.",
    )
    p.add_argument("--run-dir", type=str,
                   default="runs/protein_flex_llm_deep/discovery",
                   help="Path to the discovery output directory")
    p.add_argument("--outdir", type=str, default=None,
                   help="Output directory for figures (default: <run_dir>/figures_hq)")
    p.add_argument("--dpi", type=int, default=300)
    args = p.parse_args()

    run_dir = Path(args.run_dir).resolve()
    outdir = Path(args.outdir).resolve() if args.outdir else run_dir / "figures_hq"
    outdir.mkdir(parents=True, exist_ok=True)

    # Load all iteration records
    records = []
    for path in sorted(run_dir.glob("world_model_iter_*.json")):
        records.append(json.loads(path.read_text(encoding="utf-8")))
    if len(records) < 2:
        print("Need at least 2 iteration records.")
        return 1
    print(f"Loaded {len(records)} iteration records from {run_dir}")

    # Build schema categories
    schemas = [SCHEMA_BUILDERS[i](records[i]) for i in range(len(records))]

    # Compute Kan extensions for each transition
    transitions = []
    for i in range(len(records) - 1):
        tr = compute_transition(i, i + 1, schemas[i], schemas[i + 1],
                                records[i], records[i + 1])
        transitions.append(tr)

    # Print summary
    print_summary(transitions)

    # Render figures
    for tr in transitions:
        path = outdir / f"kan_extension_{tr.iteration_from}_{tr.iteration_to}"
        print(f"\nRendering {path.stem}...")
        render_transition_table(tr, path, dpi=args.dpi)

    print(f"\nRendering overview...")
    render_overview(transitions, records, outdir / "kan_extension_overview",
                    dpi=args.dpi)

    print(f"\nAll Kan extension figures written to: {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
