#!/usr/bin/env python3
"""
Re-render high-quality figures from an existing discovery run directory.

Reads world_model_iter_*.json and run_summary.json, then produces clean SVG/PNG
figures with non-overlapping labels, proper spacing, and publication styling.

Usage:
    python src/rerender_figures.py runs/protein_flex_llm_deep/discovery
    python src/rerender_figures.py runs/protein_flex_llm_deep/discovery --dpi 400
    python src/rerender_figures.py runs/protein_flex_llm_deep/discovery --outdir figures_hq
"""
from __future__ import annotations

import argparse
import json
import math
import re
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
import numpy as np


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_run(run_dir: Path) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    summary = json.loads((run_dir / "run_summary.json").read_text(encoding="utf-8"))
    models: List[Dict[str, Any]] = []
    for path in sorted(run_dir.glob("world_model_iter_*.json")):
        models.append(json.loads(path.read_text(encoding="utf-8")))
    return summary, models


# ---------------------------------------------------------------------------
# DAG graph rendering
# ---------------------------------------------------------------------------

_KIND_COLORS = {
    "observable": "#d0e8f2",
    "factor": "#fde4c8",
    "feature": "#d4edda",
    "target": "#ddd0eb",
}
_KIND_EDGE = {
    "observable": "#3a7ca5",
    "factor": "#c48220",
    "feature": "#3d8b3d",
    "target": "#6b4d8a",
}

_LABEL_MAP = {
    "gnm_fluct_z": "GNM fluct.",
    "gnm_fluct_log_z": "GNM log-fl.",
    "contact_degree_z": "contact deg.",
    "contact_degree_raw": "contact deg. raw",
    "terminal_exposure": "terminal exp.",
    "mode1_abs_z": "|mode1| z",
    "hinge_score_z": "hinge score",
    "chain_break_proximity": "chain break",
    "res_index_norm": "res. index",
    "is_hydrophobic": "hydrophobic",
    "is_charged": "charged",
    "is_polar": "polar",
    "is_gly": "is_gly",
    "is_pro": "is_pro",
    "is_terminal": "terminal",
    "bfactor_z": "B-factor z",
}


def _shorten_factor(factor: str) -> str:
    """Shorten a single factor label (may contain function wrappers)."""
    if factor in _LABEL_MAP:
        return _LABEL_MAP[factor]
    # relu(var-threshold) or relu(var--threshold)
    m = re.match(r'(relu|pow)\((\w+)(.*)\)', factor, re.IGNORECASE)
    if m:
        func, var, rest = m.group(1), m.group(2), m.group(3)
        short_var = _LABEL_MAP.get(var, var)
        return f"{func}({short_var}{rest})"
    # [var=val] indicator
    m = re.match(r'\[(\w+)=(\S+)\]', factor)
    if m:
        var, val = m.group(1), m.group(2)
        short_var = _LABEL_MAP.get(var, var)
        return f"[{short_var}={val}]"
    return factor


def _display_label(label: str) -> str:
    """Shorten a node label for display, handling products.

    Product labels longer than 22 chars are wrapped at the multiply sign
    so they fit inside a reasonably-sized box.
    """
    if label in _LABEL_MAP:
        return _LABEL_MAP[label]
    if " * " in label:
        parts = label.split(" * ")
        short_parts = [_shorten_factor(p.strip()) for p in parts]
        joined = " \u00d7 ".join(short_parts)
        if len(joined) > 22:
            return "\n\u00d7 ".join(short_parts)  # wrap at multiply sign
        return joined
    return _shorten_factor(label)


def _active_subgraph(record: Dict[str, Any]) -> Tuple[List[Dict], List[Dict]]:
    """Return only nodes/edges that participate in the active DAG.

    Removes observable nodes that have no outgoing edges (unused variables).
    """
    nodes = record.get("nodes", [])
    edges = record.get("edges", [])
    active_ids = set()
    for e in edges:
        active_ids.add(e["source"])
        active_ids.add(e["target"])
    active_nodes = [
        n for n in nodes
        if n["id"] in active_ids or n.get("kind") not in ("observable",)
    ]
    active_node_ids = {n["id"] for n in active_nodes}
    active_edges = [e for e in edges if e["source"] in active_node_ids and e["target"] in active_node_ids]
    return active_nodes, active_edges


def _fold_identity_chains(record: Dict[str, Any]) -> Dict[str, Any]:
    """Collapse identity factor nodes out of the DAG.

    Any factor with kind ``Ident`` is a pass-through — it adds a node and
    an edge without changing the value.  We fold *every* Ident factor
    (including those inside multi-factor product features), rewiring the
    observable edge directly to the feature node.

    Additionally, if an observable now feeds *only* a single purely-identity
    feature (single Ident factor), we absorb the observable too so the
    feature node alone represents the input.
    """
    features = record.get("features", [])
    nodes = list(record.get("nodes", []))
    edges = list(record.get("edges", []))

    # --- Step 1: fold every Ident factor regardless of feature size ---
    fold_ids: set = set()
    fold_map: dict = {}
    # Track purely-identity features (single Ident factor) for step 2
    identity_feat_ids: set = set()

    for i, feat in enumerate(features):
        factors = feat.get("factors", [])
        if len(factors) == 1 and factors[0].get("kind") == "Ident":
            identity_feat_ids.add(f"feat:{i}")
        for j, factor in enumerate(factors):
            if factor.get("kind") == "Ident":
                prefix = f"fact:{i}:{j}:"
                for n in nodes:
                    if n["kind"] == "factor" and n["id"].startswith(prefix):
                        fold_ids.add(n["id"])
                        fold_map[n["id"]] = f"feat:{i}"
                        break

    if not fold_map:
        return record

    new_nodes = [n for n in nodes if n["id"] not in fold_ids]
    new_edges = []
    for e in edges:
        if e["target"] in fold_ids:
            new_edges.append({"source": e["source"], "target": fold_map[e["target"]]})
        elif e["source"] in fold_ids:
            pass  # factor->feature edge absorbed
        else:
            new_edges.append(e)

    # --- Step 2: absorb observables that only feed a single identity feature ---
    obs_ids = {n["id"] for n in new_nodes if n["kind"] == "observable"}
    obs_outgoing: Dict[str, List[str]] = {oid: [] for oid in obs_ids}
    for e in new_edges:
        if e["source"] in obs_outgoing:
            obs_outgoing[e["source"]].append(e["target"])

    absorb_obs: set = set()
    for obs_id, targets in obs_outgoing.items():
        if len(targets) == 1 and targets[0] in identity_feat_ids:
            absorb_obs.add(obs_id)

    if absorb_obs:
        new_nodes = [n for n in new_nodes if n["id"] not in absorb_obs]
        new_edges = [e for e in new_edges if e["source"] not in absorb_obs]

    result = dict(record)
    result["nodes"] = new_nodes
    result["edges"] = new_edges
    return result


def _compute_layout(nodes: List[Dict], edges: List[Dict]) -> Dict[str, Tuple[float, float]]:
    """Horizontal layered layout: observables left, target right.

    Features that have no incoming edges (absorbed identity inputs) are
    positioned at layer 0 alongside observables.
    """
    has_incoming = {e["target"] for e in edges}

    layer_map = {"observable": 0, "factor": 1, "feature": 2, "target": 3}
    groups: Dict[int, List[str]] = {}
    for n in nodes:
        kind = n.get("kind", "observable")
        layer = layer_map.get(kind, 1)
        # Root features (no incoming edges) act as direct inputs
        if kind == "feature" and n["id"] not in has_incoming:
            layer = 0
        groups.setdefault(layer, []).append(n["id"])

    for layer in groups:
        groups[layer] = sorted(groups[layer])

    pos: Dict[str, Tuple[float, float]] = {}
    x_positions = {0: 0.0, 1: 1.6, 2: 3.2, 3: 5.0}

    for layer, nids in groups.items():
        x = x_positions.get(layer, layer * 1.5)
        n_items = len(nids)
        if n_items == 1:
            ys = [0.5]
        else:
            ys = np.linspace(0.0, 1.0, n_items).tolist()
        for nid, y_val in zip(nids, ys):
            pos[nid] = (x, y_val)
    return pos


def _draw_box(ax, cx: float, cy: float, w: float, h: float,
              label: str, fc: str, ec: str, lw: float = 1.0,
              fontsize: float = 8.0, fontweight: str = "normal",
              fontcolor: str = "#1a1a2e") -> None:
    box = patches.FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=3,
    )
    ax.add_patch(box)
    ax.text(cx, cy, label, ha="center", va="center",
            fontsize=fontsize, fontweight=fontweight, color=fontcolor, zorder=4)


def render_dag_panel(
    ax: plt.Axes,
    record: Dict[str, Any],
    prev_record: Optional[Dict[str, Any]] = None,
    title: str = "",
    show_unused_count: bool = True,
) -> None:
    # Fold identity chains for cleaner visualization
    record = _fold_identity_chains(record)
    if prev_record is not None:
        prev_record = _fold_identity_chains(prev_record)

    all_nodes = record.get("nodes", [])
    nodes, edges = _active_subgraph(record)
    if not nodes:
        ax.text(0.5, 0.5, "No DAG structure", ha="center", va="center", fontsize=10)
        ax.axis("off")
        return

    n_unused = len([n for n in all_nodes if n.get("kind") == "observable"]) - \
               len([n for n in nodes if n.get("kind") == "observable"])

    pos = _compute_layout(nodes, edges)
    node_map = {n["id"]: n for n in nodes}
    prev_ids = set()
    if prev_record:
        prev_ids = {n["id"] for n in prev_record.get("nodes", [])}

    # Fixed coordinate system so all panels have identical viewport
    ax.set_xlim(-0.8, 5.8)
    ax.set_ylim(-0.25, 1.25)
    ax.axis("off")

    # Base box dimensions per kind — scaled up for long labels
    base_dims = {
        "observable": (0.80, 0.10),
        "factor": (0.95, 0.10),
        "feature": (1.10, 0.10),
        "target": (0.70, 0.10),
    }

    def _box_params(kind: str, label: str):
        w, h = base_dims.get(kind, (0.8, 0.1))
        fs = 8.5 if kind == "target" else 7.5
        n_lines = label.count("\n") + 1
        if n_lines > 1:
            h = 0.10 * n_lines + 0.02  # taller for wrapped labels
        # Size to the longest line, capped at 1.5
        longest = max(len(line) for line in label.split("\n"))
        if longest > 18:
            w = max(w, min(1.5, 0.06 * longest))
            fs = min(fs, 6.5)
        if longest > 28:
            fs = min(fs, 5.8)
        return w, h, fs

    # Draw edges
    for edge in edges:
        src, tgt = edge.get("source", ""), edge.get("target", "")
        if src in pos and tgt in pos:
            x0, y0 = pos[src]
            x1, y1 = pos[tgt]
            kind_src = node_map.get(src, {}).get("kind", "observable")
            kind_tgt = node_map.get(tgt, {}).get("kind", "observable")
            lbl_src = _display_label(node_map.get(src, {}).get("label", ""))
            lbl_tgt = _display_label(node_map.get(tgt, {}).get("label", ""))
            w_src = _box_params(kind_src, lbl_src)[0]
            w_tgt = _box_params(kind_tgt, lbl_tgt)[0]
            ax.annotate(
                "", xy=(x1 - w_tgt / 2 - 0.02, y1), xytext=(x0 + w_src / 2 + 0.02, y0),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color="#8899aa",
                    lw=1.1,
                    connectionstyle="arc3,rad=0.08" if abs(y1 - y0) > 0.01 else "arc3,rad=0",
                ),
                zorder=1,
            )

    # Draw nodes
    for n in nodes:
        nid = n["id"]
        if nid not in pos:
            continue
        x, y = pos[nid]
        kind = n.get("kind", "observable")
        is_new = nid not in prev_ids and prev_record is not None
        fc = _KIND_COLORS.get(kind, "#eeeeee")
        ec = "#2e7d32" if is_new else _KIND_EDGE.get(kind, "#666666")
        lw_val = 2.4 if is_new else 1.2
        label = _display_label(n.get("label", nid))
        w, h, fontsize = _box_params(kind, label)
        fontweight = "bold" if (kind == "target" or is_new) else "normal"
        _draw_box(ax, x, y, w, h, label, fc, ec, lw_val, fontsize, fontweight)

    # Show count of unused observables
    if show_unused_count and n_unused > 0:
        obs_nodes = [n for n in nodes if n.get("kind") == "observable"]
        if obs_nodes:
            obs_ys = [pos[n["id"]][1] for n in obs_nodes if n["id"] in pos]
            note_y = min(obs_ys) - 0.15 if obs_ys else -0.15
            ax.text(
                pos[obs_nodes[0]["id"]][0], note_y,
                f"+ {n_unused} unused observables",
                ha="center", va="top", fontsize=6.5,
                color="#888888", fontstyle="italic",
            )

    # Legend — spread items across the full viewport width
    legend_y = -0.20
    legend_items = [
        ("observable", _KIND_COLORS["observable"]),
        ("factor", _KIND_COLORS["factor"]),
        ("feature", _KIND_COLORS["feature"]),
        ("target", _KIND_COLORS["target"]),
    ]
    legend_xs = np.linspace(-0.6, 4.0, len(legend_items))
    for i, (kind, color) in enumerate(legend_items):
        lx = legend_xs[i]
        ax.add_patch(patches.Rectangle(
            (lx, legend_y), 0.08, 0.04,
            facecolor=color, edgecolor="#999", linewidth=0.6, zorder=5,
        ))
        ax.text(lx + 0.10, legend_y + 0.02, kind,
                fontsize=5.5, va="center", color="#555", zorder=5)

    if title:
        ax.set_title(title, fontsize=10, fontweight="bold", pad=8)


# ---------------------------------------------------------------------------
# Full iteration frame
# ---------------------------------------------------------------------------

def render_iteration_frame(
    record: Dict[str, Any],
    prev_record: Optional[Dict[str, Any]],
    summary: Dict[str, Any],
    all_records: List[Dict[str, Any]],
    out_path: Path,
    dpi: int = 300,
) -> None:
    version = record.get("version", 0)
    bits = record.get("bits", {})
    equations = record.get("equation_lines", [])
    features = record.get("features", [])

    fig = plt.figure(figsize=(18, 11), dpi=dpi)
    fig.patch.set_facecolor("white")
    gs = gridspec.GridSpec(
        3, 2, figure=fig,
        width_ratios=[1.2, 1.0],
        height_ratios=[1.0, 0.55, 0.55],
        hspace=0.32, wspace=0.30,
    )

    # --- DAG panel (top right) ---
    ax_dag = fig.add_subplot(gs[0, 1])
    slice_label = record.get("revealed_slice") or "initial"
    render_dag_panel(
        ax_dag, record, prev_record,
        title=f"DAG iter {version}: {slice_label}",
    )

    # --- MDL budget (middle right) ---
    ax_bits = fig.add_subplot(gs[1, 1])
    l_models = [r["bits"]["L_model"] for r in all_records if r["version"] <= version]
    l_datas = [r["bits"]["L_data"] for r in all_records if r["version"] <= version]
    shown_iters = [r["version"] for r in all_records if r["version"] <= version]
    ax_bits.bar(shown_iters, l_models, color="#4a90a4", edgecolor="#2a5a6a", label="L_model", zorder=2)
    ax_bits.bar(shown_iters, l_datas, bottom=l_models, color="#e8944a", edgecolor="#a86020", label="L_data", zorder=2)
    ax_bits.set_title("MDL budget (bits)", fontsize=11, fontweight="bold")
    ax_bits.set_xlabel("Iteration", fontsize=9)
    ax_bits.set_ylabel("Bits", fontsize=9)
    ax_bits.legend(fontsize=8, loc="upper left", framealpha=0.9)
    ax_bits.grid(alpha=0.2, axis="y")
    ax_bits.set_xticks(shown_iters)

    # --- Equation + metrics panel (top left) ---
    ax_eq = fig.add_subplot(gs[0, 0])
    ax_eq.axis("off")
    eq_text = equations[0] if equations else "y = intercept only"
    wrapped_eq = "\n".join(textwrap.wrap(eq_text, width=60))

    feature_labels = [f.get("label", "?") for f in features]
    feature_text = ", ".join(feature_labels) if feature_labels else "none"

    info = (
        f"Iteration {version}\n"
        f"Revealed: {slice_label}\n\n"
        f"Equation:\n{wrapped_eq}\n\n"
        f"Features ({bits.get('k_features', 0)}): {feature_text}\n\n"
        f"RMSE: {bits.get('rmse', 0):.4f}    R\u00b2: {bits.get('r2', 0):.4f}\n"
        f"L_model: {bits.get('L_model', 0):.1f}    L_data: {bits.get('L_data', 0):.1f}    "
        f"L_total: {bits.get('L_total', 0):.1f} bits\n"
        f"Break: {record.get('break_detected', False)} ({record.get('break_type', 'none')}), "
        f"gain: {record.get('break_bits_gain', 0):.1f} bits\n"
        f"n = {bits.get('n', 0)} observations"
    )
    ax_eq.text(
        0.05, 0.95, info,
        transform=ax_eq.transAxes, va="top", ha="left",
        fontsize=10.5, family="monospace",
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#f7f9fc", edgecolor="#b0c4de", alpha=0.95),
    )
    ax_eq.set_title("World-model summary", fontsize=12, fontweight="bold", loc="left")

    # --- Search trace (bottom left) ---
    ax_search = fig.add_subplot(gs[2, 0])
    trace = record.get("search_trace", [])
    if trace:
        xs = [s["step"] for s in trace]
        bits_after = [s["bits_after"] if s["bits_after"] == s["bits_after"] else np.nan for s in trace]
        ax_search.plot(xs, bits_after, color="#999", linewidth=0.8, alpha=0.5, zorder=1)
        acc_xs = [s["step"] for s in trace if s["accepted"]]
        acc_ys = [s["bits_after"] for s in trace if s["accepted"]]
        rej_xs = [s["step"] for s in trace if not s["accepted"] and s["bits_after"] == s["bits_after"]]
        rej_ys = [s["bits_after"] for s in trace if not s["accepted"] and s["bits_after"] == s["bits_after"]]
        ax_search.scatter(rej_xs, rej_ys, s=12, color="#d94040", alpha=0.5, marker="x", label="rejected", zorder=2)
        ax_search.scatter(acc_xs, acc_ys, s=40, color="#2d8e2d", label="accepted", zorder=3)
        n_acc = sum(1 for s in trace if s["accepted"])
        ax_search.set_title(
            f"Hill-climb trace ({n_acc} accepted / {len(trace)} steps)",
            fontsize=11, fontweight="bold",
        )
        ax_search.set_xlabel("Inner step", fontsize=9)
        ax_search.set_ylabel("Total bits", fontsize=9)
        ax_search.legend(fontsize=8, loc="upper right")
        ax_search.grid(alpha=0.2)
    else:
        ax_search.text(0.5, 0.5, "No search trace", ha="center", va="center")
        ax_search.axis("off")

    # --- Hypothesis / rationale panel (bottom right) ---
    ax_info = fig.add_subplot(gs[2, 1])
    ax_info.axis("off")
    hypothesis = record.get("breaker_hypothesis", "") or "n/a"
    builder_diag = ""
    bh = record.get("builder_hypothesis", {})
    if isinstance(bh, dict):
        builder_diag = bh.get("diagnosis", "")
    rationale_text = (
        f"Breaker hypothesis:\n"
        f"{textwrap.fill(hypothesis[:280], width=55)}\n\n"
        f"Builder diagnosis:\n"
        f"{textwrap.fill(builder_diag[:280], width=55)}"
    )
    ax_info.text(
        0.05, 0.95, rationale_text,
        transform=ax_info.transAxes, va="top", ha="left",
        fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#fefefe", edgecolor="#d0d0d0"),
    )
    ax_info.set_title("Agent reasoning", fontsize=11, fontweight="bold", loc="left")

    fig.suptitle(
        "Breaking and Expanding the DAG World Model (MDL-scored)",
        fontsize=15, fontweight="bold", y=0.98,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
# DAG evolution overview
# ---------------------------------------------------------------------------

def render_dag_evolution(
    records: List[Dict[str, Any]],
    out_path: Path,
    dpi: int = 300,
) -> None:
    n = len(records)
    cols = min(2, n)
    rows = math.ceil(n / cols)

    fig = plt.figure(figsize=(9.0 * cols, 5.0 * rows))
    fig.patch.set_facecolor("white")
    gs = gridspec.GridSpec(rows, cols, figure=fig, hspace=0.35, wspace=0.20)

    for i, record in enumerate(records):
        r_idx = i // cols
        c_idx = i % cols
        ax = fig.add_subplot(gs[r_idx, c_idx])

        prev = records[i - 1] if i > 0 else None
        version = record.get("version", i)
        bits = record.get("bits", {})
        eq = (record.get("equation_lines") or [""])[0]
        # Show full equation, wrapped and in smaller font
        wrapped_eq = "\n".join(textwrap.wrap(eq, width=90))
        title = f"iter {version}  |  k={bits.get('k_features', '?')}  RMSE={bits.get('rmse', 0):.3f}"
        render_dag_panel(ax, record, prev, title=title)
        ax.text(
            0.5, -0.04, wrapped_eq,
            transform=ax.transAxes, ha="center", va="top",
            fontsize=6, family="monospace", color="#334155",
        )

    # Hide any leftover axes
    for j in range(n, rows * cols):
        r_idx = j // cols
        c_idx = j % cols
        ax = fig.add_subplot(gs[r_idx, c_idx])
        ax.axis("off")

    fig.suptitle("DAG World-Model Evolution", fontsize=16, fontweight="bold")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for suffix in [".svg", ".png"]:
        fig.savefig(out_path.with_suffix(suffix), dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
# MDL trajectory
# ---------------------------------------------------------------------------

def render_mdl_trajectory(
    records: List[Dict[str, Any]],
    out_path: Path,
    dpi: int = 300,
) -> None:
    iters = [r["version"] for r in records]
    l_model = [r["bits"]["L_model"] for r in records]
    l_data = [r["bits"]["L_data"] for r in records]
    l_total = [r["bits"]["L_total"] for r in records]
    rmse = [r["bits"]["rmse"] for r in records]
    r2 = [r["bits"]["r2"] for r in records]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), constrained_layout=True)
    fig.patch.set_facecolor("white")

    # --- MDL Budget panel ---
    ax = axes[0]
    ax.bar(iters, l_model, color="#4a90a4", edgecolor="#2a5a6a", label="L_model")
    ax.bar(iters, l_data, bottom=l_model, color="#e8944a", edgecolor="#a86020", label="L_data")
    ax.plot(iters, l_total, color="#1a1a2e", marker="o", linewidth=2, label="L_total", zorder=3)
    # Annotate break bit gains on the MDL panel (where they semantically belong)
    for r in records[1:]:
        gain = r.get("break_bits_gain", 0)
        if gain > 0:
            ax.annotate(
                f"+{gain:.1f}",
                xy=(r["version"], r["bits"]["L_total"]),
                xytext=(0, 10), textcoords="offset points",
                ha="center", fontsize=8, color="#555",
                clip_on=False,
            )
    ax.set_title("MDL Budget", fontsize=12, fontweight="bold")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Bits")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.2, axis="y")
    ax.set_xticks(iters)
    ax.set_ylim(0, max(l_total) * 1.12)

    # --- RMSE panel ---
    ax = axes[1]
    ax.plot(iters, rmse, marker="s", color="#2d8e2d", linewidth=2.2, markersize=8)
    ax.set_title("RMSE", fontsize=12, fontweight="bold")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("RMSE")
    ax.grid(alpha=0.2)
    ax.set_xticks(iters)

    # --- R-squared panel ---
    ax = axes[2]
    ax.plot(iters, r2, marker="D", color="#7b3fa0", linewidth=2.2, markersize=8)
    ax.set_title("R\u00b2", fontsize=12, fontweight="bold")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("R\u00b2")
    ax.grid(alpha=0.2)
    ax.set_xticks(iters)
    ax.set_ylim(0, 1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    for suffix in [".svg", ".png"]:
        fig.savefig(out_path.with_suffix(suffix), dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Discovery timeline
# ---------------------------------------------------------------------------

def render_discovery_timeline(
    records: List[Dict[str, Any]],
    summary: Dict[str, Any],
    out_path: Path,
    dpi: int = 300,
) -> None:
    stage_labels = {}
    for entry in summary.get("collection_history", []):
        stage_labels[entry["selected_slice"]] = entry.get("slice_label", entry["selected_slice"])

    fig, ax = plt.subplots(figsize=(max(10, 2.5 * len(records)), 2.8))
    fig.patch.set_facecolor("white")
    ax.axis("off")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)

    xs = np.linspace(0.08, 0.92, max(1, len(records)))
    y = 0.52

    for i, r in enumerate(records):
        x = xs[i]
        detected = r.get("break_detected", False)
        color = "#3a7ca5" if i == 0 else "#2d8e2d" if detected else "#aaaaaa"
        ax.scatter([x], [y], s=620, color=color, edgecolor="#222", linewidths=1.5, zorder=3)
        ax.text(x, y, str(r["version"]), ha="center", va="center",
                color="white", weight="bold", fontsize=12, zorder=4)
        if i > 0:
            ax.plot([xs[i - 1], x], [y, y], color="#555", linewidth=2, zorder=1)

        if r["version"] == 0:
            label = "Initial build"
        else:
            sid = r.get("revealed_slice", "")
            label = stage_labels.get(sid, sid.replace("_", " "))
        ax.text(x, y - 0.18, "\n".join(textwrap.wrap(label, width=18)),
                ha="center", va="top", fontsize=8.5)
        if r["version"] > 0:
            btype = r.get("break_type", "none")
            gain = r.get("break_bits_gain", 0)
            ax.text(x, y + 0.16, f"{btype}\n+{gain:.1f} bits",
                    ha="center", va="bottom", fontsize=8, color="#334155")

    eq = (records[-1].get("equation_lines") or [""])[0]
    ax.text(0.5, 0.0, f"Final: {eq}",
            ha="center", va="bottom",
            fontsize=8, family="monospace", color="#555",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#f0f4f8", edgecolor="#ccc"))

    ax.set_title("Breaker Collection Sequence", fontsize=14, fontweight="bold", pad=6)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    for suffix in [".svg", ".png"]:
        fig.savefig(out_path.with_suffix(suffix), dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Per-iteration hill-climb search trace
# ---------------------------------------------------------------------------

def _render_search_trace_axes(
    ax: plt.Axes,
    record: Dict[str, Any],
    numbered: bool = False,
) -> List[Tuple[int, str, float]]:
    """Draw the hill-climb trace on *ax*.  Returns accepted-step table rows.

    If *numbered* is True, each accepted dot gets a circled index [1], [2], ...
    Otherwise the plot is clean (no text on the data area).
    """
    trace = record.get("search_trace", [])
    version = record.get("version", 0)
    if not trace:
        return []

    xs = [s["step"] for s in trace]
    bits_after = [s["bits_after"] if s["bits_after"] == s["bits_after"] else np.nan for s in trace]
    ax.plot(xs, bits_after, color="#bbbbbb", linewidth=0.8, alpha=0.6, zorder=1)

    rej_xs = [s["step"] for s in trace if not s["accepted"] and s["bits_after"] == s["bits_after"]]
    rej_ys = [s["bits_after"] for s in trace if not s["accepted"] and s["bits_after"] == s["bits_after"]]
    acc_steps = [s for s in trace if s["accepted"]]
    acc_xs = [s["step"] for s in acc_steps]
    acc_ys = [s["bits_after"] for s in acc_steps]

    ax.scatter(rej_xs, rej_ys, s=18, color="#d94040", alpha=0.5, marker="x", label="rejected", zorder=2)
    ax.scatter(acc_xs, acc_ys, s=50, color="#2d8e2d", label="accepted", zorder=3)

    table_rows: List[Tuple[int, str, float]] = []
    if numbered:
        for idx, s in enumerate(acc_steps, start=1):
            ax.annotate(
                str(idx),
                xy=(s["step"], s["bits_after"]),
                xytext=(0, 0), textcoords="offset points",
                ha="center", va="center",
                fontsize=5.5, fontweight="bold", color="white", zorder=4,
            )
            table_rows.append((idx, s.get("description", ""), s["bits_after"]))

    n_acc = len(acc_xs)
    search_summary = record.get("search_summary", {})
    stop_reason = search_summary.get("stop_reason", "")

    ax.set_title(
        f"Hill-climb trace \u2014 iter {version}  ({n_acc} accepted / {len(trace)} steps)",
        fontsize=12, fontweight="bold",
    )
    ax.set_xlabel("Proposal step", fontsize=10)
    ax.set_ylabel("Total description length (bits)", fontsize=10)
    ax.legend(fontsize=9, loc="upper right", framealpha=0.9)
    ax.grid(alpha=0.2)

    return table_rows


def render_search_trace(
    record: Dict[str, Any],
    out_path: Path,
    dpi: int = 300,
) -> None:
    """Render two variants of the hill-climb search trace:

    1. ``<name>.{svg,png}``           — clean plot, no text labels on dots.
    2. ``<name>_annotated.{svg,png}`` — numbered dots [1],[2],... with a
       legend table of accepted proposals printed below the chart.
    """
    trace = record.get("search_trace", [])
    if not trace:
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # --- Variant 1: clean (no labels) ---
    fig, ax = plt.subplots(figsize=(8, 4.5))
    fig.patch.set_facecolor("white")
    _render_search_trace_axes(ax, record, numbered=False)
    for suffix in [".svg", ".png"]:
        fig.savefig(out_path.with_suffix(suffix), dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # --- Variant 2: numbered + table ---
    n_acc = sum(1 for s in trace if s["accepted"])
    # Allocate space: plot on top, table below
    table_height = max(1.2, 0.28 * n_acc)
    fig = plt.figure(figsize=(8, 4.5 + table_height))
    fig.patch.set_facecolor("white")
    gs = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[4.5, table_height], hspace=0.25)

    ax_plot = fig.add_subplot(gs[0])
    rows = _render_search_trace_axes(ax_plot, record, numbered=True)

    ax_tbl = fig.add_subplot(gs[1])
    ax_tbl.axis("off")
    if rows:
        header = f"{'#':>3}  {'Step':>5}  {'Bits':>9}  Description"
        lines = [header, "\u2500" * 70]
        for idx, (num, desc, bits_val) in enumerate(rows):
            # recover step number from trace
            acc_steps = [s for s in trace if s["accepted"]]
            step = acc_steps[idx]["step"] if idx < len(acc_steps) else "?"
            short_desc = desc if len(desc) < 52 else desc[:49] + "..."
            lines.append(f"[{num:>2}]  {step:>5}  {bits_val:>9.1f}  {short_desc}")
        table_text = "\n".join(lines)
        ax_tbl.text(
            0.02, 0.98, table_text,
            transform=ax_tbl.transAxes, va="top", ha="left",
            fontsize=7, family="monospace", color="#333",
        )

    ann_path = out_path.parent / (out_path.stem + "_annotated")
    for suffix in [".svg", ".png"]:
        fig.savefig(ann_path.with_suffix(suffix), dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Re-render high-quality figures from an existing discovery run.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("run_dir", type=str,
                   help="Path to the discovery output directory (contains run_summary.json)")
    p.add_argument("--outdir", type=str, default=None,
                   help="Output directory for figures. Defaults to <run_dir>/figures_hq/")
    p.add_argument("--dpi", type=int, default=300, help="Resolution for PNG output")
    p.add_argument("--skip-frames", action="store_true",
                   help="Skip per-iteration frame rendering")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    if not (run_dir / "run_summary.json").exists():
        print(f"Error: {run_dir / 'run_summary.json'} not found.")
        return 1

    outdir = Path(args.outdir).resolve() if args.outdir else run_dir / "figures_hq"
    outdir.mkdir(parents=True, exist_ok=True)

    summary, records = load_run(run_dir)
    print(f"Loaded {len(records)} iterations from {run_dir}")

    # DAG evolution
    print("Rendering DAG evolution...")
    render_dag_evolution(records, outdir / "dag_evolution", dpi=args.dpi)

    # MDL trajectory
    print("Rendering MDL trajectory...")
    render_mdl_trajectory(records, outdir / "mdl_trajectory", dpi=args.dpi)

    # Discovery timeline
    print("Rendering discovery timeline...")
    render_discovery_timeline(records, summary, outdir / "discovery_timeline", dpi=args.dpi)

    # Per-iteration search traces (SVG + PNG)
    for record in records:
        v = record["version"]
        trace_path = outdir / f"search_trace_iter_{v:02d}"
        print(f"Rendering search trace iter {v}...")
        render_search_trace(record, trace_path, dpi=args.dpi)

    # Per-iteration frames
    if not args.skip_frames:
        for i, record in enumerate(records):
            prev = records[i - 1] if i > 0 else None
            frame_path = outdir / f"frame_iter_{record['version']:02d}.png"
            print(f"Rendering frame iter {record['version']}...")
            render_iteration_frame(record, prev, summary, records, frame_path, dpi=args.dpi)

        # GIF
        try:
            from PIL import Image
            frames = sorted(outdir.glob("frame_iter_*.png"))
            if frames:
                images = [Image.open(f) for f in frames]
                gif_path = outdir / "evolution.gif"
                images[0].save(gif_path, save_all=True, append_images=images[1:],
                               duration=1200, loop=0)
                print(f"Wrote {gif_path}")
        except ImportError:
            pass

    print(f"\nAll figures written to: {outdir}")
    for f in sorted(outdir.iterdir()):
        print(f"  {f.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
