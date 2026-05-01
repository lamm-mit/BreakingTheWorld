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
    "gnm_fluct_log_z": "GNM log-fluct.",
    "contact_degree_z": "contact deg. z",
    "contact_degree_raw": "contact deg.",
    "terminal_exposure": "terminal exp.",
    "mode1_abs_z": "|mode1| z",
    "hinge_score_z": "hinge score z",
    "chain_break_proximity": "chain break",
    "res_index_norm": "res. index",
    "is_hydrophobic": "is hydrophobic",
    "bfactor_z": "B-factor z",
}


def _display_label(label: str) -> str:
    return _LABEL_MAP.get(label, label)


def _active_subgraph(record: Dict[str, Any]) -> Tuple[List[Dict], List[Dict]]:
    """Return only nodes/edges that participate in the active DAG.

    Removes observable nodes that have no outgoing edges (unused variables).
    """
    nodes = record.get("nodes", [])
    edges = record.get("edges", [])
    # Find nodes that participate in at least one edge
    active_ids = set()
    for e in edges:
        active_ids.add(e["source"])
        active_ids.add(e["target"])
    # Keep all non-observable nodes, plus observables that are used
    active_nodes = [
        n for n in nodes
        if n["id"] in active_ids or n.get("kind") not in ("observable",)
    ]
    # Actually, only keep nodes that are reachable
    active_node_ids = {n["id"] for n in active_nodes}
    active_edges = [e for e in edges if e["source"] in active_node_ids and e["target"] in active_node_ids]
    return active_nodes, active_edges


def _compute_layout(nodes: List[Dict], edges: List[Dict]) -> Dict[str, Tuple[float, float]]:
    """Horizontal layered layout: observables left, target right."""
    layer_map = {"observable": 0, "factor": 1, "feature": 2, "target": 3}
    groups: Dict[int, List[str]] = {}
    node_kind = {}
    for n in nodes:
        kind = n.get("kind", "observable")
        layer = layer_map.get(kind, 1)
        groups.setdefault(layer, []).append(n["id"])
        node_kind[n["id"]] = kind

    # Sort within each layer for stability
    for layer in groups:
        groups[layer] = sorted(groups[layer])

    pos: Dict[str, Tuple[float, float]] = {}
    x_positions = {0: 0.0, 1: 1.5, 2: 3.0, 3: 4.5}

    for layer, nids in groups.items():
        x = x_positions.get(layer, layer * 1.5)
        n = len(nids)
        if n == 1:
            ys = [0.5]
        else:
            ys = np.linspace(0.0, 1.0, n).tolist()
        for nid, y in zip(nids, ys):
            pos[nid] = (x, y)
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

    # Determine axis limits
    all_x = [p[0] for p in pos.values()]
    all_y = [p[1] for p in pos.values()]
    x_margin = 0.8
    y_margin = 0.18
    ax.set_xlim(min(all_x) - x_margin, max(all_x) + x_margin)
    ax.set_ylim(min(all_y) - y_margin, max(all_y) + y_margin)
    ax.set_aspect("equal")
    ax.axis("off")

    # Box dimensions per kind
    box_dims = {
        "observable": (0.80, 0.10),
        "factor": (0.90, 0.10),
        "feature": (1.00, 0.10),
        "target": (0.70, 0.10),
    }

    # Draw edges
    for edge in edges:
        src, tgt = edge.get("source", ""), edge.get("target", "")
        if src in pos and tgt in pos:
            x0, y0 = pos[src]
            x1, y1 = pos[tgt]
            kind_src = node_map.get(src, {}).get("kind", "observable")
            kind_tgt = node_map.get(tgt, {}).get("kind", "observable")
            w_src = box_dims.get(kind_src, (0.8, 0.1))[0]
            w_tgt = box_dims.get(kind_tgt, (0.8, 0.1))[0]
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
        lw = 2.4 if is_new else 1.2
        w, h = box_dims.get(kind, (0.8, 0.1))
        label = _display_label(n.get("label", nid))
        fontsize = 8.5 if kind == "target" else 7.5
        fontweight = "bold" if (kind == "target" or is_new) else "normal"
        _draw_box(ax, x, y, w, h, label, fc, ec, lw, fontsize, fontweight)

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

    # Legend
    legend_x = min(all_x) - x_margin + 0.1
    legend_y = min(all_y) - y_margin + 0.03
    for i, (kind, color) in enumerate([
        ("observable", _KIND_COLORS["observable"]),
        ("factor", _KIND_COLORS["factor"]),
        ("feature", _KIND_COLORS["feature"]),
        ("target", _KIND_COLORS["target"]),
    ]):
        ax.add_patch(patches.Rectangle(
            (legend_x + i * 0.55, legend_y), 0.08, 0.04,
            facecolor=color, edgecolor="#999", linewidth=0.6, zorder=5,
        ))
        ax.text(legend_x + i * 0.55 + 0.10, legend_y + 0.02, kind,
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

    fig, axes = plt.subplots(
        rows, cols,
        figsize=(9.0 * cols, 4.5 * rows),
    )
    fig.patch.set_facecolor("white")
    axes = np.asarray(axes).reshape(-1)

    for i, (ax, record) in enumerate(zip(axes, records)):
        prev = records[i - 1] if i > 0 else None
        version = record.get("version", i)
        bits = record.get("bits", {})
        eq = (record.get("equation_lines") or [""])[0]
        short_eq = eq if len(eq) < 55 else eq[:52] + "..."
        title = f"iter {version}  |  k={bits.get('k_features', '?')}  RMSE={bits.get('rmse', 0):.3f}"
        render_dag_panel(ax, record, prev, title=title)
        ax.text(
            0.5, -0.02, short_eq,
            transform=ax.transAxes, ha="center", va="top",
            fontsize=7, family="monospace", color="#334155",
        )

    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle("DAG World-Model Evolution", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
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

    ax = axes[0]
    ax.bar(iters, l_model, color="#4a90a4", edgecolor="#2a5a6a", label="L_model")
    ax.bar(iters, l_data, bottom=l_model, color="#e8944a", edgecolor="#a86020", label="L_data")
    ax.plot(iters, l_total, color="#1a1a2e", marker="o", linewidth=2, label="L_total", zorder=3)
    ax.set_title("MDL Budget", fontsize=12, fontweight="bold")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Bits")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.2, axis="y")
    ax.set_xticks(iters)

    ax = axes[1]
    ax.plot(iters, rmse, marker="s", color="#2d8e2d", linewidth=2.2, markersize=8)
    for r in records[1:]:
        ax.annotate(
            f"{r.get('break_bits_gain', 0):+.1f}b",
            xy=(r["version"], r["bits"]["rmse"]),
            xytext=(0, 10), textcoords="offset points",
            ha="center", fontsize=8, color="#555",
        )
    ax.set_title("RMSE", fontsize=12, fontweight="bold")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("RMSE")
    ax.grid(alpha=0.2)
    ax.set_xticks(iters)

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
            ax.text(x, y + 0.16, f"{btype}\n{gain:+.1f} bits",
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
