#!/usr/bin/env python3
"""
Render a publication slide showing the 8 proteins across 4 discovery
stages, with C-alpha backbone traces colored by experimental B-factor.

Produces: protein_stages_slide.{svg,png}

Usage:
    python src/render_protein_slide.py
    python src/render_protein_slide.py --outdir figures --dpi 300
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import numpy as np

# Add src/ to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent))
from protein_nma_oracle import (
    PDBSpec, parse_pdb_spec, parse_ca_residues, gnm_modes, zscore,
    DEFAULT_PROTEIN_STAGES, DEFAULT_STAGE_LABELS,
)


CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "pdb_cache"

# Stage metadata for the slide
STAGE_INFO = {
    "stage_0_compact_single_domain": {
        "short": "Stage 0: Compact",
        "color": "#3a7ca5",
        "desc": "Initial build",
        "equation": "bfactor_z = β · gnm_fluct_z",
        "metrics": "R² = 0.48, 236 bits",
        "break": None,
    },
    "stage_1_terminal_flexibility": {
        "short": "Stage 1: Terminal flex.",
        "color": "#2d8e2d",
        "desc": "Regime split",
        "equation": "+ terminal_exp × ReLU(gnm_fluct_z)\n+ mode1_abs_z",
        "metrics": "R² = 0.68, +9.0 bits gain",
        "break": "regime_split",
    },
    "stage_2_hinge_domain_motion": {
        "short": "Stage 2: Hinge motion",
        "color": "#c44e20",
        "desc": "Ontology break",
        "equation": "→ gnm_fluct_z + mode1_abs_z\n(terminal term retracted)",
        "metrics": "R² = 0.54, +37.3 bits gain",
        "break": "ontology_break",
    },
    "stage_3_validation_mixed": {
        "short": "Stage 3: Validation",
        "color": "#7b3fa0",
        "desc": "Regime split",
        "equation": "→ gnm_fluct_log_z\n  × ReLU(mode1_abs_z)",
        "metrics": "R² = 0.41, +54.3 bits gain",
        "break": "regime_split",
    },
}


def load_chain(spec_str: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Parse a PDB chain and return coords, bfactors, gnm_fluct, n_residues."""
    spec = parse_pdb_spec(spec_str)
    pdb_path = CACHE_DIR / f"{spec.pdb_id}.pdb"
    residues = parse_ca_residues(pdb_path, spec)
    coords = np.vstack([r.coord for r in residues])
    bfactors = np.array([r.bfactor for r in residues], dtype=float)
    contact, eigvals, eigvecs = gnm_modes(coords, cutoff=10.0, n_modes=20)
    inv_eigs = 1.0 / eigvals
    fluctuations = np.sum((eigvecs * eigvecs) * inv_eigs[None, :], axis=1)
    bfactor_z = zscore(bfactors)
    gnm_z = zscore(fluctuations)
    corr = float(np.corrcoef(gnm_z, bfactor_z)[0, 1])
    return coords, bfactor_z, gnm_z, len(residues), corr


def _optimal_2d_projection(coords: np.ndarray) -> np.ndarray:
    """Project 3D coords to 2D via PCA for the best visual spread."""
    centered = coords - coords.mean(axis=0)
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    proj = centered @ Vt[:2].T
    return proj


def draw_protein(ax: plt.Axes, coords: np.ndarray, colors: np.ndarray,
                 label: str, n_res: int, corr: float) -> None:
    """Draw a single protein backbone on the given axes."""
    proj = _optimal_2d_projection(coords)

    # Normalize for consistent scaling
    span = max(proj.max(axis=0) - proj.min(axis=0))
    if span > 0:
        proj = (proj - proj.mean(axis=0)) / span

    # Color by B-factor z-score
    norm = mcolors.TwoSlopeNorm(vmin=-2, vcenter=0, vmax=2)
    cmap = cm.coolwarm

    # Draw backbone trace
    for i in range(len(proj) - 1):
        c = cmap(norm(colors[i]))
        ax.plot(proj[i:i+2, 0], proj[i:i+2, 1],
                color=c, linewidth=1.8, solid_capstyle="round", zorder=1)

    # Draw residue dots
    scatter_colors = cmap(norm(colors))
    ax.scatter(proj[:, 0], proj[:, 1], c=scatter_colors,
               s=12, edgecolors="none", zorder=2)

    # Highlight termini
    ax.scatter(proj[0, 0], proj[0, 1], s=50, c="black", marker="^",
               zorder=3, label="N-term")
    ax.scatter(proj[-1, 0], proj[-1, 1], s=50, c="black", marker="v",
               zorder=3, label="C-term")

    ax.set_xlim(-0.6, 0.6)
    ax.set_ylim(-0.6, 0.6)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"{label}  ({n_res} res, r = {corr:.2f})",
                 fontsize=8.5, fontweight="bold", pad=4)


def render_slide(outdir: Path, dpi: int = 300) -> None:
    stage_order = list(DEFAULT_PROTEIN_STAGES.keys())

    fig = plt.figure(figsize=(16, 12))
    fig.patch.set_facecolor("white")

    # Layout: 4 columns (stages), 3 rows (2 proteins + info)
    outer_gs = gridspec.GridSpec(
        4, 4, figure=fig,
        height_ratios=[0.12, 1.0, 1.0, 0.35],
        hspace=0.25, wspace=0.15,
    )

    # Stage header row
    stage_colors = []
    for col, stage_id in enumerate(stage_order):
        info = STAGE_INFO[stage_id]
        stage_colors.append(info["color"])
        ax_hdr = fig.add_subplot(outer_gs[0, col])
        ax_hdr.axis("off")
        ax_hdr.set_xlim(0, 1)
        ax_hdr.set_ylim(0, 1)
        # Stage box
        rect = patches.FancyBboxPatch(
            (0.05, 0.1), 0.9, 0.8,
            boxstyle="round,pad=0.05,rounding_size=0.1",
            facecolor=info["color"], edgecolor="none", alpha=0.85,
        )
        ax_hdr.add_patch(rect)
        ax_hdr.text(0.5, 0.55, info["short"],
                    ha="center", va="center", fontsize=10,
                    fontweight="bold", color="white")

    # Protein rows
    for col, stage_id in enumerate(stage_order):
        specs = DEFAULT_PROTEIN_STAGES[stage_id]
        for row, spec_str in enumerate(specs):
            ax = fig.add_subplot(outer_gs[1 + row, col])
            spec = parse_pdb_spec(spec_str)
            coords, bfz, gnm_z, n_res, corr = load_chain(spec_str)
            label = f"{spec.pdb_id.upper()}:{spec.chain}"
            draw_protein(ax, coords, bfz, label, n_res, corr)

    # Info row: equation / metrics / break type
    for col, stage_id in enumerate(stage_order):
        info = STAGE_INFO[stage_id]
        ax_info = fig.add_subplot(outer_gs[3, col])
        ax_info.axis("off")
        ax_info.set_xlim(0, 1)
        ax_info.set_ylim(0, 1)

        # Background box
        rect = patches.FancyBboxPatch(
            (0.03, 0.05), 0.94, 0.9,
            boxstyle="round,pad=0.04,rounding_size=0.06",
            facecolor="#f5f7fa", edgecolor="#c0c8d4", linewidth=0.8,
        )
        ax_info.add_patch(rect)

        break_label = info["desc"]
        break_color = info["color"]

        text = f"{break_label}\n{info['metrics']}\n{info['equation']}"
        ax_info.text(0.5, 0.55, text,
                    ha="center", va="center", fontsize=7,
                    family="monospace", color="#333",
                    linespacing=1.4)

    # Progression arrows between columns
    for col in range(3):
        # Arrow from column col to col+1, between the protein rows
        x_start = (col + 1) / 4 - 0.005
        x_end = (col + 1) / 4 + 0.005
        fig.patches.append(patches.FancyArrowPatch(
            (x_start, 0.53), (x_end, 0.53),
            transform=fig.transFigure,
            arrowstyle="-|>,head_width=8,head_length=6",
            color="#555", linewidth=2, zorder=10,
            mutation_scale=1,
        ))

    # Colorbar for B-factor z-score
    cbar_ax = fig.add_axes([0.35, 0.02, 0.3, 0.015])
    norm = mcolors.TwoSlopeNorm(vmin=-2, vcenter=0, vmax=2)
    sm = cm.ScalarMappable(cmap=cm.coolwarm, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    cbar.set_label("B-factor z-score (blue = rigid, red = flexible)",
                   fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    fig.suptitle(
        "Staged Protein Discovery: Progressive Revelation of Structural Complexity",
        fontsize=15, fontweight="bold", y=0.98,
    )

    # Subtitle
    fig.text(0.5, 0.955,
             "C-alpha backbone traces colored by experimental B-factor z-score  |  "
             "r = Pearson correlation between GNM fluctuation and B-factor",
             ha="center", fontsize=8.5, color="#555")

    outdir.mkdir(parents=True, exist_ok=True)
    for suffix in [".svg", ".png"]:
        fig.savefig(outdir / f"protein_stages_slide{suffix}",
                    dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote protein_stages_slide.{{svg,png}} to {outdir}")


def main():
    p = argparse.ArgumentParser(description="Render protein discovery stages slide.")
    p.add_argument("--outdir", type=str, default="runs/protein_flex_llm_deep/discovery/figures_hq")
    p.add_argument("--dpi", type=int, default=300)
    args = p.parse_args()
    render_slide(Path(args.outdir).resolve(), dpi=args.dpi)


if __name__ == "__main__":
    main()
