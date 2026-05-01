#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from discovery_data import write_dataset_json, write_observations_csv
from protein_nma_oracle import (
    PDBSpec,
    build_protein_flex_dataset,
    compute_chain_features,
    default_stage_specs,
    fetch_pdb,
    parse_stage_spec,
    stage_counts,
)


def collect_stage_specs(stage_args: Sequence[str]) -> Dict[str, List[PDBSpec]]:
    if not stage_args:
        return default_stage_specs()
    out: Dict[str, List[PDBSpec]] = {}
    for item in stage_args:
        stage_id, specs = parse_stage_spec(item)
        out[stage_id] = specs
    return out


def add_dataset_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--outdir", type=str, default="runs/protein_flex")
    p.add_argument("--pdb-cache", type=str, default="data/pdb_cache")
    p.add_argument(
        "--stage",
        action="append",
        default=[],
        help=(
            "Stage definition as stage_id=pdb:chain,pdb:chain. May be repeated. "
            "If omitted, a small curated default set is used."
        ),
    )
    p.add_argument(
        "--initial-stage",
        action="append",
        default=[],
        help="Stage id initially revealed. Defaults to the first stage.",
    )
    p.add_argument("--cutoff", type=float, default=10.0)
    p.add_argument("--n-modes", type=int, default=20)
    p.add_argument("--terminal-window", type=int, default=5)
    p.add_argument("--min-residues", type=int, default=20)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Protein normal-mode world-model breaker. Builds a real PDB/GNM "
            "flexibility dataset and can run the existing DAG+MDL discovery loop."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = p.add_subparsers(dest="command", required=True)

    p_build = sub.add_parser("build", help="Build only the staged protein dataset")
    add_dataset_args(p_build)
    p_build.add_argument("--json-name", type=str, default="dataset.json")
    p_build.add_argument("--csv-name", type=str, default="observations.csv")

    p_run = sub.add_parser("run", help="Build the dataset, then run DAG+MDL discovery")
    add_dataset_args(p_run)
    p_run.add_argument("--dataset-subdir", type=str, default="dataset")
    p_run.add_argument("--discovery-subdir", type=str, default="discovery")
    p_run.add_argument("--rounds", type=int, default=3)
    p_run.add_argument("--search-steps", type=int, default=260)
    p_run.add_argument("--search-patience", type=int, default=35)
    p_run.add_argument("--search-restarts", type=int, default=6)
    p_run.add_argument("--seed", type=int, default=7)
    p_run.add_argument("--llm-builder", action="store_true")
    p_run.add_argument("--model", type=str, default="gpt-5.5")
    p_run.add_argument("--reasoning-effort", type=str, default="medium", choices=["low", "medium", "high"])
    p_run.add_argument("--no-llm", action="store_true", help="Force deterministic/no-LLM discovery")
    p_run.add_argument("--quiet", action="store_true")

    return p.parse_args(argv)


def build_dataset_from_args(args: argparse.Namespace, dataset_dir: Path,
                            json_name: str = "dataset.json",
                            csv_name: str = "observations.csv"):
    stage_specs = collect_stage_specs(args.stage)
    dataset = build_protein_flex_dataset(
        stage_specs=stage_specs,
        cache_dir=Path(args.pdb_cache).resolve(),
        cutoff=args.cutoff,
        n_modes=args.n_modes,
        terminal_window=args.terminal_window,
        min_residues=args.min_residues,
        initial_stage_ids=args.initial_stage or None,
    )
    dataset_dir.mkdir(parents=True, exist_ok=True)
    json_path = dataset_dir / json_name
    csv_path = dataset_dir / csv_name
    write_dataset_json(dataset, json_path)
    write_observations_csv(dataset, csv_path)
    return dataset, json_path, csv_path, stage_specs


def write_dataset_readme(dataset_dir: Path, dataset, json_path: Path, csv_path: Path) -> None:
    counts = stage_counts(dataset)
    (dataset_dir / "README.md").write_text(
        "\n".join([
            "# Protein Normal-Mode Discovery Dataset",
            "",
            dataset.system_description,
            "",
            "Stages:",
            *[f"- `{stage}`: {counts.get(stage, 0)} residue observations" for stage in dataset.stage_order],
            "",
            "Files:",
            f"- `{json_path.name}`",
            f"- `{csv_path.name}`",
        ]),
        encoding="utf-8",
    )


def _feature_interpretation(labels: Sequence[str]) -> List[str]:
    lines: List[str] = []
    if "gnm_fluct_z" in labels:
        lines.append(
            "`gnm_fluct_z` is the direct GNM prediction: the residue is flexible "
            "because the C-alpha elastic network says it has a large mean-square fluctuation."
        )
    if "gnm_fluct_log_z" in labels:
        lines.append(
            "`gnm_fluct_log_z` is a compressed GNM scale. It keeps the normal-mode "
            "mechanics but reduces domination by extreme fluctuation outliers."
        )
    if "terminal_exposure" in labels:
        lines.append(
            "`terminal_exposure` says the original elastic-network ontology was incomplete: "
            "chain-boundary position explains B-factor beyond the contact-network fluctuation."
        )
    if "mode1_abs_z" in labels:
        lines.append(
            "`mode1_abs_z` adds global slow-mode participation. This points to collective "
            "domain-scale motion rather than only local packing."
        )
    if "mode1_abs_z^2" in labels:
        lines.append(
            "`mode1_abs_z^2` treats positive and negative first-mode amplitudes symmetrically, "
            "so magnitude of participation in the slowest motion is what matters."
        )
    if any("contact_degree" in label for label in labels):
        lines.append(
            "Contact-degree terms indicate a local packing explanation: fewer contacts usually "
            "mean weaker mechanical constraint."
        )
    if not lines:
        lines.append(
            "The retained DAG features should be read as the current compact hypothesis for "
            "normalized residue B-factor."
        )
    return lines


def _markdown_table(rows: Sequence[Dict[str, Any]], columns: Sequence[str]) -> List[str]:
    out = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        out.append("| " + " | ".join(str(row.get(col, "")) for col in columns) + " |")
    return out


def _safe_corr(a, b) -> float:
    import numpy as np

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) < 3 or float(np.std(a)) < 1e-12 or float(np.std(b)) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _read_pdb_resolution(path: Path) -> Optional[float]:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("REMARK   2 RESOLUTION."):
                parts = line.replace(".", " ").split()
                for token in parts:
                    try:
                        value = float(token)
                    except ValueError:
                        continue
                    if 0.2 <= value <= 20.0:
                        return value
    return None


def _sanitize_name(text: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in text).strip("_")


def _set_equal_3d_axes(ax, coords) -> None:
    import numpy as np

    lo = np.min(coords, axis=0)
    hi = np.max(coords, axis=0)
    center = 0.5 * (lo + hi)
    radius = 0.55 * float(np.max(hi - lo))
    radius = max(radius, 1.0)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.grid(False)


def _plot_chain_feature_pair(features, out_path: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    coords = np.vstack([r.coord for r in features.residues])
    gnm = np.asarray([row["gnm_fluct_z"] for row in features.observables], dtype=float)
    bfac = features.target_z
    fig = plt.figure(figsize=(10.5, 4.6), constrained_layout=True)
    panels = [
        (bfac, "Experimental B-factor z"),
        (gnm, "GNM fluctuation z"),
    ]
    for i, (values, panel_title) in enumerate(panels, start=1):
        ax = fig.add_subplot(1, 2, i, projection="3d")
        ax.plot(coords[:, 0], coords[:, 1], coords[:, 2], color="#333333", alpha=0.35, linewidth=1.0)
        vmax = max(1.0, float(np.nanmax(np.abs(values))))
        scatter = ax.scatter(
            coords[:, 0], coords[:, 1], coords[:, 2],
            c=values, cmap="coolwarm", vmin=-vmax, vmax=vmax,
            s=28, edgecolor="#222222", linewidth=0.2,
        )
        ax.set_title(panel_title, fontsize=10)
        _set_equal_3d_axes(ax, coords)
        fig.colorbar(scatter, ax=ax, shrink=0.72, pad=0.02)
    fig.suptitle(title, fontsize=12)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _project_coords_2d(coords):
    import numpy as np

    centered = coords - np.mean(coords, axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    return centered @ vt[:2].T


def _plot_chain_thumbnail(features, out_path: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    coords = np.vstack([r.coord for r in features.residues])
    xy = _project_coords_2d(coords)
    gnm = np.asarray([row["gnm_fluct_z"] for row in features.observables], dtype=float)
    bfac = features.target_z
    fig, axes = plt.subplots(1, 2, figsize=(5.6, 2.35), constrained_layout=True)
    for ax, values, label in [
        (axes[0], bfac, "B-factor z"),
        (axes[1], gnm, "GNM z"),
    ]:
        vmax = max(1.0, float(np.nanmax(np.abs(values))))
        ax.plot(xy[:, 0], xy[:, 1], color="#334155", alpha=0.35, linewidth=1.0, zorder=1)
        scatter = ax.scatter(
            xy[:, 0], xy[:, 1],
            c=values,
            cmap="coolwarm",
            vmin=-vmax,
            vmax=vmax,
            s=18,
            edgecolor="#1f2937",
            linewidth=0.18,
            zorder=2,
        )
        ax.set_title(label, fontsize=8)
        ax.set_aspect("equal", adjustable="box")
        ax.axis("off")
        fig.colorbar(scatter, ax=ax, fraction=0.055, pad=0.01)
    fig.suptitle(title, fontsize=9, weight="bold")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def generate_protein_report_assets(
    stage_specs: Dict[str, List[PDBSpec]],
    cache_dir: Path,
    assets_dir: Path,
    cutoff: float,
    n_modes: int,
    terminal_window: int,
    min_residues: int,
) -> Dict[str, Any]:
    import numpy as np

    figures_dir = assets_dir / "protein_figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    proteins: List[Dict[str, Any]] = []
    for stage_id, specs in stage_specs.items():
        for spec in specs:
            try:
                pdb_path = fetch_pdb(spec, cache_dir=cache_dir)
                features = compute_chain_features(
                    spec=spec,
                    pdb_path=pdb_path,
                    cutoff=cutoff,
                    n_modes=n_modes,
                    terminal_window=terminal_window,
                    min_residues=min_residues,
                )
            except Exception as exc:
                proteins.append({
                    "stage_id": stage_id,
                    "spec": spec.label,
                    "error": str(exc),
                })
                continue
            label = f"{features.spec.pdb_id.upper()} chain {features.residues[0].chain}"
            safe = _sanitize_name(features.spec.label)
            image_path = figures_dir / f"{safe}_bfactor_vs_gnm.png"
            thumbnail_path = figures_dir / f"{safe}_thumbnail.png"
            _plot_chain_feature_pair(features, image_path, label)
            _plot_chain_thumbnail(features, thumbnail_path, label)
            gnm = np.asarray([row["gnm_fluct_z"] for row in features.observables], dtype=float)
            terminal = np.asarray([row["is_terminal"] for row in features.observables], dtype=float)
            top_idx = np.argsort(features.target_z)[-5:][::-1]
            proteins.append({
                "stage_id": stage_id,
                "stage_label": stage_id.replace("_", " ").title(),
                "spec": spec.label,
                "pdb_id": features.spec.pdb_id.upper(),
                "chain": features.residues[0].chain,
                "n_residues": len(features.residues),
                "resolution_angstrom": _read_pdb_resolution(pdb_path),
                "gnm_bfactor_corr": _safe_corr(gnm, features.target_z),
                "terminal_fraction": float(np.mean(terminal)),
                "bfactor_z_min": float(np.min(features.target_z)),
                "bfactor_z_max": float(np.max(features.target_z)),
                "gnm_fluct_z_min": float(np.min(gnm)),
                "gnm_fluct_z_max": float(np.max(gnm)),
                "image": str(image_path.relative_to(assets_dir.parent)),
                "thumbnail": str(thumbnail_path.relative_to(assets_dir.parent)),
                "top_bfactor_residues": [
                    {
                        "residue": features.residues[int(i)].residue_id,
                        "bfactor_z": float(features.target_z[int(i)]),
                        "gnm_fluct_z": float(gnm[int(i)]),
                    }
                    for i in top_idx
                ],
            })
    metadata = {
        "asset_dir": str(assets_dir),
        "cutoff": cutoff,
        "n_modes": n_modes,
        "terminal_window": terminal_window,
        "proteins": proteins,
    }
    assets_dir.mkdir(parents=True, exist_ok=True)
    (assets_dir / "protein_report_assets.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return metadata


def write_attachable_protein_report(discovery_dir: Path) -> Path:
    summary_path = discovery_dir / "run_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    history = summary.get("history", [])
    iterations = summary.get("iterations", [])
    dataset_analysis = summary.get("dataset_analysis", {})

    lines: List[str] = [
        "# Protein World-Model Building and Breaking Report",
        "",
        "## Executive Summary",
        "",
        (
            "This run demonstrates a physically grounded world-model discovery loop on real "
            "protein structures. The system downloads PDB structures, represents each residue "
            "as a node in a C-alpha contact network, computes Gaussian Network Model normal-mode "
            "features, and tries to explain experimental crystallographic B-factors with a compact "
            "DAG equation. New protein regimes are revealed stage by stage. Whenever the current "
            "equation fails, the Builder searches for a revised DAG, and the revision is accepted "
            "only when it lowers Minimum Description Length (MDL)."
        ),
        "",
        "The run should be interpreted as a controlled demonstration of discovery mechanics, not "
        "as a claim of a new protein-physics law. It rediscovers known biophysical structure: "
        "elastic-network fluctuations explain much of residue flexibility, termini and boundary "
        "effects can add explanatory power, and slow collective modes matter in hinge/domain-like "
        "proteins.",
        "",
        "## Dataset and Physics",
        "",
        summary.get("system_description", ""),
        "",
        "The target variable is `bfactor_z`: the C-alpha crystallographic B-factor normalized "
        "within each protein chain. This normalization makes the task focus on residue-to-residue "
        "flexibility patterns rather than absolute differences in crystallographic conditions.",
        "",
        "The central physics feature is `gnm_fluct_z`, derived from the GNM pseudoinverse of the "
        "C-alpha Kirchhoff/contact matrix. Other features encode boundary position, local contact "
        "density, first-mode participation, hinge-like curvature, and residue class.",
        "",
    ]

    if dataset_analysis.get("protocol_logic"):
        lines.extend(["## Staged Protocol", ""])
        for item in dataset_analysis["protocol_logic"]:
            lines.append(f"- {item}")
        lines.append("")

    metric_rows: List[Dict[str, Any]] = []
    for row in iterations:
        metric_rows.append({
            "iter": row.get("iteration"),
            "slice": row.get("revealed_slice"),
            "features": row.get("k_features"),
            "RMSE": row.get("rmse"),
            "R2": row.get("r2"),
            "L_total": row.get("L_total_bits"),
            "break": row.get("break_detected"),
            "gain_bits": row.get("break_bits_gain"),
        })
    lines.extend(["## Quantitative Trajectory", ""])
    lines.extend(_markdown_table(metric_rows, ["iter", "slice", "features", "RMSE", "R2", "L_total", "break", "gain_bits"]))
    lines.append("")

    lines.extend(["## Iteration-by-Iteration Interpretation", ""])
    for record in history:
        version = record.get("version")
        revealed = record.get("revealed_slice") or "initial compact-protein stage"
        bits = record.get("bits", {})
        features = record.get("features", [])
        labels = [f.get("label", "") for f in features]
        equations = record.get("equation_lines", [])
        lines.extend([
            f"### Iteration {version}: {revealed}",
            "",
            f"Equation: `{equations[0] if equations else 'n/a'}`",
            "",
            (
                f"Fit: RMSE {bits.get('rmse', float('nan')):.4f}, "
                f"R2 {bits.get('r2', float('nan')):.4f}, "
                f"L_total {bits.get('L_total', float('nan')):.1f} bits."
            ),
            "",
            "Accepted DAG features: " + (", ".join(f"`{label}`" for label in labels) if labels else "none"),
            "",
        ])
        if record.get("revealed_slice"):
            lines.append(
                f"Break status: `{record.get('break_type')}`, detected={record.get('break_detected')}, "
                f"MDL gain={record.get('break_bits_gain', 0.0):.1f} bits."
            )
            lines.append("")
        for item in _feature_interpretation(labels):
            lines.append(f"- {item}")
        lines.append("")
        if version == 0:
            lines.append(
                "Interpretation: the initial world model is the classical contact-network "
                "mechanics hypothesis. B-factor variation is explained primarily by predicted "
                "GNM fluctuation."
            )
        elif "terminal_exposure" in labels:
            lines.append(
                "Interpretation: the first break adds a boundary ontology. The model no longer "
                "claims that a uniform elastic network fully explains flexibility; chain-end "
                "position is a separate explanatory factor."
            )
        elif "mode1_abs_z" in labels or "mode1_abs_z^2" in labels:
            lines.append(
                "Interpretation: the model shifts toward collective-motion ontology. The key "
                "quantity is not just total GNM fluctuation, but participation in the slowest "
                "global mode."
            )
        elif "gnm_fluct_log_z" in labels:
            lines.append(
                "Interpretation: the model keeps the normal-mode explanation but changes the "
                "scale, suggesting that raw GNM fluctuations are too heavy-tailed across the "
                "mixed validation set."
            )
        lines.append("")

    lines.extend([
        "## What Was Discovered",
        "",
        "The system did not discover a new protein law. It rediscovered a known hierarchy of "
        "protein-flexibility explanations in an automated, falsifiable loop:",
        "",
        "1. Contact-network normal modes are a useful first world model for residue B-factors.",
        "2. Boundary/terminus effects can break that model and require an explicit feature.",
        "3. Hinge/domain-like regimes can favor slow-mode participation features.",
        "4. Across a mixed set, a compressed normal-mode fluctuation scale can be more robust than raw fluctuation.",
        "",
        "The important result is methodological: the system builds a compact physical model, "
        "tests it against new regimes, and revises the ontology only when MDL says the new "
        "structure explains enough data to justify its complexity.",
        "",
        "## Limitations",
        "",
        "- B-factors are affected by crystallographic resolution, refinement, crystal contacts, and disorder, not only intrinsic dynamics.",
        "- The current PDB set is intentionally small and curated for demonstration.",
        "- The no-LLM run uses deterministic feature seeds; an LLM-assisted run should provide richer hypotheses, but MDL should remain the judge.",
        "- This is not yet a claim of novel biology. A publishable claim would need a larger benchmark, careful train/test splits, stronger baselines, and case-by-case structural validation.",
        "",
        "## How To Read The Output",
        "",
        "- `report.md` contains the raw run narrative and equations.",
        "- `metrics.csv` contains the quantitative trajectory.",
        "- `world_model_iter_*.json` contains machine-readable DAG snapshots.",
        "- `paper_figures/*.svg` and `evolution.gif` show the model evolution visually.",
    ])

    out = discovery_dir / "protein_world_model_detailed_report.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def _tex_escape(value: Any) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in text)


def _tex_mono(value: Any) -> str:
    return r"\texttt{" + _tex_escape(value) + "}"


def _tex_float(value: Any, digits: int = 3) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if math.isnan(number):
        return "n/a"
    return f"{number:.{digits}f}"


def _tex_itemize(items: Sequence[str]) -> List[str]:
    out = [r"\begin{itemize}[leftmargin=*]"]
    for item in items:
        out.append(r"\item " + _tex_escape(item))
    out.append(r"\end{itemize}")
    return out


def _copy_pdf_safe_world_figures(discovery_dir: Path, assets_dir: Path) -> List[Path]:
    copied: List[Path] = []
    world_dir = assets_dir / "world_figures"
    world_dir.mkdir(parents=True, exist_ok=True)
    for src in sorted(discovery_dir.glob("frame_iter_*.png")):
        dst = world_dir / src.name
        shutil.copy2(src, dst)
        copied.append(dst)
    return copied


def write_latex_protein_report(discovery_dir: Path, protein_assets: Optional[Dict[str, Any]] = None) -> Path:
    summary = json.loads((discovery_dir / "run_summary.json").read_text(encoding="utf-8"))
    history = summary.get("history", [])
    iterations = summary.get("iterations", [])
    assets_dir = discovery_dir / "report_assets"
    if protein_assets is None:
        assets_path = assets_dir / "protein_report_assets.json"
        protein_assets = json.loads(assets_path.read_text(encoding="utf-8")) if assets_path.exists() else {"proteins": []}
    world_figures = _copy_pdf_safe_world_figures(discovery_dir, assets_dir)
    proteins = protein_assets.get("proteins", [])
    good_proteins = [p for p in proteins if not p.get("error")]

    tex: List[str] = [
        r"\documentclass[10pt]{article}",
        r"\usepackage[margin=0.72in]{geometry}",
        r"\usepackage[T1]{fontenc}",
        r"\usepackage[utf8]{inputenc}",
        r"\usepackage{graphicx}",
        r"\usepackage{booktabs}",
        r"\usepackage{longtable}",
        r"\usepackage{array}",
        r"\usepackage{xcolor}",
        r"\usepackage{hyperref}",
        r"\usepackage{enumitem}",
        r"\usepackage{float}",
        r"\usepackage{caption}",
        r"\usepackage{amsmath}",
        r"\hypersetup{colorlinks=true, linkcolor=blue, urlcolor=blue}",
        r"\setlength{\parskip}{0.55em}",
        r"\setlength{\parindent}{0pt}",
        r"\title{Protein World-Model Building and Breaking with DAG+MDL Discovery}",
        r"\author{Automatically generated report}",
        r"\date{\today}",
        r"\begin{document}",
        r"\maketitle",
        r"\begin{abstract}",
        (
            "This report documents an automatically generated protein world-model discovery run. "
            "The system studies real PDB structures, computes Gaussian Network Model normal-mode "
            "features from C-alpha contact networks, and fits compact DAG equations to normalized "
            "experimental B-factors. The run demonstrates a staged build-break-rebuild loop: a "
            "simple elastic-network model is built on compact proteins, challenged by terminal "
            "and hinge/domain-motion regimes, and revised only when Minimum Description Length "
            "(MDL) supports the added structure."
        ),
        r"\end{abstract}",
        r"\tableofcontents",
        r"\newpage",
        r"\section{Purpose and Thesis}",
        (
            "The goal is not to claim a new law of protein dynamics. The goal is to demonstrate "
            "the central thesis of world-model discovery on real scientific data: an AI system "
            "should build an explicit model, seek regimes where it fails, revise the ontology, "
            "and ground each revision against physical measurements. Here the physical substrate "
            "is a Gaussian Network Model (GNM), and the measurement target is crystallographic "
            "B-factor variation within each protein chain."
        ),
        r"\section{Dataset and Physics Pipeline}",
        _tex_escape(summary.get("system_description", "")),
        (
            "For each protein chain, residues are represented by C-alpha nodes. Edges connect "
            f"nodes within {_tex_float(protein_assets.get('cutoff'), 1)} Angstrom. The GNM "
            "Kirchhoff matrix is diagonalized, and low-frequency modes are used to compute "
            "mean-square fluctuation features. The target, "
            + _tex_mono("bfactor_z")
            + ", is the per-chain z-score of C-alpha B-factor."
        ),
        r"\subsection{Core Variables}",
        r"\begin{itemize}[leftmargin=*]",
        r"\item " + _tex_mono("gnm_fluct_z") + ": normalized mean-square fluctuation from the GNM pseudoinverse.",
        r"\item " + _tex_mono("gnm_fluct_log_z") + ": log-transformed fluctuation, useful when raw fluctuations are heavy-tailed.",
        r"\item " + _tex_mono("terminal_exposure") + ": boundary/terminus exposure, highest at chain ends.",
        r"\item " + _tex_mono("mode1_abs_z") + ": magnitude of participation in the slowest nonzero GNM mode.",
        r"\item " + _tex_mono("hinge_score_z") + ": local curvature of the slowest mode, used as a hinge-like proxy.",
        r"\end{itemize}",
        r"\section{World-Building and Breaking Loop}",
        (
            "At each iteration, the Builder fits a compact DAG equation to the revealed residues. "
            "The Breaker reveals the next staged protein regime. The previous DAG is refit on the "
            "expanded data, then the search proposes candidate DAG edits. A new DAG is accepted "
            "only if its total description length is shorter: "
            r"$L_{\mathrm{total}}=L_{\mathrm{model}}+L_{\mathrm{data}}$."
        ),
        r"\section{Quantitative Trajectory}",
        r"\begin{longtable}{r p{0.23\linewidth} r r r r r}",
        r"\toprule",
        r"Iter & Revealed slice & Features & RMSE & $R^2$ & $L_{\mathrm{total}}$ & Gain bits \\",
        r"\midrule",
        r"\endhead",
    ]
    for row in iterations:
        tex.append(
            f"{row.get('iteration')} & "
            f"{_tex_escape(row.get('revealed_slice'))} & "
            f"{row.get('k_features')} & "
            f"{_tex_float(row.get('rmse'), 4)} & "
            f"{_tex_float(row.get('r2'), 3)} & "
            f"{_tex_float(row.get('L_total_bits'), 1)} & "
            f"{_tex_float(row.get('break_bits_gain'), 1)} \\\\"
        )
    tex.extend([r"\bottomrule", r"\end{longtable}"])

    if world_figures:
        tex.extend([
            r"\section{World-Model Evolution Figures}",
            (
                "The following automatically generated frames show the evolving equation, "
                "DAG structure, MDL terms, and fit against revealed observations."
            ),
        ])
        for fig in world_figures:
            tex.extend([
                r"\begin{figure}[H]",
                r"\centering",
                r"\includegraphics[width=0.94\linewidth]{" + _tex_escape(fig.relative_to(discovery_dir)) + "}",
                r"\caption{" + _tex_escape(fig.stem.replace("_", " ").title()) + "}",
                r"\end{figure}",
            ])

    tex.append(r"\section{Iteration-by-Iteration Scientific Interpretation}")
    for record in history:
        version = record.get("version")
        features = record.get("features", [])
        labels = [f.get("label", "") for f in features]
        bits = record.get("bits", {})
        equations = record.get("equation_lines", [])
        tex.extend([
            r"\subsection{Iteration " + str(version) + ": " + _tex_escape(record.get("revealed_slice") or "initial compact-protein stage") + "}",
            "Accepted equation: " + _tex_mono(equations[0] if equations else "n/a"),
            (
                f"RMSE={_tex_float(bits.get('rmse'), 4)}, "
                f"$R^2$={_tex_float(bits.get('r2'), 3)}, "
                f"$L_{{\\mathrm{{total}}}}$={_tex_float(bits.get('L_total'), 1)} bits."
            ),
            "Accepted DAG features: " + (", ".join(_tex_mono(label) for label in labels) if labels else "none") + ".",
        ])
        if record.get("revealed_slice"):
            tex.append(
                "Break status: "
                + _tex_mono(record.get("break_type"))
                + f", detected={record.get('break_detected')}, MDL gain={_tex_float(record.get('break_bits_gain'), 1)} bits."
            )
        tex.extend(_tex_itemize(_feature_interpretation(labels)))
        if version == 0:
            tex.append(
                "Scientific reading: the initial ontology is a classical elastic-network world. "
                "Residue flexibility is explained mainly by the contact graph's predicted GNM fluctuation."
            )
        elif "terminal_exposure" in labels:
            tex.append(
                "Scientific reading: the model has been broken by boundary effects. It now treats chain-end "
                "position as an explanatory mechanism separate from normal-mode fluctuation."
            )
        elif "mode1_abs_z" in labels or "mode1_abs_z^2" in labels:
            tex.append(
                "Scientific reading: the model has shifted toward collective-motion ontology. Participation "
                "in the slowest global mode becomes part of the explanation."
            )
        elif "gnm_fluct_log_z" in labels:
            tex.append(
                "Scientific reading: the model retains normal-mode physics but changes scale, compressing "
                "extreme fluctuations for a more robust mixed-regime explanation."
            )

    if good_proteins:
        tex.extend([
            r"\section{Protein Case Studies}",
            (
                "Each figure shows the same C-alpha trace twice: left colored by normalized experimental "
                "B-factor, right colored by GNM-predicted fluctuation. Agreement indicates where the "
                "elastic-network world model is adequate; disagreement marks places where boundary, "
                "collective, crystallographic, or chemical context may be missing."
            ),
        ])
        for protein in good_proteins:
            caption = (
                f"{protein.get('pdb_id')} chain {protein.get('chain')} in {protein.get('stage_id')}. "
                f"n={protein.get('n_residues')}, GNM/B-factor correlation="
                f"{_tex_float(protein.get('gnm_bfactor_corr'), 3)}."
            )
            tex.extend([
                r"\begin{figure}[H]",
                r"\centering",
                r"\includegraphics[width=0.92\linewidth]{" + _tex_escape(protein.get("image", "")) + "}",
                r"\caption{" + _tex_escape(caption) + "}",
                r"\end{figure}",
            ])

    tex.extend([
        r"\section{Final Conclusions}",
        r"\begin{enumerate}[leftmargin=*]",
        r"\item The first world model is physically meaningful: C-alpha contact-network normal modes explain a substantial fraction of residue-level B-factor variation in compact proteins.",
        r"\item The model breaks when new protein regimes expose systematic residual structure. In this run, accepted revisions involved boundary exposure, slow-mode participation, and log-scaled GNM fluctuation.",
        r"\item The accepted equations are not black boxes. Each feature is a mechanistic statement about contact topology, boundary position, or collective motion.",
        r"\item The result is a methodological discovery demonstration rather than a new biological law. A stronger scientific claim would require a larger benchmark, resolution controls, external validation, and manual structural interpretation of outliers.",
        r"\end{enumerate}",
        r"\appendix",
        r"\section{Appendix: Proteins Studied}",
        r"\begin{longtable}{p{0.14\linewidth} p{0.18\linewidth} r r r r}",
        r"\toprule",
        r"PDB chain & Stage & Residues & Resolution & Corr(GNM,B) & Terminal frac \\",
        r"\midrule",
        r"\endhead",
    ])
    for protein in proteins:
        if protein.get("error"):
            tex.append(
                _tex_escape(protein.get("spec"))
                + " & "
                + _tex_escape(protein.get("stage_id"))
                + r" & \multicolumn{4}{p{0.45\linewidth}}{Skipped: "
                + _tex_escape(protein.get("error"))
                + r"} \\"
            )
            continue
        tex.append(
            f"{_tex_escape(protein.get('pdb_id'))}:{_tex_escape(protein.get('chain'))} & "
            f"{_tex_escape(protein.get('stage_id'))} & "
            f"{protein.get('n_residues')} & "
            f"{_tex_float(protein.get('resolution_angstrom'), 2)} & "
            f"{_tex_float(protein.get('gnm_bfactor_corr'), 3)} & "
            f"{_tex_float(protein.get('terminal_fraction'), 3)} \\\\"
        )
    tex.extend([
        r"\bottomrule",
        r"\end{longtable}",
        r"\section{Appendix: Highest B-Factor Residues}",
    ])
    for protein in good_proteins:
        tex.append(r"\subsection{" + _tex_escape(f"{protein.get('pdb_id')} chain {protein.get('chain')}") + "}")
        tex.append(r"\begin{tabular}{lrr}")
        tex.append(r"\toprule Residue & B-factor z & GNM fluct z \\ \midrule")
        for row in protein.get("top_bfactor_residues", []):
            tex.append(
                f"{_tex_escape(row.get('residue'))} & "
                f"{_tex_float(row.get('bfactor_z'), 2)} & "
                f"{_tex_float(row.get('gnm_fluct_z'), 2)} \\\\"
            )
        tex.append(r"\bottomrule\end{tabular}")
    tex.extend([r"\end{document}", ""])

    tex_path = discovery_dir / "protein_world_model_report.tex"
    tex_path.write_text("\n".join(tex), encoding="utf-8")
    return tex_path


def compile_latex_report(tex_path: Path) -> Optional[Path]:
    if shutil.which("pdflatex") is None:
        return None
    for _ in range(2):
        result = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex_path.name],
            cwd=str(tex_path.parent),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        if result.returncode != 0:
            log_path = tex_path.with_suffix(".pdflatex.log")
            log_path.write_text(result.stdout, encoding="utf-8")
            raise RuntimeError(f"pdflatex failed; see {log_path}")
    pdf_path = tex_path.with_suffix(".pdf")
    return pdf_path if pdf_path.exists() else None


def _short_equation(equation: str, max_len: int = 86) -> str:
    equation = equation.replace("bfactor_z = ", "B_z = ")
    equation = equation.replace("mode1_abs_z^2", "mode1^2")
    equation = equation.replace("gnm_fluct_z", "GNM")
    equation = equation.replace("gnm_fluct_log_z", "logGNM")
    equation = equation.replace("terminal_exposure", "terminus")
    equation = equation.replace("mode1_abs_z", "|mode1|")
    if len(equation) <= max_len:
        return equation
    return equation[: max_len - 3] + "..."


def _draw_stage_box(ax, x: float, y: float, w: float, h: float, title: str,
                    equation: str, color: str, edge: str = "#1f2933") -> None:
    import matplotlib.patches as patches

    box = patches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.018,rounding_size=0.018",
        facecolor=color,
        edgecolor=edge,
        linewidth=1.15,
    )
    ax.add_patch(box)
    ax.text(x + 0.02, y + h - 0.07, title, fontsize=8.5, weight="bold", va="top", color="#111827")
    wrapped = "\n".join(textwrap.wrap(equation, width=25))
    ax.text(x + 0.02, y + 0.07, wrapped, fontsize=6.8, va="bottom", family="monospace", color="#111827")


def write_integrated_summary_figure(discovery_dir: Path, protein_assets: Optional[Dict[str, Any]] = None) -> Dict[str, Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt
    import numpy as np

    summary = json.loads((discovery_dir / "run_summary.json").read_text(encoding="utf-8"))
    history = summary.get("history", [])
    iterations = summary.get("iterations", [])
    assets_dir = discovery_dir / "report_assets"
    if protein_assets is None:
        assets_path = assets_dir / "protein_report_assets.json"
        protein_assets = json.loads(assets_path.read_text(encoding="utf-8")) if assets_path.exists() else {"proteins": []}
    proteins = [p for p in protein_assets.get("proteins", []) if not p.get("error")]
    representative_order = ["1UBQ", "1AAR", "4AKE", "1P38"]
    representatives = []
    for pdb_id in representative_order:
        hit = next((p for p in proteins if p.get("pdb_id") == pdb_id), None)
        if hit is not None:
            representatives.append(hit)
    for p in proteins:
        if len(representatives) >= 4:
            break
        if p not in representatives:
            representatives.append(p)

    fig = plt.figure(figsize=(17.0, 10.4), dpi=220)
    fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(
        nrows=5, ncols=4,
        height_ratios=[0.20, 1.0, 0.12, 1.22, 0.88],
        width_ratios=[1.05, 1.0, 1.0, 1.05],
        hspace=0.28,
        wspace=0.28,
    )

    title_ax = fig.add_subplot(gs[0, :])
    title_ax.axis("off")
    title_ax.text(
        0.0, 0.82,
        "World-model building and breaking in protein flexibility",
        fontsize=18,
        weight="bold",
        color="#0b1220",
        va="top",
    )
    title_ax.text(
        0.0, 0.25,
        "Real PDB structures -> C-alpha Gaussian Network Model -> DAG+MDL equations for normalized experimental B-factors",
        fontsize=10.5,
        color="#475569",
        va="top",
    )

    flow_ax = fig.add_subplot(gs[1, :3])
    flow_ax.axis("off")
    flow_ax.set_xlim(0, 1)
    flow_ax.set_ylim(0, 1)
    palette = ["#d9f0ff", "#e9f7df", "#fff3c4", "#f2e8ff"]
    stage_names = [
        "Build: compact proteins",
        "Break I: termini",
        "Break II: hinge/domain",
        "Remodel: mixed validation",
    ]
    box_w = 0.22
    for i, record in enumerate(history[:4]):
        x = 0.01 + i * 0.245
        eq = _short_equation((record.get("equation_lines") or [""])[0])
        _draw_stage_box(flow_ax, x, 0.24, box_w, 0.60, stage_names[i] if i < len(stage_names) else f"Iter {i}", eq, palette[i % len(palette)])
        if i < min(3, len(history) - 1):
            flow_ax.annotate(
                "", xy=(x + box_w + 0.018, 0.55), xytext=(x + box_w + 0.003, 0.55),
                arrowprops=dict(arrowstyle="-|>", lw=1.4, color="#334155"),
            )
    flow_ax.text(0.01, 0.08, "A  Ontology evolves from raw GNM fluctuation to boundary and collective-mode terms.", fontsize=10.5, weight="bold")

    metric_ax = fig.add_subplot(gs[1, 3])
    iters = np.asarray([row["iteration"] for row in iterations], dtype=float)
    rmse = np.asarray([row["rmse"] for row in iterations], dtype=float)
    gain = np.asarray([max(0.0, float(row["break_bits_gain"])) for row in iterations], dtype=float)
    metric_ax.plot(iters, rmse, marker="o", lw=2.2, color="#2563eb", label="RMSE")
    metric_ax.set_xlabel("Iteration")
    metric_ax.set_ylabel("RMSE", color="#2563eb")
    metric_ax.tick_params(axis="y", labelcolor="#2563eb")
    metric_ax.set_xticks(iters)
    metric_ax.grid(True, alpha=0.28)
    gain_ax = metric_ax.twinx()
    gain_ax.bar(iters + 0.18, gain, width=0.28, color="#ef4444", alpha=0.72, label="Break gain")
    gain_ax.set_ylabel("MDL gain (bits)", color="#ef4444")
    gain_ax.tick_params(axis="y", labelcolor="#ef4444")
    metric_ax.set_title("B  Break/remodel signal", loc="left", fontsize=10.5, weight="bold")

    protein_label_ax = fig.add_subplot(gs[2, :])
    protein_label_ax.axis("off")
    protein_label_ax.text(
        0.0, 0.56,
        "C  Protein grounding: experimental B-factor z (left) versus GNM fluctuation z (right)",
        fontsize=10.5,
        weight="bold",
        color="#0b1220",
        va="center",
    )

    protein_axes = [fig.add_subplot(gs[3, i]) for i in range(4)]
    for ax, protein in zip(protein_axes, representatives[:4]):
        image_rel = protein.get("thumbnail") or protein.get("image")
        image_path = discovery_dir / image_rel if image_rel else None
        ax.axis("off")
        if image_path and image_path.exists():
            try:
                from PIL import Image

                pil_img = Image.open(image_path).convert("RGB")
                arr = np.asarray(pil_img)
                mask = np.any(arr < 246, axis=2)
                ys, xs = np.where(mask)
                if len(xs) and len(ys):
                    pad = 18
                    left = max(0, int(xs.min()) - pad)
                    right = min(arr.shape[1], int(xs.max()) + pad)
                    top = max(0, int(ys.min()) - pad)
                    bottom = min(arr.shape[0], int(ys.max()) + pad)
                    img = np.asarray(pil_img.crop((left, top, right, bottom)))
                else:
                    img = mpimg.imread(image_path)
            except Exception:
                img = mpimg.imread(image_path)
            ax.imshow(img, aspect="auto")
        ax.set_title(
            f"{protein.get('pdb_id')}:{protein.get('chain')}  r={_tex_float(protein.get('gnm_bfactor_corr'), 2)}",
            fontsize=9.5,
            weight="bold",
            pad=3,
        )
    for ax in protein_axes[len(representatives[:4]):]:
        ax.axis("off")

    law_ax = fig.add_subplot(gs[4, :2])
    law_ax.axis("off")
    final = history[-1]
    final_eq = (final.get("equation_lines") or ["n/a"])[0]
    law_ax.text(0.0, 0.95, "D  Final compact law from this run", fontsize=11, weight="bold", va="top")
    law_ax.text(
        0.02, 0.68,
        "\n".join(textwrap.wrap(final_eq, width=74)),
        fontsize=11,
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8fafc", edgecolor="#94a3b8"),
        va="top",
    )
    law_ax.text(
        0.02, 0.27,
        "Interpretation: the final model retains normal-mode physics but remodels the scale of flexibility "
        "and the role of collective slow-mode participation after harder protein regimes are revealed.",
        fontsize=10,
        color="#334155",
        va="top",
        wrap=True,
    )

    conclusion_ax = fig.add_subplot(gs[4, 2:])
    conclusion_ax.axis("off")
    conclusion_ax.text(0.0, 0.95, "E  What the discovery demonstrates", fontsize=11, weight="bold", va="top")
    bullets = [
        "World model: residue B-factor is first explained by C-alpha elastic-network fluctuation.",
        "Breaking: terminal and hinge/domain regimes expose systematic failures.",
        "Remodeling: MDL accepts only compact new DAG structure that pays for itself.",
        "Grounding: all hypotheses are tested against real PDB-derived measurements.",
    ]
    y = 0.75
    for b in bullets:
        conclusion_ax.text(0.02, y, u"\u2022 " + b, fontsize=10, color="#1f2937", va="top", wrap=True)
        y -= 0.18

    out_base = discovery_dir / "protein_integrated_discovery_figure"
    fig.subplots_adjust(left=0.035, right=0.985, top=0.965, bottom=0.045)
    paths = {
        "png": out_base.with_suffix(".png"),
        "pdf": out_base.with_suffix(".pdf"),
        "svg": out_base.with_suffix(".svg"),
    }
    fig.savefig(paths["png"], dpi=300, bbox_inches="tight")
    fig.savefig(paths["pdf"], bbox_inches="tight")
    fig.savefig(paths["svg"], bbox_inches="tight")
    plt.close(fig)
    return paths


def command_build(args: argparse.Namespace) -> int:
    outdir = Path(args.outdir).resolve()
    dataset, json_path, csv_path, _ = build_dataset_from_args(
        args, outdir, json_name=args.json_name, csv_name=args.csv_name,
    )
    write_dataset_readme(outdir, dataset, json_path, csv_path)
    print(f"Wrote dataset JSON: {json_path}")
    print(f"Wrote observations CSV: {csv_path}")
    for stage, count in stage_counts(dataset).items():
        print(f"  {stage}: {count} residue observations")
    return 0


def command_run(args: argparse.Namespace) -> int:
    outdir = Path(args.outdir).resolve()
    dataset_dir = outdir / args.dataset_subdir
    discovery_dir = outdir / args.discovery_subdir
    dataset, json_path, csv_path, stage_specs = build_dataset_from_args(args, dataset_dir)
    write_dataset_readme(dataset_dir, dataset, json_path, csv_path)

    from world_model_breaker_cli import main as discovery_main

    discovery_args = [
        "--dataset-json", str(json_path),
        "--outdir", str(discovery_dir),
        "--rounds", str(args.rounds),
        "--search-steps", str(args.search_steps),
        "--search-patience", str(args.search_patience),
        "--search-restarts", str(args.search_restarts),
        "--seed", str(args.seed),
        "--no-llm-dataset-analysis",
    ]
    if args.no_llm or not args.llm_builder:
        discovery_args.append("--no-llm")
    if args.llm_builder:
        discovery_args.extend([
            "--llm-builder",
            "--model", args.model,
            "--reasoning-effort", args.reasoning_effort,
        ])
    if args.quiet:
        discovery_args.append("--quiet")

    print(f"Wrote dataset JSON: {json_path}")
    print(f"Wrote observations CSV: {csv_path}")
    print(f"Running discovery into: {discovery_dir}")
    code = discovery_main(discovery_args)
    if code == 0:
        assets = generate_protein_report_assets(
            stage_specs=stage_specs,
            cache_dir=Path(args.pdb_cache).resolve(),
            assets_dir=discovery_dir / "report_assets",
            cutoff=args.cutoff,
            n_modes=args.n_modes,
            terminal_window=args.terminal_window,
            min_residues=args.min_residues,
        )
        report_path = write_attachable_protein_report(discovery_dir)
        tex_path = write_latex_protein_report(discovery_dir, assets)
        figure_paths = write_integrated_summary_figure(discovery_dir, assets)
        print(f"Wrote detailed protein report: {report_path}")
        print(f"Wrote LaTeX report: {tex_path}")
        print(f"Wrote integrated figure: {figure_paths['pdf']}")
        try:
            pdf_path = compile_latex_report(tex_path)
            if pdf_path is not None:
                print(f"Wrote PDF report: {pdf_path}")
        except RuntimeError as exc:
            print(str(exc))
    return code


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.command == "build":
        return command_build(args)
    if args.command == "run":
        return command_run(args)
    raise RuntimeError(f"unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
