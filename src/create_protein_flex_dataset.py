#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from discovery_data import write_dataset_json, write_observations_csv
from protein_nma_oracle import (
    PDBSpec,
    build_protein_flex_dataset,
    default_stage_specs,
    parse_stage_spec,
    stage_counts,
)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Build a staged protein-flexibility DiscoveryDataset from PDB structures. "
            "Residue observations contain C-alpha GNM normal-mode features and the "
            "target is normalized experimental B-factor."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--outdir", type=str, default="data/protein_flex")
    p.add_argument("--pdb-cache", type=str, default="data/pdb_cache")
    p.add_argument(
        "--stage",
        action="append",
        default=[],
        help=(
            "Override/add a stage as stage_id=pdb:chain,pdb:chain. Example: "
            "--stage compact=1ubq:A,1crn:A. If omitted, a small curated default "
            "set is used."
        ),
    )
    p.add_argument(
        "--initial-stage",
        action="append",
        default=[],
        help="Stage id initially revealed to the Builder. Defaults to the first stage.",
    )
    p.add_argument("--cutoff", type=float, default=10.0, help="C-alpha contact cutoff in Angstrom")
    p.add_argument("--n-modes", type=int, default=20, help="Number of nonzero GNM modes to keep")
    p.add_argument("--terminal-window", type=int, default=5, help="Residues counted as terminal")
    p.add_argument("--min-residues", type=int, default=20, help="Skip chains shorter than this")
    p.add_argument("--json-name", type=str, default="dataset.json")
    p.add_argument("--csv-name", type=str, default="observations.csv")
    return p.parse_args(argv)


def collect_stage_specs(stage_args: Sequence[str]) -> Dict[str, List[PDBSpec]]:
    if not stage_args:
        return default_stage_specs()
    out: Dict[str, List[PDBSpec]] = {}
    for item in stage_args:
        stage_id, specs = parse_stage_spec(item)
        out[stage_id] = specs
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.pdb_cache).resolve()
    stage_specs = collect_stage_specs(args.stage)

    dataset = build_protein_flex_dataset(
        stage_specs=stage_specs,
        cache_dir=cache_dir,
        cutoff=args.cutoff,
        n_modes=args.n_modes,
        terminal_window=args.terminal_window,
        min_residues=args.min_residues,
        initial_stage_ids=args.initial_stage or None,
    )

    json_path = outdir / args.json_name
    csv_path = outdir / args.csv_name
    write_dataset_json(dataset, json_path)
    write_observations_csv(dataset, csv_path)

    counts = stage_counts(dataset)
    readme = outdir / "README.md"
    readme.write_text(
        "\n".join([
            "# Protein Flexibility Dataset",
            "",
            dataset.system_description,
            "",
            "Target:",
            f"- `{dataset.target_name}`: {dataset.target_description}",
            "",
            "Stages:",
            *[f"- `{stage}`: {counts.get(stage, 0)} residue observations" for stage in dataset.stage_order],
            "",
            "Files:",
            f"- `{json_path.name}`: full staged `DiscoveryDataset`.",
            f"- `{csv_path.name}`: flat observation table.",
            "",
            "Run discovery:",
            "",
            "```bash",
            f"python src/world_model_breaker_cli.py --dataset-json {json_path} --outdir runs/protein_flex --no-llm --rounds 3",
            "```",
        ]),
        encoding="utf-8",
    )

    print(f"Wrote dataset JSON: {json_path}")
    print(f"Wrote observations CSV: {csv_path}")
    print(f"Wrote README: {readme}")
    for stage, count in counts.items():
        print(f"  {stage}: {count} residue observations")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

