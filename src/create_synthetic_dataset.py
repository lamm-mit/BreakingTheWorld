#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

from discovery_data import write_dataset_json, write_observations_csv
from oracle_adapters import DuffingOscillatorOracle
from tensile_test_oracle import TensileFractureOracle, TensileTestOracle


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Create a synthetic staged DiscoveryDataset snapshot.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--outdir", type=str, default="synthetic_tensile_dataset")
    p.add_argument("--example", choices=["cyclic", "fracture", "duffing"], default="cyclic",
                   help="Synthetic example to generate as a reusable dataset snapshot")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--noise-std", type=float, default=0.9)
    p.add_argument("--json-name", type=str, default="dataset.json")
    p.add_argument("--csv-name", type=str, default="observations.csv")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    if args.example == "cyclic":
        oracle = TensileTestOracle(seed=args.seed, noise_std=args.noise_std)
        dataset = oracle.generate_dataset()
    elif args.example == "fracture":
        oracle = TensileFractureOracle(seed=args.seed, noise_std=args.noise_std)
        dataset = oracle.generate_dataset()
    else:
        oracle = DuffingOscillatorOracle(seed=args.seed, noise_std=args.noise_std)
        dataset = oracle.dataset_shell()
        observations = []
        idx = 0
        t = 0
        for stage_id in dataset.stage_order:
            stage_obs = oracle.collect_stage(stage_id, idx_start=idx, t_start=t)
            observations.extend(stage_obs)
            idx += len(stage_obs)
            t += len(stage_obs)
        dataset.observations = observations

    json_path = outdir / args.json_name
    csv_path = outdir / args.csv_name
    write_dataset_json(dataset, json_path)
    write_observations_csv(dataset, csv_path)

    readme = outdir / "README.md"
    readme.write_text(
        "\n".join([
            f"# Synthetic Dataset: {args.example}",
            "",
            dataset.system_description,
            "",
            "Files:",
            f"- `{json_path.name}`: full `DiscoveryDataset` with metadata, stage prerequisites, and observations.",
            f"- `{csv_path.name}`: flat observation table for inspection/plotting.",
            "",
            "Use with discovery CLI:",
            "",
            "```bash",
            f"python src/world_model_breaker_cli.py --dataset-json {json_path} --outdir demo_run",
            "```",
            "",
            "For live active discovery, prefer running the relevant oracle directly when available.",
        ]),
        encoding="utf-8",
    )

    print(f"Wrote dataset JSON: {json_path}")
    print(f"Wrote observations CSV: {csv_path}")
    print(f"Wrote README: {readme}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
