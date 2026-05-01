from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class Observation:
    idx: int
    t: int
    strain: float
    direction: int
    stress: float
    stage_id: str
    stage_label: str
    observables: Dict[str, Any] = field(default_factory=dict)
    target_value: Optional[float] = None


@dataclass
class DiscoveryDataset:
    """Staged dataset consumed by DiscoveryRunner.

    Observations may come from a saved file, a simulator, a lab interface, or
    another oracle. The runner assumes observations are grouped into protocol
    stages and that one or more initial stages are already revealed.
    """
    observations: List[Observation]
    stage_order: List[str]
    initial_stage_ids: List[str]
    stage_prerequisites: Dict[str, List[str]] = field(default_factory=dict)
    system_description: str = ""
    observable_descriptions: Dict[str, str] = field(default_factory=dict)
    target_description: str = ""
    target_name: str = "stress"
    observable_types: Dict[str, str] = field(default_factory=dict)
    protocol_descriptions: Dict[str, str] = field(default_factory=dict)
    source: str = ""


def dataset_to_dict(dataset: DiscoveryDataset) -> Dict[str, Any]:
    return {
        "observations": [asdict(o) for o in dataset.observations],
        "stage_order": dataset.stage_order,
        "initial_stage_ids": dataset.initial_stage_ids,
        "stage_prerequisites": dataset.stage_prerequisites,
        "system_description": dataset.system_description,
        "observable_descriptions": dataset.observable_descriptions,
        "target_description": dataset.target_description,
        "target_name": dataset.target_name,
        "observable_types": dataset.observable_types,
        "protocol_descriptions": dataset.protocol_descriptions,
        "source": dataset.source,
    }


def dataset_from_dict(data: Dict[str, Any]) -> DiscoveryDataset:
    return DiscoveryDataset(
        observations=[Observation(**row) for row in data["observations"]],
        stage_order=list(data["stage_order"]),
        initial_stage_ids=list(data["initial_stage_ids"]),
        stage_prerequisites={k: list(v) for k, v in data.get("stage_prerequisites", {}).items()},
        system_description=data.get("system_description", ""),
        observable_descriptions=dict(data.get("observable_descriptions", {})),
        target_description=data.get("target_description", ""),
        target_name=data.get("target_name", "stress"),
        observable_types=dict(data.get("observable_types", {})),
        protocol_descriptions=dict(data.get("protocol_descriptions", {})),
        source=data.get("source", ""),
    )


def observation_inputs(obs: Observation) -> Dict[str, Any]:
    if obs.observables:
        return dict(obs.observables)
    return {"strain": obs.strain, "direction": obs.direction}


def observation_target(obs: Observation) -> float:
    if obs.target_value is not None:
        return float(obs.target_value)
    return float(obs.stress)


def write_dataset_json(dataset: DiscoveryDataset, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dataset_to_dict(dataset), indent=2, sort_keys=True), encoding="utf-8")


def read_dataset_json(path: Path) -> DiscoveryDataset:
    return dataset_from_dict(json.loads(path.read_text(encoding="utf-8")))


def write_observations_csv(dataset: DiscoveryDataset, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "idx", "t", "strain", "direction", "stress", "stage_id", "stage_label",
        "observables", "target_value",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for obs in dataset.observations:
            writer.writerow(asdict(obs))
