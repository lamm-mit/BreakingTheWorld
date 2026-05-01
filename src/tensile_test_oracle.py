from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

from discovery_data import DiscoveryDataset, Observation


STAGE_ORDER = [
    "stage_1_elastic_loading",
    "stage_2_post_yield_loading",
    "stage_3_unloading",
    "stage_4_reloading",
]

STAGE_LABELS = {
    "stage_1_elastic_loading": "Elastic loading",
    "stage_2_post_yield_loading": "Post-yield loading",
    "stage_3_unloading": "Unloading",
    "stage_4_reloading": "Reloading",
}

STAGE_PREREQUISITES = {
    "stage_2_post_yield_loading": ["stage_1_elastic_loading"],
    "stage_3_unloading": ["stage_2_post_yield_loading"],
    "stage_4_reloading": ["stage_3_unloading"],
}

PROTOCOL_DESCRIPTIONS = {
    "stage_1_elastic_loading": (
        "Initial monotonic loading at low strain. Expected to be close to linear "
        "elastic behavior."
    ),
    "stage_2_post_yield_loading": (
        "Continued monotonic loading past the yield threshold. This may reveal "
        "hardening, curvature, or a slope change relative to elastic loading."
    ),
    "stage_3_unloading": (
        "Unloading from the high-strain state. This probes path dependence, "
        "hysteresis, residual strain, or a changed tangent stiffness."
    ),
    "stage_4_reloading": (
        "Reloading after partial unloading. It starts continuously from the final "
        "unloading stress and probes whether the material rejoins the loading curve "
        "or follows a history-dependent branch."
    ),
}

SYSTEM_DESCRIPTION = (
    "Synthetic one-dimensional tensile-test experiment. The observable inputs are "
    "engineering strain and loading direction; the target is measured stress. The "
    "physical protocol is ordered: begin with elastic loading, continue into "
    "post-yield loading, unload from the high-strain state, then reload from the "
    "partially unloaded state. Later protocol stages are not physically collectable "
    "until their prerequisites have occurred, so data discovery should respect this "
    "experimental sequence rather than freely jumping to any hidden slice."
)


@dataclass
class TensileTestOracle:
    """Simulator oracle for a staged tensile-test dataset.

    The oracle can generate all protocol stages up front for reproducible demos,
    or future runners can call `collect_stage` incrementally. Reloading is
    continuous with the end of unloading.
    """
    seed: int = 7
    noise_std: float = 0.9
    elastic_modulus: float = 2000.0
    yield_strain: float = 0.02
    hardening_modulus: float = 260.0
    nonlinear_hardening: float = 4200.0
    unload_slope: float = 1450.0
    reload_slope: float = 1650.0
    reload_curvature: float = 500.0

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)
        self.collected_stages: Dict[str, List[Observation]] = {}
        self.sigma_y = self.elastic_modulus * self.yield_strain
        self.peak_strain = 0.060
        self.unload_end_strain = 0.035
        self.sigma_peak = self.loading_backbone(self.peak_strain)
        self.sigma_unload_end = self.unloading_branch(self.unload_end_strain)

    def loading_backbone(self, strain: float) -> float:
        z = max(strain - self.yield_strain, 0.0)
        if strain <= self.yield_strain:
            return self.elastic_modulus * strain
        return self.sigma_y + self.hardening_modulus * z + self.nonlinear_hardening * z * z

    def unloading_branch(self, strain: float) -> float:
        return self.sigma_peak - self.unload_slope * (self.peak_strain - strain)

    def reloading_branch(self, strain: float) -> float:
        z = max(strain - self.unload_end_strain, 0.0)
        return self.sigma_unload_end + self.reload_slope * z + self.reload_curvature * z * z

    def noisy(self, stress: float) -> float:
        return float(stress + self.rng.normal(0.0, self.noise_std))

    def stage_strains(self, stage_id: str) -> np.ndarray:
        if stage_id == "stage_1_elastic_loading":
            return np.linspace(0.000, 0.018, 28)
        if stage_id == "stage_2_post_yield_loading":
            return np.linspace(0.018, self.peak_strain, 34)[1:]
        if stage_id == "stage_3_unloading":
            return np.linspace(self.peak_strain, self.unload_end_strain, 26)
        if stage_id == "stage_4_reloading":
            return np.linspace(self.unload_end_strain, 0.055, 24)
        raise KeyError(f"unknown tensile-test stage: {stage_id}")

    def stage_direction(self, stage_id: str) -> int:
        return -1 if stage_id == "stage_3_unloading" else 1

    def stress_for(self, stage_id: str, strain: float) -> float:
        if stage_id in ("stage_1_elastic_loading", "stage_2_post_yield_loading"):
            return self.loading_backbone(strain)
        if stage_id == "stage_3_unloading":
            return self.unloading_branch(strain)
        if stage_id == "stage_4_reloading":
            return self.reloading_branch(strain)
        raise KeyError(f"unknown tensile-test stage: {stage_id}")

    def collect_stage(self, stage_id: str, idx_start: int = 0, t_start: int = 0) -> List[Observation]:
        observations: List[Observation] = []
        idx = idx_start
        t = t_start
        direction = self.stage_direction(stage_id)
        for strain in self.stage_strains(stage_id):
            stress = self.noisy(self.stress_for(stage_id, float(strain)))
            observations.append(
                Observation(
                    idx=idx,
                    t=t,
                    strain=float(strain),
                    direction=direction,
                    stress=stress,
                    stage_id=stage_id,
                    stage_label=STAGE_LABELS[stage_id],
                    observables={"strain": float(strain), "direction": direction},
                    target_value=stress,
                )
            )
            idx += 1
            t += 1
        if (
            stage_id == "stage_3_unloading"
            and observations
            and "stage_2_post_yield_loading" in self.collected_stages
            and self.collected_stages["stage_2_post_yield_loading"]
        ):
            observations[0].stress = self.collected_stages["stage_2_post_yield_loading"][-1].stress
            observations[0].target_value = observations[0].stress
        if (
            stage_id == "stage_4_reloading"
            and observations
            and "stage_3_unloading" in self.collected_stages
            and self.collected_stages["stage_3_unloading"]
        ):
            # The underlying branch is continuous; make the measured staged
            # demo continuous as well, even when observation noise is enabled.
            observations[0].stress = self.collected_stages["stage_3_unloading"][-1].stress
            observations[0].target_value = observations[0].stress
        self.collected_stages[stage_id] = observations
        return observations

    def generate_observations(self, stage_order: Sequence[str] = STAGE_ORDER) -> List[Observation]:
        observations: List[Observation] = []
        idx = 0
        t = 0
        for stage_id in stage_order:
            stage_obs = self.collect_stage(stage_id, idx_start=idx, t_start=t)
            observations.extend(stage_obs)
            idx += len(stage_obs)
            t += len(stage_obs)
        return observations

    def generate_dataset(self) -> DiscoveryDataset:
        return DiscoveryDataset(
            observations=self.generate_observations(),
            stage_order=list(STAGE_ORDER),
            initial_stage_ids=["stage_1_elastic_loading"],
            stage_prerequisites={k: list(v) for k, v in STAGE_PREREQUISITES.items()},
            system_description=SYSTEM_DESCRIPTION,
            observable_descriptions={
                "strain": "Engineering strain applied during the tensile test.",
                "direction": "Loading direction: +1 for loading/reloading, -1 for unloading.",
            },
            target_description="Measured stress response of the material specimen.",
            target_name="stress",
            observable_types={"strain": "continuous", "direction": "discrete"},
            protocol_descriptions=dict(PROTOCOL_DESCRIPTIONS),
            source=f"tensile_test_oracle(seed={self.seed}, noise_std={self.noise_std})",
        )


def generate_oracle_dataset(seed: int = 7, noise_std: float = 0.9) -> List[Observation]:
    return TensileTestOracle(seed=seed, noise_std=noise_std).generate_observations()


FRACTURE_STAGE_ORDER = [
    "stage_1_elastic_loading",
    "stage_2_strain_hardening",
    "stage_3_fracture_softening",
    "stage_4_release",
]

FRACTURE_STAGE_LABELS = {
    "stage_1_elastic_loading": "Elastic loading",
    "stage_2_strain_hardening": "Strain hardening",
    "stage_3_fracture_softening": "Fracture softening",
    "stage_4_release": "Release",
}

FRACTURE_STAGE_PREREQUISITES = {
    "stage_2_strain_hardening": ["stage_1_elastic_loading"],
    "stage_3_fracture_softening": ["stage_2_strain_hardening"],
    "stage_4_release": ["stage_3_fracture_softening"],
}

FRACTURE_PROTOCOL_DESCRIPTIONS = {
    "stage_1_elastic_loading": (
        "Initial tensile loading at small strain with an approximately linear stress response."
    ),
    "stage_2_strain_hardening": (
        "Continued loading after first yield/damage onset. Stress continues increasing but "
        "with a changed tangent stiffness and curvature."
    ),
    "stage_3_fracture_softening": (
        "Continued displacement-controlled loading through damage localization and fracture. "
        "Stress drops as strain increases, representing loss of load-bearing capacity."
    ),
    "stage_4_release": (
        "Release/unloading after fracture. Stress relaxes continuously from the fractured "
        "residual load toward near-zero stress."
    ),
}

FRACTURE_SYSTEM_DESCRIPTION = (
    "Synthetic one-dimensional tensile fracture experiment. The observable inputs are "
    "engineering strain and loading direction; the target is measured stress. The physical "
    "protocol is ordered: elastic loading, increased loading with hardening/damage onset, "
    "fracture softening under continued displacement, then release/unloading after fracture. "
    "The fracture and release stages are not physically collectable before the preceding "
    "loading history has occurred."
)


@dataclass
class TensileFractureOracle:
    """Simulator oracle for tensile loading through fracture and release."""
    seed: int = 11
    noise_std: float = 0.7
    elastic_modulus: float = 2200.0
    yield_strain: float = 0.015
    peak_strain: float = 0.045
    fracture_end_strain: float = 0.065
    release_end_strain: float = 0.050
    hardening_modulus: float = 520.0
    hardening_curvature: float = 9000.0
    residual_stress: float = 8.0

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)
        self.collected_stages: Dict[str, List[Observation]] = {}
        self.sigma_y = self.elastic_modulus * self.yield_strain
        self.sigma_peak = self.hardening_branch(self.peak_strain)

    def hardening_branch(self, strain: float) -> float:
        if strain <= self.yield_strain:
            return self.elastic_modulus * strain
        z = strain - self.yield_strain
        return self.sigma_y + self.hardening_modulus * z + self.hardening_curvature * z * z

    def fracture_branch(self, strain: float) -> float:
        q = min(1.0, max(0.0, (strain - self.peak_strain) / (self.fracture_end_strain - self.peak_strain)))
        # Smooth monotone softening from peak stress to residual stress.
        smooth = q * q * (3.0 - 2.0 * q)
        return self.sigma_peak * (1.0 - smooth) + self.residual_stress * smooth

    def release_branch(self, strain: float) -> float:
        q = min(1.0, max(0.0, (self.fracture_end_strain - strain) / (self.fracture_end_strain - self.release_end_strain)))
        return self.residual_stress * (1.0 - q)

    def noisy(self, stress: float) -> float:
        return float(stress + self.rng.normal(0.0, self.noise_std))

    def stage_strains(self, stage_id: str) -> np.ndarray:
        if stage_id == "stage_1_elastic_loading":
            return np.linspace(0.000, self.yield_strain, 26)
        if stage_id == "stage_2_strain_hardening":
            return np.linspace(self.yield_strain, self.peak_strain, 30)[1:]
        if stage_id == "stage_3_fracture_softening":
            return np.linspace(self.peak_strain, self.fracture_end_strain, 28)
        if stage_id == "stage_4_release":
            return np.linspace(self.fracture_end_strain, self.release_end_strain, 24)
        raise KeyError(f"unknown tensile-fracture stage: {stage_id}")

    def stage_direction(self, stage_id: str) -> int:
        return -1 if stage_id == "stage_4_release" else 1

    def stress_for(self, stage_id: str, strain: float) -> float:
        if stage_id in ("stage_1_elastic_loading", "stage_2_strain_hardening"):
            return self.hardening_branch(strain)
        if stage_id == "stage_3_fracture_softening":
            return self.fracture_branch(strain)
        if stage_id == "stage_4_release":
            return self.release_branch(strain)
        raise KeyError(f"unknown tensile-fracture stage: {stage_id}")

    def collect_stage(self, stage_id: str, idx_start: int = 0, t_start: int = 0) -> List[Observation]:
        observations: List[Observation] = []
        idx = idx_start
        t = t_start
        direction = self.stage_direction(stage_id)
        for strain in self.stage_strains(stage_id):
            stress = self.noisy(self.stress_for(stage_id, float(strain)))
            observations.append(
                Observation(
                    idx=idx,
                    t=t,
                    strain=float(strain),
                    direction=direction,
                    stress=stress,
                    stage_id=stage_id,
                    stage_label=FRACTURE_STAGE_LABELS[stage_id],
                    observables={"strain": float(strain), "direction": direction},
                    target_value=stress,
                )
            )
            idx += 1
            t += 1
        if (
            stage_id == "stage_3_fracture_softening"
            and observations
            and "stage_2_strain_hardening" in self.collected_stages
        ):
            observations[0].stress = self.collected_stages["stage_2_strain_hardening"][-1].stress
            observations[0].target_value = observations[0].stress
        if (
            stage_id == "stage_4_release"
            and observations
            and "stage_3_fracture_softening" in self.collected_stages
        ):
            observations[0].stress = self.collected_stages["stage_3_fracture_softening"][-1].stress
            observations[0].target_value = observations[0].stress
        self.collected_stages[stage_id] = observations
        return observations

    def generate_observations(self, stage_order: Sequence[str] = FRACTURE_STAGE_ORDER) -> List[Observation]:
        observations: List[Observation] = []
        idx = 0
        t = 0
        for stage_id in stage_order:
            stage_obs = self.collect_stage(stage_id, idx_start=idx, t_start=t)
            observations.extend(stage_obs)
            idx += len(stage_obs)
            t += len(stage_obs)
        return observations

    def generate_dataset(self) -> DiscoveryDataset:
        return DiscoveryDataset(
            observations=self.generate_observations(),
            stage_order=list(FRACTURE_STAGE_ORDER),
            initial_stage_ids=["stage_1_elastic_loading"],
            stage_prerequisites={k: list(v) for k, v in FRACTURE_STAGE_PREREQUISITES.items()},
            system_description=FRACTURE_SYSTEM_DESCRIPTION,
            observable_descriptions={
                "strain": "Engineering strain/displacement-control coordinate in the tensile fracture test.",
                "direction": "Protocol direction: +1 for increasing displacement, -1 for release/unloading.",
            },
            target_description="Measured tensile stress/load carried by the specimen.",
            target_name="stress",
            observable_types={"strain": "continuous", "direction": "discrete"},
            protocol_descriptions=dict(FRACTURE_PROTOCOL_DESCRIPTIONS),
            source=f"tensile_fracture_oracle(seed={self.seed}, noise_std={self.noise_std})",
        )


def synthetic_discovery_dataset(
    seed: int = 7,
    noise_std: float = 0.9,
    example: str = "cyclic",
) -> DiscoveryDataset:
    if example == "cyclic":
        return TensileTestOracle(seed=seed, noise_std=noise_std).generate_dataset()
    if example == "fracture":
        return TensileFractureOracle(seed=seed, noise_std=noise_std).generate_dataset()
    raise ValueError(f"unknown synthetic dataset example: {example}")
