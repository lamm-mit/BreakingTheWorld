from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Protocol

import numpy as np

from dag_model import VarSpec
from discovery_data import DiscoveryDataset, Observation


class OracleAdapter(Protocol):
    """Live data-source interface for active discovery."""

    def dataset_shell(self) -> DiscoveryDataset:
        """Return metadata, stage order, and initial stage ids without observations."""

    def var_space(self) -> List[VarSpec]:
        """Return observable variable ranges/kinds available to the world model."""

    def preview_stage(self, stage_id: str, idx_start: int = 0, t_start: int = 0) -> List[Observation]:
        """Return observable inputs for an action without hidden target values."""

    def collect_stage(self, stage_id: str, idx_start: int = 0, t_start: int = 0) -> List[Observation]:
        """Run the experiment/simulation and return newly revealed observations."""


DUFFING_STAGE_ORDER = [
    "traj_1_small_free_decay",
    "traj_2_large_free_decay",
    "traj_3_high_velocity_decay",
    "traj_4_forced_response",
]

DUFFING_STAGE_LABELS = {
    "traj_1_small_free_decay": "Small-amplitude free decay",
    "traj_2_large_free_decay": "Large-amplitude free decay",
    "traj_3_high_velocity_decay": "High-velocity decay",
    "traj_4_forced_response": "Forced response",
}

DUFFING_STAGE_PREREQUISITES = {
    "traj_2_large_free_decay": ["traj_1_small_free_decay"],
    "traj_3_high_velocity_decay": ["traj_1_small_free_decay"],
    "traj_4_forced_response": ["traj_2_large_free_decay"],
}

DUFFING_PROTOCOL_DESCRIPTIONS = {
    "traj_1_small_free_decay": (
        "Simulate a free-decay trajectory from a small initial displacement. "
        "This mostly probes the local linear restoring law and light damping."
    ),
    "traj_2_large_free_decay": (
        "Simulate a free-decay trajectory from a larger displacement. This can "
        "reveal amplitude-dependent nonlinear stiffness beyond the linear regime."
    ),
    "traj_3_high_velocity_decay": (
        "Simulate free decay from a high initial velocity. This stresses whether "
        "the law needs a velocity-dependent damping term."
    ),
    "traj_4_forced_response": (
        "Simulate a driven trajectory under sinusoidal forcing. This probes whether "
        "the governing acceleration law needs an external forcing input."
    ),
}

DUFFING_SYSTEM_DESCRIPTION = (
    "Live simulated nonlinear oscillator. The oracle integrates a hidden governing "
    "law for a single mass with displacement x, velocity v, damping, cubic stiffness, "
    "and optional external forcing. The Builder sees collected trajectory samples "
    "with observables x, v, force, and time, and tries to discover a compact law for "
    "the acceleration target dvdt. The hidden equation is not provided to the agents."
)


@dataclass
class DuffingOscillatorOracle:
    """Live simulator for governing-law discovery.

    Hidden law:
        dx/dt = v
        dv/dt = -k*x - c*v - alpha*x^3 + force(t)

    The target exposed to discovery is dvdt. Preview calls integrate internally
    but return only observable inputs with target_value=None.
    """

    seed: int = 17
    noise_std: float = 0.02
    k: float = 1.25
    c: float = 0.18
    alpha: float = 0.85
    dt: float = 0.04
    duration: float = 12.0
    sample_every: int = 4

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)
        self.collected_stages: Dict[str, List[Observation]] = {}

    def dataset_shell(self) -> DiscoveryDataset:
        return DiscoveryDataset(
            observations=[],
            stage_order=list(DUFFING_STAGE_ORDER),
            initial_stage_ids=["traj_1_small_free_decay"],
            stage_prerequisites={k: list(v) for k, v in DUFFING_STAGE_PREREQUISITES.items()},
            system_description=DUFFING_SYSTEM_DESCRIPTION,
            observable_descriptions={
                "x": "Oscillator displacement.",
                "v": "Oscillator velocity.",
                "force": "Externally applied forcing at the sampled time.",
                "time": "Simulation time for the sampled state.",
            },
            observable_types={
                "x": "continuous",
                "v": "continuous",
                "force": "continuous",
                "time": "continuous",
            },
            target_name="dvdt",
            target_description="Instantaneous oscillator acceleration.",
            protocol_descriptions=dict(DUFFING_PROTOCOL_DESCRIPTIONS),
            source=f"live_duffing_oracle(seed={self.seed}, noise_std={self.noise_std})",
        )

    def var_space(self) -> List[VarSpec]:
        return [
            VarSpec(name="x", kind="continuous", lo=-2.5, hi=2.5),
            VarSpec(name="v", kind="continuous", lo=-3.0, hi=3.0),
            VarSpec(name="force", kind="continuous", lo=-1.0, hi=1.0),
            VarSpec(name="time", kind="continuous", lo=0.0, hi=self.duration),
        ]

    def stage_config(self, stage_id: str) -> Dict[str, float]:
        if stage_id == "traj_1_small_free_decay":
            return {"x0": 0.35, "v0": 0.0, "amp": 0.0, "omega": 1.0}
        if stage_id == "traj_2_large_free_decay":
            return {"x0": 1.65, "v0": 0.0, "amp": 0.0, "omega": 1.0}
        if stage_id == "traj_3_high_velocity_decay":
            return {"x0": 0.0, "v0": 2.25, "amp": 0.0, "omega": 1.0}
        if stage_id == "traj_4_forced_response":
            return {"x0": 0.55, "v0": 0.0, "amp": 0.65, "omega": 1.35}
        raise KeyError(f"unknown Duffing stage: {stage_id}")

    def force_at(self, t: float, amp: float, omega: float) -> float:
        return float(amp * np.sin(omega * t))

    def acceleration(self, x: float, v: float, force: float) -> float:
        return float(-self.k * x - self.c * v - self.alpha * x ** 3 + force)

    def rhs(self, state: np.ndarray, t: float, amp: float, omega: float) -> np.ndarray:
        x, v = float(state[0]), float(state[1])
        force = self.force_at(t, amp, omega)
        return np.asarray([v, self.acceleration(x, v, force)], dtype=float)

    def integrate_inputs(self, stage_id: str) -> List[Dict[str, float]]:
        cfg = self.stage_config(stage_id)
        state = np.asarray([cfg["x0"], cfg["v0"]], dtype=float)
        rows: List[Dict[str, float]] = []
        n_steps = int(round(self.duration / self.dt))
        for step in range(n_steps + 1):
            t = step * self.dt
            if step % self.sample_every == 0:
                rows.append({
                    "time": float(t),
                    "x": float(state[0]),
                    "v": float(state[1]),
                    "force": self.force_at(t, cfg["amp"], cfg["omega"]),
                })
            if step == n_steps:
                break
            k1 = self.rhs(state, t, cfg["amp"], cfg["omega"])
            k2 = self.rhs(state + 0.5 * self.dt * k1, t + 0.5 * self.dt, cfg["amp"], cfg["omega"])
            k3 = self.rhs(state + 0.5 * self.dt * k2, t + 0.5 * self.dt, cfg["amp"], cfg["omega"])
            k4 = self.rhs(state + self.dt * k3, t + self.dt, cfg["amp"], cfg["omega"])
            state = state + (self.dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        return rows

    def _rows_to_observations(
        self, stage_id: str, rows: List[Dict[str, float]], idx_start: int, t_start: int,
        reveal_target: bool,
    ) -> List[Observation]:
        observations: List[Observation] = []
        for offset, row in enumerate(rows):
            clean_target = self.acceleration(row["x"], row["v"], row["force"])
            target = float(clean_target + self.rng.normal(0.0, self.noise_std)) if reveal_target else None
            observations.append(
                Observation(
                    idx=idx_start + offset,
                    t=t_start + offset,
                    strain=float(row["x"]),
                    direction=0,
                    stress=float(target) if target is not None else 0.0,
                    stage_id=stage_id,
                    stage_label=DUFFING_STAGE_LABELS[stage_id],
                    observables=dict(row),
                    target_value=target,
                )
            )
        return observations

    def preview_stage(self, stage_id: str, idx_start: int = 0, t_start: int = 0) -> List[Observation]:
        return self._rows_to_observations(
            stage_id, self.integrate_inputs(stage_id), idx_start, t_start, reveal_target=False,
        )

    def collect_stage(self, stage_id: str, idx_start: int = 0, t_start: int = 0) -> List[Observation]:
        observations = self._rows_to_observations(
            stage_id, self.integrate_inputs(stage_id), idx_start, t_start, reveal_target=True,
        )
        self.collected_stages[stage_id] = observations
        return observations
