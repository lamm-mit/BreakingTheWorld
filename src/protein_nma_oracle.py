from __future__ import annotations

import math
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from discovery_data import DiscoveryDataset, Observation


AMINO_ACID_3TO1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "SEC": "U", "PYL": "O",
}

HYDROPHOBIC = set("AILMFWYV")
CHARGED = set("DEKRH")
POLAR = set("NQSTCY")


DEFAULT_PROTEIN_STAGES: Dict[str, List[str]] = {
    "stage_0_compact_single_domain": ["1ubq:A", "1crn:A"],
    "stage_1_terminal_flexibility": ["1aar:A", "2ci2:I"],
    "stage_2_hinge_domain_motion": ["4ake:A", "1ake:A"],
    "stage_3_validation_mixed": ["1p38:A", "1hel:A"],
}

DEFAULT_STAGE_LABELS = {
    "stage_0_compact_single_domain": "Compact single-domain proteins",
    "stage_1_terminal_flexibility": "Proteins with exposed/flexible termini",
    "stage_2_hinge_domain_motion": "Proteins with hinge or domain-scale motion",
    "stage_3_validation_mixed": "Mixed validation proteins",
}

DEFAULT_PROTOCOL_DESCRIPTIONS = {
    "stage_0_compact_single_domain": (
        "Initial revealed set of compact proteins. A uniform C-alpha elastic-network "
        "world model should capture a substantial fraction of normalized residue "
        "B-factor variation from contact topology and low-frequency GNM modes."
    ),
    "stage_1_terminal_flexibility": (
        "Reveal proteins where termini and boundary regions are expected to be more "
        "mobile than a uniform contact-network model may predict. This probes whether "
        "the ontology needs explicit boundary/disorder features."
    ),
    "stage_2_hinge_domain_motion": (
        "Reveal proteins with stronger collective or hinge-like motions. This probes "
        "whether local fluctuation magnitude alone is sufficient, or whether slow-mode "
        "shape and hinge scores are needed."
    ),
    "stage_3_validation_mixed": (
        "Holdout-style mixed validation stage used to test whether the revised world "
        "model generalizes beyond the slices that forced earlier revisions."
    ),
}

SYSTEM_DESCRIPTION = (
    "Protein flexibility discovery from real PDB structures. Each residue is represented "
    "as a node in a C-alpha contact network. A Gaussian Network Model (GNM) computes "
    "normal-mode fluctuation features from the structure. The target is the residue's "
    "experimental crystallographic B-factor normalized within each protein chain. The "
    "Builder attempts to discover a compact explanatory law relating graph-normal-mode "
    "features, boundary features, and residue context to experimental flexibility."
)

OBSERVABLE_DESCRIPTIONS = {
    "gnm_fluct_z": "Per-protein z-score of residue mean-square fluctuation from the GNM pseudoinverse.",
    "gnm_fluct_log_z": "Per-protein z-score of log-transformed GNM fluctuation.",
    "contact_degree_z": "Per-protein z-score of C-alpha contact degree in the elastic network.",
    "contact_degree_raw": "Raw number of C-alpha neighbors within the GNM cutoff.",
    "terminal_exposure": "One at a chain terminus and near zero near the middle of the chain.",
    "res_index_norm": "Residue index normalized from 0 at N-terminus to 1 at C-terminus.",
    "mode1_abs_z": "Per-protein z-score of absolute amplitude in the slowest nonzero GNM mode.",
    "hinge_score_z": "Per-protein z-score of local curvature in the slowest nonzero GNM mode.",
    "chain_break_proximity": "Proximity to a numbering or coordinate chain break, if detected.",
    "is_terminal": "Discrete flag for residues within the terminal window.",
    "is_gly": "Discrete flag for glycine residues.",
    "is_pro": "Discrete flag for proline residues.",
    "is_hydrophobic": "Discrete flag for hydrophobic amino acids.",
    "is_charged": "Discrete flag for charged amino acids.",
    "is_polar": "Discrete flag for polar amino acids.",
}

OBSERVABLE_TYPES = {
    "gnm_fluct_z": "continuous",
    "gnm_fluct_log_z": "continuous",
    "contact_degree_z": "continuous",
    "contact_degree_raw": "continuous",
    "terminal_exposure": "continuous",
    "res_index_norm": "continuous",
    "mode1_abs_z": "continuous",
    "hinge_score_z": "continuous",
    "chain_break_proximity": "continuous",
    "is_terminal": "discrete",
    "is_gly": "discrete",
    "is_pro": "discrete",
    "is_hydrophobic": "discrete",
    "is_charged": "discrete",
    "is_polar": "discrete",
}


@dataclass(frozen=True)
class PDBSpec:
    """A structure input, either `1ubq:A` or `/path/to/file.pdb:A`."""

    source: str
    chain: Optional[str] = None

    @property
    def label(self) -> str:
        return f"{self.source}:{self.chain}" if self.chain else self.source

    @property
    def is_file(self) -> bool:
        return Path(self.source).exists()

    @property
    def pdb_id(self) -> str:
        return Path(self.source).stem if self.is_file else self.source.lower()


@dataclass
class ResidueCA:
    pdb_id: str
    chain: str
    resseq: int
    icode: str
    resname: str
    aa: str
    coord: np.ndarray
    bfactor: float

    @property
    def residue_id(self) -> str:
        suffix = self.icode.strip()
        return f"{self.chain}:{self.resseq}{suffix}:{self.aa}"


@dataclass
class ProteinChainFeatures:
    spec: PDBSpec
    residues: List[ResidueCA]
    cutoff: float
    observables: List[Dict[str, float]]
    target_z: np.ndarray


def parse_pdb_spec(text: str) -> PDBSpec:
    text = text.strip()
    if not text:
        raise ValueError("empty PDB spec")
    path = Path(text)
    if path.exists():
        return PDBSpec(source=str(path), chain=None)
    if ":" in text:
        source, chain = text.rsplit(":", 1)
        return PDBSpec(source=source.strip(), chain=chain.strip() or None)
    return PDBSpec(source=text, chain=None)


def parse_stage_spec(text: str) -> Tuple[str, List[PDBSpec]]:
    if "=" not in text:
        raise ValueError(f"stage spec must look like stage_id=pdb:chain,pdb:chain; got {text!r}")
    stage_id, values = text.split("=", 1)
    specs = [parse_pdb_spec(v) for v in values.split(",") if v.strip()]
    if not specs:
        raise ValueError(f"stage {stage_id!r} has no PDB specs")
    return stage_id.strip(), specs


def fetch_pdb(spec: PDBSpec, cache_dir: Path, timeout: float = 30.0) -> Path:
    if spec.is_file:
        return Path(spec.source)
    cache_dir.mkdir(parents=True, exist_ok=True)
    pdb_id = spec.pdb_id.lower()
    out = cache_dir / f"{pdb_id}.pdb"
    if out.exists() and out.stat().st_size > 0:
        return out
    url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"
    with urllib.request.urlopen(url, timeout=timeout) as response:
        data = response.read()
    if not data:
        raise RuntimeError(f"downloaded empty PDB file for {spec.label}")
    out.write_bytes(data)
    return out


def parse_ca_residues(path: Path, spec: PDBSpec, min_residues: int = 20) -> List[ResidueCA]:
    chains: Dict[str, List[ResidueCA]] = {}
    seen = set()
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            atom = line[12:16].strip()
            altloc = line[16:17]
            if atom != "CA" or altloc not in (" ", "A"):
                continue
            resname = line[17:20].strip().upper()
            aa = AMINO_ACID_3TO1.get(resname)
            if aa is None:
                continue
            chain = line[21:22].strip() or "_"
            if spec.chain is not None and chain != spec.chain:
                continue
            try:
                resseq = int(line[22:26])
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                bfactor = float(line[60:66])
            except ValueError:
                continue
            icode = line[26:27]
            key = (chain, resseq, icode)
            if key in seen:
                continue
            seen.add(key)
            residue = ResidueCA(
                pdb_id=spec.pdb_id,
                chain=chain,
                resseq=resseq,
                icode=icode,
                resname=resname,
                aa=aa,
                coord=np.asarray([x, y, z], dtype=float),
                bfactor=bfactor,
            )
            chains.setdefault(chain, []).append(residue)
    if spec.chain is not None:
        residues = chains.get(spec.chain, [])
        if len(residues) < min_residues:
            raise ValueError(f"{spec.label} has only {len(residues)} parsed C-alpha residues")
        return residues
    candidates = [rows for rows in chains.values() if len(rows) >= min_residues]
    if not candidates:
        raise ValueError(f"{spec.label} has no protein chain with at least {min_residues} C-alpha residues")
    return max(candidates, key=len)


def zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    std = float(np.std(values))
    if std < 1e-12:
        return np.zeros_like(values, dtype=float)
    return (values - float(np.mean(values))) / std


def pairwise_distances(coords: np.ndarray) -> np.ndarray:
    delta = coords[:, None, :] - coords[None, :, :]
    return np.sqrt(np.sum(delta * delta, axis=2))


def gnm_modes(coords: np.ndarray, cutoff: float, n_modes: int = 20) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    distances = pairwise_distances(coords)
    contact = (distances <= cutoff) & (distances > 1e-9)
    kirchhoff = -contact.astype(float)
    np.fill_diagonal(kirchhoff, np.sum(contact, axis=1))
    eigvals, eigvecs = np.linalg.eigh(kirchhoff)
    valid = eigvals > 1e-8
    eigvals = eigvals[valid]
    eigvecs = eigvecs[:, valid]
    if eigvals.size == 0:
        raise ValueError("GNM has no nonzero modes; contact network may be disconnected or empty")
    keep = min(n_modes, eigvals.size)
    eigvals = eigvals[:keep]
    eigvecs = eigvecs[:, :keep]
    return contact.astype(float), eigvals, eigvecs


def mode_curvature(mode: np.ndarray) -> np.ndarray:
    out = np.zeros_like(mode, dtype=float)
    if mode.size < 3:
        return out
    out[1:-1] = np.abs(mode[:-2] - 2.0 * mode[1:-1] + mode[2:])
    out[0] = out[1]
    out[-1] = out[-2]
    return out


def chain_break_proximity(residues: Sequence[ResidueCA], coords: np.ndarray) -> np.ndarray:
    n = len(residues)
    break_positions: List[int] = []
    for i in range(n - 1):
        numbering_gap = residues[i + 1].resseq - residues[i].resseq > 1
        coord_gap = float(np.linalg.norm(coords[i + 1] - coords[i])) > 4.5
        if numbering_gap or coord_gap:
            break_positions.extend([i, i + 1])
    if not break_positions:
        return np.zeros(n, dtype=float)
    idx = np.arange(n)
    dist = np.min(np.abs(idx[:, None] - np.asarray(break_positions)[None, :]), axis=1)
    return 1.0 / (1.0 + dist.astype(float))


def compute_chain_features(
    spec: PDBSpec,
    pdb_path: Path,
    cutoff: float = 10.0,
    n_modes: int = 20,
    terminal_window: int = 5,
    min_residues: int = 20,
) -> ProteinChainFeatures:
    residues = parse_ca_residues(pdb_path, spec, min_residues=min_residues)
    coords = np.vstack([r.coord for r in residues])
    contact, eigvals, eigvecs = gnm_modes(coords, cutoff=cutoff, n_modes=n_modes)
    inv_eigs = 1.0 / eigvals
    fluctuations = np.sum((eigvecs * eigvecs) * inv_eigs[None, :], axis=1)
    degree = np.sum(contact, axis=1)
    slowest = eigvecs[:, 0]
    hinge = mode_curvature(slowest)
    bfactor = np.asarray([r.bfactor for r in residues], dtype=float)

    n = len(residues)
    idx = np.arange(n, dtype=float)
    denom = max(1.0, float(n - 1))
    nearest_terminus = np.minimum(idx, (n - 1) - idx)
    terminal_exposure = 1.0 - np.clip(nearest_terminus / max(1.0, n / 2.0), 0.0, 1.0)
    break_proximity = chain_break_proximity(residues, coords)

    gnm_fluct_z = zscore(fluctuations)
    gnm_fluct_log_z = zscore(np.log(np.maximum(fluctuations, 1e-12)))
    degree_z = zscore(degree)
    mode1_abs_z = zscore(np.abs(slowest))
    hinge_z = zscore(hinge)
    target_z = zscore(bfactor)

    observations: List[Dict[str, float]] = []
    for i, residue in enumerate(residues):
        aa = residue.aa
        observations.append({
            "gnm_fluct_z": float(gnm_fluct_z[i]),
            "gnm_fluct_log_z": float(gnm_fluct_log_z[i]),
            "contact_degree_z": float(degree_z[i]),
            "contact_degree_raw": float(degree[i]),
            "terminal_exposure": float(terminal_exposure[i]),
            "res_index_norm": float(idx[i] / denom),
            "mode1_abs_z": float(mode1_abs_z[i]),
            "hinge_score_z": float(hinge_z[i]),
            "chain_break_proximity": float(break_proximity[i]),
            "is_terminal": int(nearest_terminus[i] < terminal_window),
            "is_gly": int(aa == "G"),
            "is_pro": int(aa == "P"),
            "is_hydrophobic": int(aa in HYDROPHOBIC),
            "is_charged": int(aa in CHARGED),
            "is_polar": int(aa in POLAR),
        })
    return ProteinChainFeatures(
        spec=spec,
        residues=residues,
        cutoff=cutoff,
        observables=observations,
        target_z=target_z,
    )


def default_stage_specs() -> Dict[str, List[PDBSpec]]:
    return {
        stage: [parse_pdb_spec(item) for item in specs]
        for stage, specs in DEFAULT_PROTEIN_STAGES.items()
    }


def make_stage_prerequisites(stage_order: Sequence[str]) -> Dict[str, List[str]]:
    prerequisites: Dict[str, List[str]] = {}
    for i, stage_id in enumerate(stage_order[1:], start=1):
        prerequisites[stage_id] = [stage_order[i - 1]]
    return prerequisites


def build_protein_flex_dataset(
    stage_specs: Dict[str, List[PDBSpec]],
    cache_dir: Path,
    cutoff: float = 10.0,
    n_modes: int = 20,
    terminal_window: int = 5,
    min_residues: int = 20,
    initial_stage_ids: Optional[Sequence[str]] = None,
) -> DiscoveryDataset:
    observations: List[Observation] = []
    requested_stage_order = list(stage_specs.keys())
    stage_order: List[str] = []
    idx = 0
    t = 0
    skipped: List[str] = []
    for stage_id in requested_stage_order:
        stage_start = len(observations)
        stage_label = DEFAULT_STAGE_LABELS.get(stage_id, stage_id.replace("_", " ").title())
        for spec in stage_specs[stage_id]:
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
                skipped.append(f"{spec.label}: {exc}")
                continue
            for residue_i, (residue, observables, target) in enumerate(
                zip(features.residues, features.observables, features.target_z)
            ):
                observations.append(
                    Observation(
                        idx=idx,
                        t=t,
                        strain=float(observables["gnm_fluct_z"]),
                        direction=int(observables["is_terminal"]),
                        stress=float(target),
                        stage_id=stage_id,
                        stage_label=stage_label,
                        observables={
                            **observables,
                            # Encoded metadata for stratified plots/debugging. The DAG can use
                            # only numeric variables, so these do not leak labels into fitting.
                        },
                        target_value=float(target),
                    )
                )
                idx += 1
                t += 1
        if len(observations) > stage_start:
            stage_order.append(stage_id)
    if not observations:
        detail = "\n".join(skipped) if skipped else "no structures specified"
        raise RuntimeError(f"no protein observations could be built:\n{detail}")
    if initial_stage_ids:
        missing_initial = [sid for sid in initial_stage_ids if sid not in stage_order]
        if missing_initial:
            raise RuntimeError(
                "initial stage has no built observations: " + ", ".join(missing_initial)
            )

    source_lines = [
        f"{stage}={','.join(spec.label for spec in specs)}"
        for stage, specs in stage_specs.items()
    ]
    if skipped:
        source_lines.append("skipped=" + " | ".join(skipped))

    active_observables = dict(OBSERVABLE_DESCRIPTIONS)
    active_types = dict(OBSERVABLE_TYPES)
    dropped_constants: List[str] = []
    for name in list(active_observables):
        values = [obs.observables.get(name) for obs in observations if name in obs.observables]
        if not values:
            dropped_constants.append(name)
            continue
        try:
            numeric = np.asarray(values, dtype=float)
            if float(np.max(numeric) - np.min(numeric)) <= 1e-12:
                dropped_constants.append(name)
        except (TypeError, ValueError):
            if len({str(v) for v in values}) <= 1:
                dropped_constants.append(name)
    for name in dropped_constants:
        active_observables.pop(name, None)
        active_types.pop(name, None)
        for obs in observations:
            obs.observables.pop(name, None)
    if dropped_constants:
        source_lines.append("dropped_constant_observables=" + ",".join(sorted(dropped_constants)))

    return DiscoveryDataset(
        observations=observations,
        stage_order=stage_order,
        initial_stage_ids=list(initial_stage_ids or [stage_order[0]]),
        stage_prerequisites=make_stage_prerequisites(stage_order),
        system_description=SYSTEM_DESCRIPTION,
        observable_descriptions=active_observables,
        target_description=(
            "Per-protein z-score of experimental crystallographic C-alpha B-factor. "
            "Higher values indicate residues that are experimentally more flexible "
            "or more disordered relative to the same chain."
        ),
        target_name="bfactor_z",
        observable_types=active_types,
        protocol_descriptions={
            stage: DEFAULT_PROTOCOL_DESCRIPTIONS.get(stage, "")
            for stage in stage_order
        },
        source="protein_gnm_pdb(" + "; ".join(source_lines) + ")",
    )


def stage_counts(dataset: DiscoveryDataset) -> Dict[str, int]:
    counts = {stage: 0 for stage in dataset.stage_order}
    for obs in dataset.observations:
        counts[obs.stage_id] = counts.get(obs.stage_id, 0) + 1
    return counts
