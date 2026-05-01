"""
Stochastic hill-climb over DAG world models.

Each proposal is a structural edit (add feature, remove feature, swap a factor,
perturb a threshold). The candidate is refit by lstsq and scored by total MDL
bits. Proposals that decrease total bits are accepted; otherwise rejected. The
search terminates when `patience` consecutive proposals are rejected, i.e. we
are locally happy.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from dag_model import (
    DAGModel,
    Factor,
    Feature,
    K_MAX,
    MAX_FACTORS_PER_FEATURE,
    MAX_FEATURES,
    VarSpec,
)


@dataclass
class ProposalStep:
    step: int
    operator: str
    accepted: bool
    bits_before: float
    bits_after: float
    l_model: float
    l_data: float
    k_features: int
    description: str


@dataclass
class SearchTrace:
    steps: List[ProposalStep] = field(default_factory=list)
    best_bits_history: List[float] = field(default_factory=list)
    accepted_bits_history: List[float] = field(default_factory=list)
    converged: bool = False
    stop_reason: str = ""


# ---- random sampling helpers -------------------------------------------------

def _sample_factor(var_space: Sequence[VarSpec], rng: random.Random) -> Factor:
    continuous = [v for v in var_space if v.kind == "continuous"]
    discrete = [v for v in var_space if v.kind == "discrete"]
    pool = ["Const"]
    if continuous:
        pool += ["Ident", "Pow", "IndLE", "ReLU"]
    if discrete:
        pool += ["IndEq"]
    kind = rng.choice(pool)
    if kind == "Const":
        return Factor(kind="Const")
    if kind == "Ident":
        v = rng.choice(continuous)
        return Factor(kind="Ident", var=v.name)
    if kind == "Pow":
        v = rng.choice(continuous)
        k = rng.randint(1, K_MAX)
        return Factor(kind="Pow", var=v.name, k=k)
    if kind == "IndEq":
        v = rng.choice(discrete)
        a = int(rng.choice(v.values))
        return Factor(kind="IndEq", var=v.name, a=a)
    if kind in ("IndLE", "ReLU"):
        v = rng.choice(continuous)
        t = rng.uniform(v.lo, v.hi)
        return Factor(kind=kind, var=v.name, threshold=t)
    raise RuntimeError("unreachable")


def _sample_feature(var_space: Sequence[VarSpec], rng: random.Random) -> Feature:
    n = rng.randint(1, MAX_FACTORS_PER_FEATURE)
    factors = tuple(_sample_factor(var_space, rng) for _ in range(n))
    return Feature.make(factors)


def _is_const_feature(feat: Feature) -> bool:
    return len(feat.factors) == 1 and feat.factors[0].kind == "Const"


def _perturb_threshold(factor: Factor, var_space: Sequence[VarSpec], rng: random.Random) -> Factor:
    v = next(vs for vs in var_space if vs.name == factor.var)
    span = max(1e-6, v.hi - v.lo)
    new = float(factor.threshold) + rng.gauss(0, span * 0.15)
    new = max(v.lo, min(v.hi, new))
    return Factor(kind=factor.kind, var=factor.var, threshold=new)


# ---- proposal operators ------------------------------------------------------

def propose_add_feature(model: DAGModel, rng: random.Random) -> Tuple[Optional[DAGModel], str]:
    if len(model.features) >= MAX_FEATURES:
        return None, "add_feature skipped (MAX_FEATURES)"
    feat = _sample_feature(model.var_space, rng)
    if _is_const_feature(feat):
        return None, "add_feature skipped (constant already represented by intercept)"
    if any(f.canonical_key() == feat.canonical_key() for f in model.features):
        return None, "add_feature skipped (duplicate)"
    new = model.clone()
    new.features = list(model.features) + [feat]
    return new, f"add feature {feat.label()}"


def propose_remove_feature(model: DAGModel, rng: random.Random) -> Tuple[Optional[DAGModel], str]:
    if len(model.features) == 0:
        return None, "remove_feature skipped (empty)"
    idx = rng.randrange(len(model.features))
    removed = model.features[idx]
    new = model.clone()
    new.features = [f for i, f in enumerate(model.features) if i != idx]
    return new, f"remove feature {removed.label()}"


def propose_swap_factor(model: DAGModel, rng: random.Random) -> Tuple[Optional[DAGModel], str]:
    if not model.features:
        return None, "swap_factor skipped (no features)"
    fi = rng.randrange(len(model.features))
    feat = model.features[fi]
    if not feat.factors:
        return None, "swap_factor skipped (empty feature)"
    j = rng.randrange(len(feat.factors))
    new_factor = _sample_factor(model.var_space, rng)
    new_factors = list(feat.factors)
    old_label = new_factors[j].label()
    new_factors[j] = new_factor
    new_feat = Feature.make(new_factors)
    if _is_const_feature(new_feat):
        return None, "swap_factor skipped (constant already represented by intercept)"
    new = model.clone()
    features = list(model.features)
    features[fi] = new_feat
    new.features = features
    return new, f"swap {old_label} -> {new_factor.label()}"


def propose_perturb_threshold(model: DAGModel, rng: random.Random) -> Tuple[Optional[DAGModel], str]:
    candidates: List[Tuple[int, int]] = []
    for fi, feat in enumerate(model.features):
        for j, f in enumerate(feat.factors):
            if f.threshold is not None:
                candidates.append((fi, j))
    if not candidates:
        return None, "perturb_threshold skipped (no thresholds)"
    fi, j = rng.choice(candidates)
    feat = model.features[fi]
    old = feat.factors[j]
    new_factor = _perturb_threshold(old, model.var_space, rng)
    new_factors = list(feat.factors)
    new_factors[j] = new_factor
    new_feat = Feature.make(new_factors)
    new = model.clone()
    features = list(model.features)
    features[fi] = new_feat
    new.features = features
    return new, (
        f"threshold {old.threshold:.3g} -> {new_factor.threshold:.3g} on {old.kind}({old.var})"
    )


PROPOSAL_OPERATORS = [
    ("add_feature", propose_add_feature, 0.35),
    ("remove_feature", propose_remove_feature, 0.20),
    ("swap_factor", propose_swap_factor, 0.25),
    ("perturb_threshold", propose_perturb_threshold, 0.20),
]


def _choose_operator(rng: random.Random) -> Tuple[str, callable]:
    r = rng.random()
    acc = 0.0
    for name, fn, p in PROPOSAL_OPERATORS:
        acc += p
        if r <= acc:
            return name, fn
    return PROPOSAL_OPERATORS[-1][0], PROPOSAL_OPERATORS[-1][1]


# ---- inner search ------------------------------------------------------------

def hill_climb(
    start: DAGModel,
    X: Dict[str, np.ndarray],
    y: np.ndarray,
    rng: random.Random,
    max_steps: int = 80,
    patience: int = 18,
    seed_features: Optional[Sequence[Feature]] = None,
) -> Tuple[DAGModel, SearchTrace]:
    start = start.clone()
    start.fit(X, y)
    current_bits = start.total_bits(X, y)["L_total"]
    best_model = start.clone()
    best_bits = current_bits

    trace = SearchTrace()
    trace.best_bits_history.append(best_bits)
    trace.accepted_bits_history.append(current_bits)
    rejects = 0

    # --- LLM seed phase: front-load Builder-proposed features as add_feature trials ---
    seed_offset = 0
    if seed_features:
        for feat in seed_features:
            seed_offset += 1
            bits_before = best_bits
            accepted = False
            if len(best_model.features) >= MAX_FEATURES:
                trace.steps.append(ProposalStep(
                    step=seed_offset, operator="llm_seed", accepted=False,
                    bits_before=bits_before, bits_after=bits_before,
                    l_model=float("nan"), l_data=float("nan"),
                    k_features=len(best_model.features),
                    description=f"seed add {feat.label()} skipped (MAX_FEATURES)",
                ))
            elif any(f.canonical_key() == feat.canonical_key() for f in best_model.features):
                trace.steps.append(ProposalStep(
                    step=seed_offset, operator="llm_seed", accepted=False,
                    bits_before=bits_before, bits_after=bits_before,
                    l_model=float("nan"), l_data=float("nan"),
                    k_features=len(best_model.features),
                    description=f"seed add {feat.label()} skipped (duplicate)",
                ))
            else:
                candidate = best_model.clone()
                candidate.features = list(best_model.features) + [feat]
                candidate.fit(X, y)
                if not candidate.fit_valid:
                    trace.steps.append(ProposalStep(
                        step=seed_offset, operator="llm_seed", accepted=False,
                        bits_before=bits_before, bits_after=bits_before,
                        l_model=float("nan"), l_data=float("nan"),
                        k_features=len(best_model.features),
                        description=f"seed add {feat.label()} [fit failed]",
                    ))
                else:
                    report = candidate.total_bits(X, y)
                    bits_after = report["L_total"]
                    if bits_after + 1e-9 < best_bits:
                        best_model = candidate
                        best_bits = bits_after
                        accepted = True
                    trace.steps.append(ProposalStep(
                        step=seed_offset, operator="llm_seed", accepted=accepted,
                        bits_before=bits_before, bits_after=bits_after,
                        l_model=report["L_model"], l_data=report["L_data"],
                        k_features=report["k_features"],
                        description=f"seed add {feat.label()}",
                    ))
            trace.best_bits_history.append(best_bits)
            trace.accepted_bits_history.append(
                best_bits if accepted else trace.accepted_bits_history[-1]
            )

    for i in range(1, max_steps + 1):
        step = seed_offset + i
        name, fn = _choose_operator(rng)
        proposal, desc = fn(best_model, rng)
        bits_before = best_bits
        accepted = False
        if proposal is None:
            step_info = ProposalStep(
                step=step, operator=name, accepted=False,
                bits_before=bits_before, bits_after=bits_before,
                l_model=float("nan"), l_data=float("nan"),
                k_features=len(best_model.features),
                description=desc,
            )
            trace.steps.append(step_info)
            rejects += 1
        else:
            proposal.fit(X, y)
            if not proposal.fit_valid:
                step_info = ProposalStep(
                    step=step, operator=name, accepted=False,
                    bits_before=bits_before, bits_after=bits_before,
                    l_model=float("nan"), l_data=float("nan"),
                    k_features=len(best_model.features),
                    description=desc + " [fit failed]",
                )
                trace.steps.append(step_info)
                rejects += 1
            else:
                report = proposal.total_bits(X, y)
                bits_after = report["L_total"]
                if bits_after + 1e-9 < best_bits:
                    best_model = proposal
                    best_bits = bits_after
                    accepted = True
                    rejects = 0
                else:
                    rejects += 1
                trace.steps.append(ProposalStep(
                    step=step, operator=name, accepted=accepted,
                    bits_before=bits_before, bits_after=bits_after,
                    l_model=report["L_model"], l_data=report["L_data"],
                    k_features=report["k_features"],
                    description=desc,
                ))
        trace.best_bits_history.append(best_bits)
        trace.accepted_bits_history.append(best_bits if accepted else trace.accepted_bits_history[-1])
        if rejects >= patience:
            trace.converged = True
            trace.stop_reason = f"patience exhausted after {rejects} consecutive rejections"
            break
    else:
        trace.stop_reason = f"step budget exhausted ({max_steps})"

    return best_model, trace
