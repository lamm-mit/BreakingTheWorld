"""
DAG-based world model with MDL (minimum description length) scoring.

A world model is a linear combination of features:

    y_hat = beta_0 + sum_i beta_i * f_i(x)

where each feature f_i is a product of primitive factors, and each factor is
one of:

    Const        -> 1
    Ident(v)     -> v
    Pow(v, k)    -> v ** k                (k integer, 1..K_MAX)
    IndEq(v, a)  -> 1 if v == a else 0    (v must be discrete)
    IndLE(v, t)  -> 1 if v <= t else 0    (v must be continuous)
    ReLU(v, t)   -> max(v - t, 0)         (v must be continuous)

The model complexity L_model (bits) and the data misfit L_data (bits) are on
the same footing. Total description length is L_model + L_data; minimising it
is the principled tradeoff between parsimony and fit.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

K_MAX = 4
THRESHOLD_BITS = 8
COEFFICIENT_BITS = 16
MAX_FACTORS_PER_FEATURE = 4
MAX_FEATURES = 10

FACTOR_TYPES = ("Const", "Ident", "Pow", "IndEq", "IndLE", "ReLU")
_FACTOR_TYPE_BITS = math.log2(len(FACTOR_TYPES))


@dataclass(frozen=True)
class VarSpec:
    name: str
    kind: str  # "continuous" or "discrete"
    values: Tuple[int, ...] = ()  # populated for discrete vars
    lo: float = 0.0  # populated for continuous vars
    hi: float = 1.0


@dataclass(frozen=True)
class Factor:
    """One primitive factor; evaluated on a data batch returns a 1-D ndarray."""
    kind: str
    var: Optional[str] = None
    k: Optional[int] = None       # for Pow
    a: Optional[int] = None       # for IndEq
    threshold: Optional[float] = None  # for IndLE, ReLU

    def evaluate(self, X: Dict[str, np.ndarray]) -> np.ndarray:
        if self.kind == "Const":
            any_key = next(iter(X))
            return np.ones_like(X[any_key], dtype=float)
        v = X[self.var]
        if self.kind == "Ident":
            return v.astype(float)
        if self.kind == "Pow":
            return v.astype(float) ** int(self.k)
        if self.kind == "IndEq":
            return (v == int(self.a)).astype(float)
        if self.kind == "IndLE":
            return (v <= float(self.threshold)).astype(float)
        if self.kind == "ReLU":
            return np.maximum(v.astype(float) - float(self.threshold), 0.0)
        raise ValueError(f"unknown factor kind: {self.kind}")

    def label(self) -> str:
        if self.kind == "Const":
            return "1"
        if self.kind == "Ident":
            return self.var
        if self.kind == "Pow":
            return f"{self.var}^{self.k}"
        if self.kind == "IndEq":
            return f"[{self.var}={self.a}]"
        if self.kind == "IndLE":
            return f"[{self.var}<={self.threshold:.3g}]"
        if self.kind == "ReLU":
            return f"relu({self.var}-{self.threshold:.3g})"
        return "?"

    def description_bits(self, var_space: Sequence[VarSpec]) -> float:
        bits = _FACTOR_TYPE_BITS
        if self.kind == "Const":
            return bits
        n_vars = max(1, len(var_space))
        bits += math.log2(n_vars)
        if self.kind == "Ident":
            return bits
        if self.kind == "Pow":
            return bits + math.log2(K_MAX)
        if self.kind == "IndEq":
            spec = next(v for v in var_space if v.name == self.var)
            return bits + math.log2(max(2, len(spec.values)))
        if self.kind in ("IndLE", "ReLU"):
            return bits + THRESHOLD_BITS
        raise ValueError(f"unknown factor kind: {self.kind}")


IDEMPOTENT_FACTOR_KINDS = {"Const", "IndEq", "IndLE"}


def normalize_factors(factors: Sequence[Factor]) -> Tuple[Factor, ...]:
    """Drop trivial Const factors, dedupe idempotent factors, sort by a canonical key."""
    seen: Dict[Tuple, Factor] = {}
    out: List[Factor] = []
    for f in factors:
        if f.kind == "Const":
            continue
        if f.kind in IDEMPOTENT_FACTOR_KINDS:
            key = (f.kind, f.var, f.a, round(f.threshold, 6) if f.threshold is not None else None)
            if key in seen:
                continue
            seen[key] = f
        out.append(f)
    if not out:
        out = [Factor(kind="Const")]
    out.sort(key=lambda f: (f.kind, f.var or "", f.k or 0, f.a if f.a is not None else 0,
                            round(f.threshold, 6) if f.threshold is not None else 0.0))
    return tuple(out)


@dataclass(frozen=True)
class Feature:
    """A feature is a product of factors."""
    factors: Tuple[Factor, ...]

    @staticmethod
    def make(factors: Sequence[Factor]) -> "Feature":
        return Feature(factors=normalize_factors(factors))

    def evaluate(self, X: Dict[str, np.ndarray]) -> np.ndarray:
        if not self.factors:
            any_key = next(iter(X))
            return np.ones_like(X[any_key], dtype=float)
        out = self.factors[0].evaluate(X)
        for f in self.factors[1:]:
            out = out * f.evaluate(X)
        return out

    def label(self) -> str:
        if not self.factors:
            return "1"
        return " * ".join(f.label() for f in self.factors)

    def description_bits(self, var_space: Sequence[VarSpec]) -> float:
        n_factors = len(self.factors)
        length_bits = math.log2(MAX_FACTORS_PER_FEATURE + 1)
        return length_bits + sum(f.description_bits(var_space) for f in self.factors)

    def canonical_key(self) -> Tuple:
        return tuple(sorted(
            (f.kind, f.var, f.k, f.a, round(f.threshold, 6) if f.threshold is not None else None)
            for f in self.factors
        ))


@dataclass
class DAGModel:
    features: List[Feature]
    var_space: Tuple[VarSpec, ...]
    target: str
    coefficients: np.ndarray = field(default_factory=lambda: np.zeros(0))
    intercept: float = 0.0
    sigma2: float = 1.0
    fit_valid: bool = False

    def evaluate(self, X: Dict[str, np.ndarray]) -> np.ndarray:
        if not self.features:
            any_key = next(iter(X))
            return np.full_like(X[any_key], self.intercept, dtype=float)
        cols = [f.evaluate(X) for f in self.features]
        design = np.column_stack(cols)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            return self.intercept + design @ self.coefficients

    def fit(self, X: Dict[str, np.ndarray], y: np.ndarray) -> "DAGModel":
        n = len(y)
        if not self.features:
            self.intercept = float(np.mean(y))
            self.coefficients = np.zeros(0)
            residuals = y - self.intercept
        else:
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                cols = [f.evaluate(X) for f in self.features]
                design_no_int = np.column_stack(cols)
                if not np.all(np.isfinite(design_no_int)):
                    self.fit_valid = False
                    return self
                ones = np.ones((n, 1))
                design = np.concatenate([ones, design_no_int], axis=1)
                try:
                    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
                    yhat = design @ beta
                    if not np.all(np.isfinite(yhat)):
                        raise np.linalg.LinAlgError("non-finite prediction")
                except np.linalg.LinAlgError:
                    self.fit_valid = False
                    return self
            self.intercept = float(beta[0])
            self.coefficients = np.asarray(beta[1:], dtype=float)
            residuals = y - yhat
        rss = float(np.sum(residuals ** 2))
        self.sigma2 = max(rss / max(1, n), 1e-9)
        self.fit_valid = True
        return self

    def description_bits(self) -> Tuple[float, float, float]:
        length_bits = math.log2(MAX_FEATURES + 1)
        feat_bits = sum(f.description_bits(self.var_space) for f in self.features)
        coef_bits = (len(self.features) + 1) * COEFFICIENT_BITS  # features + intercept
        structure_bits = length_bits + feat_bits
        return structure_bits, coef_bits, structure_bits + coef_bits

    def data_bits(self, X: Dict[str, np.ndarray], y: np.ndarray) -> float:
        n = len(y)
        yhat = self.evaluate(X)
        rss = float(np.sum((y - yhat) ** 2))
        sigma2 = max(rss / max(1, n), 1e-9)
        gaussian_nat = 0.5 * n * math.log(2 * math.pi * sigma2) + rss / (2 * sigma2)
        return gaussian_nat / math.log(2)

    def total_bits(self, X: Dict[str, np.ndarray], y: np.ndarray) -> Dict[str, float]:
        structure_bits, coef_bits, model_bits = self.description_bits()
        l_data = self.data_bits(X, y)
        total = model_bits + l_data
        yhat = self.evaluate(X)
        rss = float(np.sum((y - yhat) ** 2))
        tss = float(np.sum((y - np.mean(y)) ** 2))
        r2 = 1.0 - rss / tss if tss > 1e-12 else 1.0
        return {
            "L_structure": structure_bits,
            "L_params": coef_bits,
            "L_model": model_bits,
            "L_data": l_data,
            "L_total": total,
            "rss": rss,
            "r2": r2,
            "rmse": math.sqrt(rss / max(1, len(y))),
            "n": len(y),
            "k_features": len(self.features),
        }

    def clone(self) -> "DAGModel":
        return DAGModel(
            features=list(self.features),
            var_space=self.var_space,
            target=self.target,
            coefficients=self.coefficients.copy(),
            intercept=self.intercept,
            sigma2=self.sigma2,
            fit_valid=self.fit_valid,
        )

    def canonical_key(self) -> Tuple:
        return tuple(sorted(f.canonical_key() for f in self.features))

    def equation_lines(self) -> List[str]:
        parts: List[str] = [f"{self.intercept:+.4g}"]
        for coef, feat in zip(self.coefficients, self.features):
            parts.append(f"{coef:+.4g} * {feat.label()}")
        return [f"{self.target} = " + " ".join(parts)]

    # --- DAG graph spec for visualization -------------------------------------
    def graph_spec(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        nodes: List[Dict[str, Any]] = []
        edges: List[Dict[str, Any]] = []
        seen: Dict[str, Dict[str, Any]] = {}

        def add(node_id: str, label: str, kind: str) -> None:
            if node_id in seen:
                return
            node = {"id": node_id, "label": label, "kind": kind}
            nodes.append(node)
            seen[node_id] = node

        for v in self.var_space:
            add(f"in:{v.name}", v.name, "observable")
        add(f"out:{self.target}", self.target, "target")

        for fi, feat in enumerate(self.features):
            feat_id = f"feat:{fi}"
            add(feat_id, feat.label() or "1", "feature")
            for fj, factor in enumerate(feat.factors):
                factor_id = f"fact:{fi}:{fj}:{factor.label()}"
                add(factor_id, factor.label(), "factor")
                if factor.var is not None:
                    edges.append({"source": f"in:{factor.var}", "target": factor_id})
                edges.append({"source": factor_id, "target": feat_id})
            edges.append({"source": feat_id, "target": f"out:{self.target}"})
        if not self.features:
            for v in self.var_space:
                edges.append({"source": f"in:{v.name}", "target": f"out:{self.target}"})
        return nodes, edges
