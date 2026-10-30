#!/usr/bin/env python3
"""Estimate the zero-mass GB2 distribution used by the simulation.

The zero-income probability is the unweighted empirical zero share. GB2
parameters for positive income are estimated by MARSUPWT-weighted pseudo-MLE.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.optimize import minimize
from scipy.special import betaln, digamma, expit


COMMON_DIR = Path(__file__).resolve().parent
PROJECT_DIR = COMMON_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from MLD.core import mld  # noqa: E402


DEFAULT_INPUT = PROJECT_DIR / "Data" / "pppub25_clean.csv"
DEFAULT_OUTPUT = (
    COMMON_DIR / "calibration" / "zero_mass_gb2_parameters.csv"
)


def load_calibration_sample(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load nonnegative CPS income and positive person weights."""
    data = pd.read_csv(path, usecols=["PTOTVAL", "MARSUPWT"], low_memory=False)
    income = pd.to_numeric(data["PTOTVAL"], errors="coerce").to_numpy(float)
    weight = pd.to_numeric(data["MARSUPWT"], errors="coerce").to_numpy(float)
    keep = np.isfinite(income) & np.isfinite(weight) & (income >= 0) & (weight > 0)
    income, weight = income[keep], weight[keep]
    if income.size < 2 or not np.any(income == 0) or not np.any(income > 0):
        raise ValueError("The calibration sample must contain zero and positive income.")
    return income, weight


def weighted_quantile(
    values: np.ndarray,
    weights: np.ndarray,
    probability: float,
) -> float:
    order = np.argsort(values)
    values, weights = values[order], weights[order]
    cumulative = np.cumsum(weights) / np.sum(weights)
    return float(values[np.searchsorted(cumulative, probability)])


def fit_positive_gb2(
    income: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Fit ``(a,b,p,q)`` by normalized-weight pseudo-maximum likelihood."""
    positive = income > 0
    y = income[positive]
    w = weights[positive] / np.sum(weights[positive])
    log_y = np.log(y)

    def objective(log_parameters: np.ndarray) -> tuple[float, np.ndarray]:
        log_a, log_b, log_p, log_q = log_parameters
        a, p, q = np.exp([log_a, log_p, log_q])
        log_ratio = log_y - log_b
        scaled_log_ratio = a * log_ratio
        log_one_plus_power = np.logaddexp(0.0, scaled_log_ratio)
        log_density = (
            log_a
            - log_b
            - betaln(p, q)
            + (a * p - 1.0) * log_ratio
            - (p + q) * log_one_plus_power
        )
        sigmoid = expit(scaled_log_ratio)
        score = np.column_stack(
            (
                1.0 + a * log_ratio * (p - (p + q) * sigmoid),
                a * ((p + q) * sigmoid - p),
                p
                * (
                    -digamma(p)
                    + digamma(p + q)
                    + a * log_ratio
                    - log_one_plus_power
                ),
                q
                * (
                    -digamma(q)
                    + digamma(p + q)
                    - log_one_plus_power
                ),
            )
        )
        return -float(np.dot(w, log_density)), -np.dot(w, score)

    scale = weighted_quantile(y, w, 0.5)
    starts = [
        np.log([a, scale, p, q])
        for a in (1.0, 2.0, 4.0)
        for p, q in ((0.25, 0.5), (0.5, 0.5), (1.0, 1.0))
    ]
    bounds = [
        (math.log(0.05), math.log(20.0)),
        (math.log(max(float(y.min()) * 0.01, 1e-6)), math.log(float(y.max()) * 100.0)),
        (math.log(0.01), math.log(20.0)),
        (math.log(0.01), math.log(20.0)),
    ]
    fits = [
        minimize(
            objective,
            start,
            method="L-BFGS-B",
            jac=True,
            bounds=bounds,
            options={"maxiter": 5_000, "ftol": 1e-13, "gtol": 1e-8},
        )
        for start in starts
    ]
    converged = [fit for fit in fits if fit.success and np.isfinite(fit.fun)]
    if not converged:
        messages = "; ".join(str(fit.message) for fit in fits)
        raise RuntimeError(f"GB2 optimization did not converge: {messages}")
    best = min(converged, key=lambda fit: fit.fun)
    return np.exp(best.x), float(best.fun)


def shifted_population_mld(
    pi_zero: float,
    a: float,
    b: float,
    p: float,
    q: float,
) -> tuple[float, float]:
    """Return the population MLD of ``X=Y+1`` and quadrature error."""
    if q <= 1.0 / a:
        raise ValueError("The fitted GB2 distribution does not have a finite mean.")
    mean_y = b * math.exp(
        betaln(p + 1.0 / a, q - 1.0 / a) - betaln(p, q)
    )

    log_beta = betaln(p, q)

    def log_shifted_income_integrand(logit_v: float) -> float:
        log_v = -np.logaddexp(0.0, -logit_v)
        log_one_minus_v = -np.logaddexp(0.0, logit_v)
        log_density = p * log_v + q * log_one_minus_v - log_beta
        log_y = math.log(b) + logit_v / a
        return math.exp(log_density) * float(np.logaddexp(0.0, log_y))

    positive_log_mean, error = quad(
        log_shifted_income_integrand,
        -np.inf,
        np.inf,
        epsabs=1e-10,
        epsrel=1e-10,
        limit=500,
    )
    mean_x = 1.0 + (1.0 - pi_zero) * mean_y
    expected_log_x = (1.0 - pi_zero) * positive_log_mean
    return math.log(mean_x) - expected_log_x, float(error)


def estimate(path: str | Path) -> dict[str, float | int]:
    income, weight = load_calibration_sample(path)
    pi_zero = float(np.mean(income == 0.0))
    parameters, objective = fit_positive_gb2(income, weight)
    a, b, p, q = (float(value) for value in parameters)
    population_mld, integration_error = shifted_population_mld(
        pi_zero, a, b, p, q
    )
    return {
        "n_calibration": income.size,
        "n_zero": int(np.sum(income == 0.0)),
        "n_positive": int(np.sum(income > 0.0)),
        "pi_zero": pi_zero,
        "gb2_a": a,
        "gb2_b": b,
        "gb2_p": p,
        "gb2_q": q,
        "weighted_negative_log_likelihood": objective,
        "population_mld_shifted": population_mld,
        "quadrature_absolute_error": integration_error,
        "empirical_mld_shifted": mld(income + 1.0),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    result = estimate(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(result))
        writer.writeheader()
        writer.writerow(result)
    temporary.replace(args.output)
    print(pd.DataFrame([result]).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
