"""Differentially private mean log deviation (MLD).

Exact local, global and smooth sensitivity of the MLD on [L, U]^n and the
polynomial-noise release (beta = epsilon/gamma, alpha = epsilon/(4 gamma)).
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


__all__ = [
    "MLDCalibration",
    "D_value",
    "R_value",
    "S_bounds_excluding_each",
    "calibrate_mld_release",
    "exact_Ak_mld",
    "exact_local_sensitivity_mld",
    "exact_smooth_sensitivity_mld",
    "global_sensitivity_mld",
    "h_value",
    "mld",
    "polynomial_mechanism_parameters",
    "sample_dp_mld",
    "sample_polynomial_noise",
    "sums_of_q_largest_excluding_each",
    "sums_of_q_smallest_excluding_each",
]


@dataclass(frozen=True)
class MLDCalibration:
    """Deterministic quantities used by one polynomial-noise MLD release."""

    center: float
    local_sensitivity: float
    smooth_sensitivity: float
    global_sensitivity: float
    beta: float
    alpha: float
    noise_scale: float
    n: int
    lower_bound: float
    upper_bound: float
    gamma: float
    epsilon: float


def _validated_vector(x: np.ndarray, *, minimum_size: int = 2) -> np.ndarray:
    values = np.asarray(x, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("x must be one-dimensional.")
    if values.size < minimum_size:
        raise ValueError(f"x must contain at least {minimum_size} values.")
    if np.any(~np.isfinite(values)):
        raise ValueError("x contains non-finite values.")
    return values


def _validate_bounded_data(x: np.ndarray, L: float, U: float) -> np.ndarray:
    values = _validated_vector(x)
    if not (np.isfinite(L) and np.isfinite(U) and 0.0 < L < U):
        raise ValueError("Bounds must satisfy 0 < L < U.")
    if np.any(values < L) or np.any(values > U):
        raise ValueError("All observations must lie in [L,U].")
    return values


def mld(x: np.ndarray) -> float:
    """Return ``log(mean(x)) - mean(log(x))`` for positive data."""
    values = _validated_vector(x)
    if np.any(values <= 0.0):
        raise ValueError("MLD requires strictly positive observations.")
    return float(np.log(np.mean(values)) - np.mean(np.log(values)))


def global_sensitivity_mld(n: int, L: float, U: float) -> float:
    """Return the exact global sensitivity (maximum over the two extreme corners)."""
    if n < 2:
        raise ValueError("n must be at least 2.")
    if not (np.isfinite(L) and np.isfinite(U) and 0.0 < L < U):
        raise ValueError("Bounds must satisfy 0 < L < U.")

    ratio = U / L
    rho_a = math.log((n - 1 + ratio) / n) - math.log(ratio) / n
    rho_b = math.log((n - 1 + 1.0 / ratio) / n) + math.log(ratio) / n
    return float(max(rho_a, rho_b))


def h_value(
    S: np.ndarray | float,
    t: np.ndarray | float,
    n: int,
) -> np.ndarray:
    """Evaluate ``h_S(t) = log(S+t) - log(t)/n``."""
    if n < 2:
        raise ValueError("n must be at least 2.")
    S_array = np.asarray(S, dtype=np.float64)
    t_array = np.asarray(t, dtype=np.float64)
    if np.any(S_array <= 0.0) or np.any(t_array <= 0.0):
        raise ValueError("S and t must be strictly positive.")
    return np.log(S_array + t_array) - np.log(t_array) / n


def D_value(
    S: np.ndarray | float,
    x: np.ndarray | float,
    n: int,
    L: float,
    U: float,
) -> np.ndarray:
    """Evaluate the paper's exact one-record expression ``D(S,x)``."""
    if n < 2:
        raise ValueError("n must be at least 2.")
    if not (np.isfinite(L) and np.isfinite(U) and 0.0 < L < U):
        raise ValueError("Bounds must satisfy 0 < L < U.")
    S_array = np.asarray(S, dtype=np.float64)
    x_array = np.asarray(x, dtype=np.float64)
    if np.any(~np.isfinite(S_array)) or np.any(~np.isfinite(x_array)):
        raise ValueError("S and x must be finite.")
    tolerance = 1e-12 * max(1.0, abs((n - 1) * U))
    if (
        np.any(S_array < (n - 1) * L - tolerance)
        or np.any(S_array > (n - 1) * U + tolerance)
        or np.any(x_array < L)
        or np.any(x_array > U)
    ):
        raise ValueError("Require S in [(n-1)L,(n-1)U] and x in [L,U].")

    t_star = S_array / (n - 1)
    h_x = h_value(S_array, x_array, n)
    h_L = h_value(S_array, L, n)
    h_U = h_value(S_array, U, n)
    h_star = h_value(S_array, t_star, n)
    return np.maximum.reduce(
        [np.abs(h_L - h_x), np.abs(h_U - h_x), h_x - h_star]
    )


def R_value(
    S: np.ndarray | float,
    n: int,
    L: float,
    U: float,
) -> np.ndarray:
    """Evaluate ``R(S) = max_{t in [L,U]} D(S,t)``."""
    if n < 2:
        raise ValueError("n must be at least 2.")
    if not (np.isfinite(L) and np.isfinite(U) and 0.0 < L < U):
        raise ValueError("Bounds must satisfy 0 < L < U.")
    S_array = np.asarray(S, dtype=np.float64)
    tolerance = 1e-12 * max(1.0, abs((n - 1) * U))
    if (
        np.any(~np.isfinite(S_array))
        or np.any(S_array < (n - 1) * L - tolerance)
        or np.any(S_array > (n - 1) * U + tolerance)
    ):
        raise ValueError("S must lie in [(n-1)L,(n-1)U].")
    t_star = S_array / (n - 1)
    h_star = h_value(S_array, t_star, n)
    return np.maximum(
        h_value(S_array, L, n) - h_star,
        h_value(S_array, U, n) - h_star,
    )


def sums_of_q_smallest_excluding_each(
    x_sorted: np.ndarray,
    prefix_sum: np.ndarray,
    q: int,
) -> np.ndarray:
    """Sum the ``q`` smallest values after excluding each observation."""
    n = x_sorted.size
    if q < 0 or q > n - 1:
        raise ValueError("q must lie between 0 and n-1.")
    if q == 0:
        return np.zeros(n, dtype=np.float64)

    index = np.arange(n)
    result = np.full(n, prefix_sum[q], dtype=np.float64)
    mask = index < q
    result[mask] = prefix_sum[q + 1] - x_sorted[mask]
    return result


def sums_of_q_largest_excluding_each(
    x_sorted: np.ndarray,
    prefix_sum: np.ndarray,
    q: int,
) -> np.ndarray:
    """Sum the ``q`` largest values after excluding each observation."""
    n = x_sorted.size
    if q < 0 or q > n - 1:
        raise ValueError("q must lie between 0 and n-1.")
    if q == 0:
        return np.zeros(n, dtype=np.float64)

    total = prefix_sum[n]
    index = np.arange(n)
    result = np.full(n, total - prefix_sum[n - q], dtype=np.float64)
    mask = index >= n - q
    result[mask] = total - prefix_sum[n - q - 1] - x_sorted[mask]
    return result


def S_bounds_excluding_each(
    x_sorted: np.ndarray,
    prefix_sum: np.ndarray,
    q: int,
    L: float,
    U: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the paper's lower/upper reachable sums for every excluded m."""
    total = prefix_sum[-1]
    original = total - x_sorted
    if q == 0:
        return original.copy(), original.copy()

    largest = sums_of_q_largest_excluding_each(x_sorted, prefix_sum, q)
    smallest = sums_of_q_smallest_excluding_each(x_sorted, prefix_sum, q)
    lower = original - (largest - q * L)
    upper = original + (q * U - smallest)
    return lower, upper


def exact_local_sensitivity_mld(x: np.ndarray, L: float, U: float) -> float:
    """Return the exact local sensitivity ``max_m D(S_m,x_m)``."""
    values = _validate_bounded_data(x, L, U)
    S = np.sum(values, dtype=np.float64) - values
    return float(np.max(D_value(S, values, values.size, L, U)))


def _validate_sorted_inputs(
    x_sorted: np.ndarray,
    prefix_sum: np.ndarray,
    L: float,
    U: float,
) -> tuple[np.ndarray, np.ndarray]:
    values = _validate_bounded_data(x_sorted, L, U)
    prefix = np.asarray(prefix_sum, dtype=np.float64)
    if np.any(np.diff(values) < 0.0):
        raise ValueError("x_sorted must be sorted in nondecreasing order.")
    if prefix.shape != (values.size + 1,):
        raise ValueError("prefix_sum must have length n+1.")
    expected = np.concatenate(([0.0], np.cumsum(values, dtype=np.float64)))
    if not np.allclose(prefix, expected, rtol=1e-12, atol=1e-12):
        raise ValueError("prefix_sum is inconsistent with x_sorted.")
    return values, prefix


def exact_Ak_mld(
    x_sorted: np.ndarray,
    prefix_sum: np.ndarray,
    k: int,
    L: float,
    U: float,
    *,
    validate: bool = True,
) -> float:
    """Return the exact ``A^(k)(x)`` from the neighborhood theorem."""
    if validate:
        values, prefix = _validate_sorted_inputs(x_sorted, prefix_sum, L, U)
    else:
        values = np.asarray(x_sorted, dtype=np.float64)
        prefix = np.asarray(prefix_sum, dtype=np.float64)
    n = values.size
    if k < 0 or k > n:
        raise ValueError("k must lie between 0 and n.")

    if k == 0:
        S = prefix[-1] - values
        return float(np.max(D_value(S, values, n, L, U)))
    if k == n:
        return global_sensitivity_mld(n, L, U)

    lower_k, upper_k = S_bounds_excluding_each(values, prefix, k, L, U)
    lower_km1, upper_km1 = S_bounds_excluding_each(
        values, prefix, k - 1, L, U
    )
    return float(
        max(
            np.max(D_value(lower_k, values, n, L, U)),
            np.max(D_value(upper_k, values, n, L, U)),
            np.max(R_value(lower_km1, n, L, U)),
            np.max(R_value(upper_km1, n, L, U)),
        )
    )


def exact_smooth_sensitivity_mld(
    x: np.ndarray,
    beta: float,
    L: float,
    U: float,
) -> float:
    """Return ``max_k exp(-beta*k) A^(k)(x)``, stopping once exp(-beta*k) GS cannot win."""
    if not np.isfinite(beta) or beta <= 0.0:
        raise ValueError("beta must be positive.")
    values = np.sort(_validate_bounded_data(x, L, U))
    n = values.size
    prefix = np.concatenate(([0.0], np.cumsum(values, dtype=np.float64)))
    global_sensitivity = global_sensitivity_mld(n, L, U)
    best = exact_Ak_mld(values, prefix, 0, L, U, validate=False)

    for k in range(1, n):
        weighted_global_bound = math.exp(-beta * k) * global_sensitivity
        if weighted_global_bound <= best:
            break
        A_k = exact_Ak_mld(values, prefix, k, L, U, validate=False)
        best = max(best, math.exp(-beta * k) * A_k)

    return float(max(best, math.exp(-beta * n) * global_sensitivity))


def polynomial_mechanism_parameters(
    epsilon: float,
    gamma: float,
) -> tuple[float, float]:
    """Return the paper's one-dimensional pure-DP ``(beta, alpha)``."""
    if not np.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("epsilon must be positive.")
    if not np.isfinite(gamma) or gamma <= 1.0:
        raise ValueError("gamma must exceed 1.")
    return epsilon / gamma, epsilon / (4.0 * gamma)


def sample_polynomial_noise(
    gamma: float,
    size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample density proportional to ``1/(1+|z|**gamma)``."""
    if not np.isfinite(gamma) or gamma <= 1.0:
        raise ValueError("gamma must exceed 1.")
    if not isinstance(size, (int, np.integer)) or isinstance(size, bool):
        raise TypeError("size must be an integer.")
    if size < 1:
        raise ValueError("size must be positive.")
    if gamma == 2.0:
        return rng.standard_cauchy(size=size)

    a = 1.0 / gamma
    b = 1.0 - a
    beta_prime = rng.gamma(a, 1.0, size) / rng.gamma(b, 1.0, size)
    magnitude = np.power(beta_prime, 1.0 / gamma)
    signs = rng.choice(np.array([-1.0, 1.0]), size=size)
    return signs * magnitude


def calibrate_mld_release(
    x: np.ndarray,
    epsilon: float,
    L: float,
    U: float,
    gamma: float = 2.0,
) -> MLDCalibration:
    """Compute every deterministic quantity used by the MLD mechanism."""
    values = _validate_bounded_data(x, L, U)
    beta, alpha = polynomial_mechanism_parameters(epsilon, gamma)
    local = exact_local_sensitivity_mld(values, L, U)
    smooth = exact_smooth_sensitivity_mld(values, beta, L, U)
    global_sensitivity = global_sensitivity_mld(values.size, L, U)
    return MLDCalibration(
        center=mld(values),
        local_sensitivity=local,
        smooth_sensitivity=smooth,
        global_sensitivity=global_sensitivity,
        beta=beta,
        alpha=alpha,
        noise_scale=smooth / alpha,
        n=values.size,
        lower_bound=float(L),
        upper_bound=float(U),
        gamma=float(gamma),
        epsilon=float(epsilon),
    )


def sample_dp_mld(
    x: np.ndarray,
    epsilon: float,
    L: float,
    U: float,
    gamma: float,
    size: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, MLDCalibration]:
    """Draw repeated releases from the paper's polynomial MLD mechanism."""
    calibration = calibrate_mld_release(x, epsilon, L, U, gamma)
    noise = sample_polynomial_noise(gamma, size, rng)
    return calibration.center + calibration.noise_scale * noise, calibration
