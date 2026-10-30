#!/usr/bin/env python3
"""Run the DP-MLD simulation and retain draws, error summaries, and figures."""

from __future__ import annotations

import csv
from functools import partial
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd


SIMULATION_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SIMULATION_DIR.parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from MLD.core import (  # noqa: E402
    exact_Ak_mld,
    global_sensitivity_mld,
    mld,
    polynomial_mechanism_parameters,
    sample_polynomial_noise,
)
from MLD.private_upper import sample_unbounded_uqe  # noqa: E402
from common.reporting import (  # noqa: E402
    simulation_arguments, run_replications, save_simulation_report, replot_report,
    summarize_simulation_quantiles,
)
from common.population import CALIBRATION_PATH, SAMPLE_STREAMS, keyed_rng  # noqa: E402


def load_calibration(path: Path = CALIBRATION_PATH) -> dict[str, float]:
    """Load the DGP parameters created by ``python -m common.calibrate``."""
    if not path.exists():
        raise FileNotFoundError(
            f"Missing calibration file: {path}. Run "
            "`python -m common.calibrate` first."
        )
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise ValueError("The calibration CSV must contain exactly one row.")

    fields = (
        "pi_zero",
        "gb2_a",
        "gb2_b",
        "gb2_p",
        "gb2_q",
        "population_mld_shifted",
    )
    try:
        calibration = {field: float(rows[0][field]) for field in fields}
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("The calibration CSV has missing or invalid fields.") from error
    if not 0.0 <= calibration["pi_zero"] < 1.0:
        raise ValueError("pi_zero must lie in [0,1).")
    if any(
        not np.isfinite(calibration[field]) or calibration[field] <= 0.0
        for field in fields[1:5]
    ):
        raise ValueError("All GB2 parameters must be finite and positive.")
    if not np.isfinite(calibration["population_mld_shifted"]):
        raise ValueError("population_mld_shifted must be finite.")
    return calibration


SIMULATION_SAMPLE_SIZES = (10_000, 100_000, 1_000_000, 1_000)
EPSILON_MLD = (0.5, 1.0, 2.0)
EPSILON_U = 0.15
GAMMA = 2.0
LOWER_BOUND = 1.0
GRID_GROWTH = 1.01
RAW_COLUMNS = (
    "replication", "n", "epsilon_mld", "epsilon_total", "population_value",
    "sample_value", "clipped_value", "released_value", "private_u",
    "log_private_u", "grid_index", "global_sensitivity", "smooth_sensitivity",
    "ss_over_gs", "noise_scale", "standard_noise", "signed_sampling_error",
    "signed_clipping_error", "signed_noise_error", "signed_mechanism_error",
    "signed_total_error", "absolute_sampling_error", "absolute_clipping_error",
    "absolute_noise_error", "absolute_mechanism_error", "absolute_total_error",
    "n_clipped", "fraction_clipped",
)


def draw_income(
    n: int, rng: np.random.Generator,
    calibration: dict[str, float] | None = None,
) -> np.ndarray:
    """Draw shifted income ``X=Y+1`` from the calibrated zero-mass GB2."""
    y = np.zeros(n, dtype=np.float64)
    calibration = load_calibration() if calibration is None else calibration
    positive = np.flatnonzero(rng.random(n) >= calibration["pi_zero"])
    if positive.size:
        v = rng.beta(calibration["gb2_p"], calibration["gb2_q"], size=positive.size)
        invalid = (v <= 0.0) | (v >= 1.0) | (~np.isfinite(v))
        while np.any(invalid):
            v[invalid] = rng.beta(calibration["gb2_p"], calibration["gb2_q"], size=int(np.sum(invalid)))
            invalid = (v <= 0.0) | (v >= 1.0) | (~np.isfinite(v))
        log_y = math.log(calibration["gb2_b"]) + (np.log(v) - np.log1p(-v)) / calibration["gb2_a"]
        y[positive] = np.exp(log_y)
    x = y + 1.0
    if np.any(~np.isfinite(x)):
        raise FloatingPointError("The GB2 draw contains non-finite income.")
    return x


def smooth_sensitivities(
    x_sorted: np.ndarray,
    betas: np.ndarray,
    upper_bound: float,
) -> tuple[np.ndarray, float]:
    """Compute exact smooth sensitivities while sharing the A^(k) sequence."""
    n = x_sorted.size
    prefix = np.concatenate(([0.0], np.cumsum(x_sorted, dtype=np.float64)))
    global_sensitivity = global_sensitivity_mld(n, LOWER_BOUND, upper_bound)
    local_sensitivity = exact_Ak_mld(
        x_sorted,
        prefix,
        0,
        LOWER_BOUND,
        upper_bound,
        validate=False,
    )

    best = np.full(betas.size, local_sensitivity)
    active = np.ones(betas.size, dtype=bool)
    for k in range(1, n):
        active &= np.exp(-betas * k) * global_sensitivity > best
        if not np.any(active):
            break
        a_k = exact_Ak_mld(
            x_sorted,
            prefix,
            k,
            LOWER_BOUND,
            upper_bound,
            validate=False,
        )
        best[active] = np.maximum(
            best[active], np.exp(-betas[active] * k) * a_k
        )
    return np.maximum(
        best, np.exp(-betas * n) * global_sensitivity
    ), float(global_sensitivity)


def simulate_replication(
    replication: int,
    base_seed: int,
    *,
    sample_sizes: tuple[int, ...] = SIMULATION_SAMPLE_SIZES,
    epsilons: tuple[float, ...] = EPSILON_MLD,
    epsilon_u: float = EPSILON_U,
    growth: float = GRID_GROWTH,
    calibration: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Return one row per sample size and budget for one replication."""
    calibration = load_calibration() if calibration is None else calibration
    population_mld = calibration["population_mld_shifted"]
    parameters = [
        polynomial_mechanism_parameters(epsilon, GAMMA)
        for epsilon in epsilons
    ]
    betas = np.array([beta for beta, _ in parameters])
    rows = []

    for n in sample_sizes:
        n_index = SAMPLE_STREAMS.get(n, n+10)
        x = draw_income(n, keyed_rng(base_seed, replication, n_index, 100), calibration)
        sample_mld = mld(x)
        x_sorted = np.sort(x)
        upper_draw = sample_unbounded_uqe(
            x_sorted,
            epsilon_u=epsilon_u,
            n_draws=1,
            rng=keyed_rng(base_seed, replication, n_index, 200),
            lower_bound=LOWER_BOUND,
            growth=growth,
            threshold_budget_fraction=0.5,
        )
        private_u = float(upper_draw.private_u[0])
        clipped = np.minimum(x_sorted, private_u)
        clipped_mld = mld(clipped)
        n_clipped = int(np.count_nonzero(clipped < x_sorted))

        if private_u == LOWER_BOUND:
            smooth = np.zeros(len(epsilons))
            global_sensitivity = 0.0
        else:
            smooth, global_sensitivity = smooth_sensitivities(
                clipped, betas, private_u
            )

        for epsilon_index, epsilon in enumerate(epsilons, start=1):
            alpha = parameters[epsilon_index - 1][1]
            noise = sample_polynomial_noise(
                GAMMA,
                1,
                keyed_rng(
                    base_seed, replication, n_index, 300, epsilon_index
                ),
            )[0]
            scale = smooth[epsilon_index - 1] / alpha
            released_mld = clipped_mld + scale * noise
            ratio = float(smooth[epsilon_index - 1] / global_sensitivity) if global_sensitivity > 0.0 else np.nan
            record = {
                "replication": replication, "n": n, "epsilon_mld": epsilon,
                "epsilon_total": epsilon+epsilon_u, "population_value": population_mld,
                "sample_value": sample_mld, "clipped_value": clipped_mld,
                "released_value": released_mld, "private_u": private_u,
                "log_private_u": float(upper_draw.log_private_u[0]),
                "grid_index": int(upper_draw.grid_index[0]),
                "global_sensitivity": global_sensitivity,
                "smooth_sensitivity": float(smooth[epsilon_index-1]),
                "ss_over_gs": ratio, "noise_scale": float(scale), "standard_noise": float(noise),
                "signed_sampling_error": sample_mld-population_mld,
                "signed_clipping_error": clipped_mld-sample_mld,
                "signed_noise_error": released_mld-clipped_mld,
                "signed_mechanism_error": released_mld-sample_mld,
                "signed_total_error": released_mld-population_mld,
                "absolute_sampling_error": abs(sample_mld-population_mld),
                "absolute_clipping_error": abs(clipped_mld-sample_mld),
                "absolute_noise_error": abs(released_mld-clipped_mld),
                "absolute_mechanism_error": abs(released_mld-sample_mld),
                "absolute_total_error": abs(released_mld-population_mld),
                "n_clipped": n_clipped, "fraction_clipped": n_clipped/n,
            }
            rows.append(record)
    return pd.DataFrame(rows, columns=RAW_COLUMNS)


def summarize(raw):
    return summarize_simulation_quantiles(raw, "epsilon_mld", "ss_over_gs")


def main() -> int:
    args = simulation_arguments(__doc__, SIMULATION_DIR / "results", CALIBRATION_PATH,
                                SIMULATION_SAMPLE_SIZES)
    if args.plot_only:
        replot_report(args.output_dir, statistic="MLD")
        print(f"Regenerated figures from saved results in {args.output_dir.resolve()}")
        return 0
    calibration = load_calibration(args.calibration)
    task = partial(simulate_replication, base_seed=args.seed,
                   sample_sizes=tuple(args.sample_sizes), epsilons=tuple(args.epsilons),
                   epsilon_u=args.epsilon_u, growth=args.growth, calibration=calibration)
    raw = run_replications(task, args.replications, args.workers)
    save_simulation_report(raw, args, "MLD", calibration, {
        "calibration_source": str(args.calibration.resolve()),
        "sensitivity_ratio": "exact smooth sensitivity / exact global sensitivity",
        "sample_streams": {str(n): SAMPLE_STREAMS.get(n, n+10) for n in args.sample_sizes},
        "random_streams": "SeedSequence(seed, replication, sample_stream, stage, epsilon_index)",
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
