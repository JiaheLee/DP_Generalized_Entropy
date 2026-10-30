"""Personal/household MLD experiments, saved results, and figures."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from MLD.core import (
    exact_smooth_sensitivity_mld,
    global_sensitivity_mld,
    mld,
    polynomial_mechanism_parameters,
    sample_polynomial_noise,
)
from MLD.private_upper import sample_unbounded_uqe
from MLD.application.bootstrap import (
    BATCH_SIZE,
    N_BOOTSTRAP,
    bootstrap_mld,
)
from common.data import (
    DEFAULT_INCOME_COLUMN,
    DEFAULT_INPUT_CSV,
    HOUSEHOLD_INCOME_COLUMN,
    HOUSEHOLD_INPUT_CSV,
    load_cps,
)
from common.reporting import (
    add_plot_arguments, plot_real_accuracy, plot_real_releases, replot_report,
    save_real_paper_tables, save_real_plot_data, write_run_config,
)


INPUT_CSV = DEFAULT_INPUT_CSV
INCOME_COLUMN = DEFAULT_INCOME_COLUMN
OUTPUT_DIR = Path(__file__).resolve().parent / "results"

EPSILON_MLD_LIST = [0.25, 0.5, 1.0, 1.5, 2.0]
EPSILON_U = 0.15
LOWER_BOUND = 1.0
GRID_GROWTH = 1.01
GAMMA = 2.0
N_RELEASES = 10_000
RANDOM_SEED = 123


def clipping_diagnostics(
    x_sorted: np.ndarray,
    private_u: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Return the original MLD, clipped MLDs, and clipped-record counts."""
    cache = {}
    for upper in np.unique(private_u):
        upper = float(upper)
        cache[upper] = (
            mld(np.minimum(x_sorted, upper)),
            x_sorted.size - np.searchsorted(x_sorted, upper, side="right"),
        )
    centers = np.array([cache[float(upper)][0] for upper in private_u])
    counts = np.array([cache[float(upper)][1] for upper in private_u])
    return mld(x_sorted), centers, counts


def releases_for_epsilon(
    x_sorted: np.ndarray,
    private_u: np.ndarray,
    centers: np.ndarray,
    epsilon_mld: float,
    rng: np.random.Generator,
    *,
    return_details: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, np.ndarray]]:
    """Generate repeated private-upper-bound DP-MLD releases."""
    beta, alpha = polynomial_mechanism_parameters(epsilon_mld, GAMMA)
    smooth_cache = {}
    global_cache = {}
    for upper in np.unique(private_u):
        upper = float(upper)
        smooth_cache[upper] = (
            0.0
            if upper == LOWER_BOUND
            else exact_smooth_sensitivity_mld(
                np.minimum(x_sorted, upper), beta, LOWER_BOUND, upper
            )
        )
        global_cache[upper] = (
            0.0 if upper == LOWER_BOUND
            else global_sensitivity_mld(x_sorted.size, LOWER_BOUND, upper)
        )

    smooth = np.array([smooth_cache[float(upper)] for upper in private_u])
    noise = sample_polynomial_noise(GAMMA, private_u.size, rng)
    draws = centers + (smooth / alpha) * noise
    if not return_details:
        return draws
    return draws, {
        "smooth_sensitivity": smooth,
        "global_sensitivity": np.array(
            [global_cache[float(upper)] for upper in private_u]
        ),
        "noise_scale": smooth / alpha,
        "standard_noise": noise,
    }


def _ratio(numerator: float, denominator: float) -> float:
    """A zero reference SD/GS has no defined normalized error/ratio."""
    return float(numerator / denominator) if denominator > 0.0 else float("nan")


def _release_records(
    values: np.ndarray,
    epsilon: float,
    true_mld: float,
    centers: np.ndarray,
    draws: np.ndarray,
    upper: np.ndarray,
    counts: np.ndarray,
    calibration: dict[str, np.ndarray],
    *,
    private: bool,
) -> pd.DataFrame:
    smooth = calibration["smooth_sensitivity"]
    global_sensitivity = calibration["global_sensitivity"]
    ratio = np.full(smooth.size, np.nan)
    np.divide(smooth, global_sensitivity, out=ratio, where=global_sensitivity > 0)
    return pd.DataFrame({
        "release_id": np.arange(1, draws.size + 1),
        "n": values.size,
        "epsilon_mld": epsilon,
        # The sample-maximum oracle is not an end-to-end DP procedure.
        "epsilon_total": epsilon + EPSILON_U if private else np.nan,
        "sample_value": true_mld,
        "clipped_value": centers,
        "released_value": draws,
        "private_u": upper if private else np.nan,
        "log_private_u": np.log(upper) if private else np.nan,
        "upper_bound": upper,
        "is_end_to_end_dp": private,
        "clipped_count": counts,
        "clipped_percent": 100.0 * counts / values.size,
        **calibration,
        "signed_clipping_error": centers - true_mld,
        "signed_noise_error": draws - centers,
        "signed_mechanism_error": draws - true_mld,
        "absolute_error": np.abs(draws - true_mld),
        "absolute_clipping_error": np.abs(centers - true_mld),
        "ss_over_gs": ratio,
    })


def build_report(
    x_sorted: np.ndarray,
    epsilon_mld_list: list[float] = EPSILON_MLD_LIST,
    n_releases: int = N_RELEASES,
    n_bootstrap: int = N_BOOTSTRAP,
    *,
    seed: int = RANDOM_SEED,
    return_details: bool = False,
) -> tuple:
    """Return three summary tables, the releases and the sample MLD.

    All budgets share the same upper-bound draws. ``return_details=True`` also
    returns the raw, bootstrap and oracle records.
    """
    values = np.sort(np.asarray(x_sorted, dtype=np.float64))
    if values.ndim != 1 or values.size < 2:
        raise ValueError("x_sorted must be a vector of length >= 2.")
    if np.any(~np.isfinite(values)) or np.any(values < LOWER_BOUND):
        raise ValueError("All values must be finite and at least LOWER_BOUND.")
    if not isinstance(n_releases, (int, np.integer)) or n_releases < 1:
        raise ValueError("n_releases must be a positive integer.")
    epsilons = [float(epsilon) for epsilon in epsilon_mld_list]
    if len(set(epsilons)) != len(epsilons):
        raise ValueError("epsilon_mld_list must not contain duplicates.")
    if 1.0 not in epsilons:
        raise ValueError("epsilon_mld_list must contain 1.0 for Table 2.")
    for epsilon in epsilons:
        polynomial_mechanism_parameters(epsilon, GAMMA)

    bootstrap_draws = bootstrap_mld(
        values,
        n_bootstrap=n_bootstrap,
        batch_size=BATCH_SIZE,
        seed=seed,
    )
    bootstrap_sd = float(np.std(bootstrap_draws, ddof=1))
    uqe = sample_unbounded_uqe(
        x_sorted=values,
        epsilon_u=EPSILON_U,
        n_draws=n_releases,
        rng=np.random.default_rng(seed),
        lower_bound=LOWER_BOUND,
        growth=GRID_GROWTH,
        threshold_budget_fraction=0.5,
    )
    private_u = uqe.private_u
    true_mld, centers, clipped_counts = clipping_diagnostics(values, private_u)
    releases = {}
    raw_frames = []
    table_1_rows = []
    noise_rng = np.random.default_rng(seed)
    for epsilon in epsilons:
        draws, calibration = releases_for_epsilon(
            values, private_u, centers, epsilon, noise_rng,
            return_details=True,
        )
        releases[epsilon] = draws
        records = _release_records(
            values, epsilon, true_mld, centers, draws, private_u,
            clipped_counts, calibration, private=True,
        )
        records["log_private_u"] = uqe.log_private_u
        records["uqe_grid_index"] = uqe.grid_index
        records["uqe_used_geometric_tail"] = uqe.used_geometric_tail
        records["uqe_geometric_wait"] = uqe.geometric_wait
        raw_frames.append(records)
        absolute_error = records["absolute_error"].to_numpy()
        median_ae = float(np.median(absolute_error))
        q95_ae = float(np.quantile(absolute_error, 0.95))
        table_1_rows.append({
            "epsilon_mld": epsilon,
            "epsilon_total": epsilon + EPSILON_U,
            "median_released_mld": float(np.median(draws)),
            "median_absolute_error": median_ae,
            "q95_absolute_error": q95_ae,
            "median_AE_over_bootstrap_SD": _ratio(median_ae, bootstrap_sd),
            "q95_AE_over_bootstrap_SD": _ratio(q95_ae, bootstrap_sd),
            "fraction_AE_below_bootstrap_SD": float(
                np.mean(absolute_error < bootstrap_sd)
            ),
        })
    table_1 = pd.DataFrame(table_1_rows)

    oracle_upper = np.full(n_releases, values[-1])
    oracle_centers = np.full(n_releases, true_mld)
    oracle_draws, oracle_calibration = releases_for_epsilon(
        values, oracle_upper, oracle_centers, 1.0,
        np.random.default_rng(seed),
        return_details=True,
    )
    oracle_results = _release_records(
        values, 1.0, true_mld, oracle_centers, oracle_draws, oracle_upper,
        np.zeros(n_releases, dtype=int), oracle_calibration, private=False,
    )
    table_2_rows = []
    for label, draws in (
        (f"Private U, epsilon_u={EPSILON_U:g}", releases[1.0]),
        ("Observed U=max(X)", oracle_draws),
    ):
        error = np.abs(draws - true_mld)
        table_2_rows.append({
            "upper_bound_strategy": label,
            "median_absolute_error": float(np.median(error)),
            "q95_absolute_error": float(np.quantile(error, 0.95)),
        })
    table_2 = pd.DataFrame(table_2_rows)

    median_clipped_count = float(np.median(clipped_counts))
    clipping_error = np.abs(centers - true_mld)
    median_clipping_error = float(np.median(clipping_error))
    table_3 = pd.DataFrame([{
        "n": values.size,
        "true_mld": true_mld,
        "n_releases": n_releases,
        "n_bootstrap": n_bootstrap,
        "epsilon_u": EPSILON_U,
        "threshold_budget_fraction": 0.5,
        "grid_growth": GRID_GROWTH,
        "target_count": uqe.target_count,
        "sample_max": float(values[-1]),
        "median_private_u": float(np.median(private_u)),
        "percent_u_below_sample_max": float(100.0 * np.mean(private_u < values[-1])),
        "median_clipped_count": median_clipped_count,
        "median_clipped_percent": 100.0 * median_clipped_count / values.size,
        "median_absolute_clipping_error": median_clipping_error,
        "q95_absolute_clipping_error": float(np.quantile(clipping_error, 0.95)),
        "bootstrap_sd": bootstrap_sd,
        "clipping_error_percent_of_bootstrap_sd": (
            100.0 * _ratio(median_clipping_error, bootstrap_sd)
        ),
    }])
    output = (table_1, table_2, table_3, releases, true_mld)
    if not return_details:
        return output
    bootstrap_results = pd.DataFrame({
        "bootstrap_id": np.arange(1, n_bootstrap + 1),
        "n": values.size,
        "sample_value": true_mld,
        "bootstrap_value": bootstrap_draws,
        "signed_sampling_error": bootstrap_draws - true_mld,
        "absolute_sampling_error": np.abs(bootstrap_draws - true_mld),
    })
    return (*output, {
        "raw_results": pd.concat(raw_frames, ignore_index=True),
        "bootstrap_results": bootstrap_results,
        "oracle_results": oracle_results,
    })


def report_argument_parser(description: str, output_dir: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--sample", choices=["personal", "household", "both"], default="both")
    parser.add_argument("--releases", type=int, default=N_RELEASES)
    parser.add_argument("--bootstrap", type=int, default=N_BOOTSTRAP)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--output-dir", type=Path, default=output_dir)
    add_plot_arguments(parser)
    parser.add_argument("--save-details", action="store_true",
                        help="Also save raw CSVs and configuration in details/.")
    return parser


def save_report(
    report: tuple,
    output_dir: Path,
    *,
    seed: int,
    input_files: dict[str, str],
    household: bool = False,
    make_plots: bool = True,
    save_details: bool = False,
) -> None:
    """Always save reusable plot inputs; full diagnostics are optional."""
    table_1, table_2, table_3, releases, true_mld, details = report
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sample = "household" if household else "personal"
    save_real_plot_data(table_1, table_3, releases, statistic="MLD", sample=sample,
                        output_dir=output_dir)
    save_real_paper_tables(table_1, table_2, table_3, output_dir, "MLD", sample, seed)
    if make_plots:
        plot_real_accuracy(table_1, table_3, epsilon_column="epsilon_mld",
                           statistic="MLD", sample=sample, output_dir=output_dir)
        plot_real_releases(releases, true_mld, statistic="MLD", sample=sample,
                           output_dir=output_dir)
    if not save_details:
        return
    output_dir = output_dir / "details"
    output_dir.mkdir(exist_ok=True)
    suffix = "household_" if household else ""
    table_1.to_csv(output_dir / f"table_1_{suffix}accuracy.csv", index=False)
    table_2.to_csv(output_dir / f"table_2_{suffix}upper_bound_comparison.csv", index=False)
    table_3.to_csv(
        output_dir / f"table_3_{suffix}private_upper_bound_and_clipping.csv", index=False
    )
    for name in ("raw_results", "bootstrap_results", "oracle_results"):
        details[name].to_csv(output_dir / f"{name}.csv", index=False)
    write_run_config(output_dir / "config.json", {
        "statistic": "MLD",
        "analysis": "household_income" if household else "personal_income",
        "input_files": input_files,
        "income_shift": 1.0,
        "sample_weighting": "unweighted statistic; positive weights used for eligibility",
        "n": int(table_3.iloc[0]["n"]),
        "n_releases": int(table_3.iloc[0]["n_releases"]),
        "n_bootstrap": int(table_3.iloc[0]["n_bootstrap"]),
        "bootstrap_method": "ordinary iid nonparametric; SD uses ddof=1",
        "bootstrap_batch_size": BATCH_SIZE,
        "seed": int(seed),
        "rng_scheme": "legacy numpy.default_rng(seed), reinitialized for bootstrap, UQE, DP noise and oracle",
        "dp_noise_order": "one continuous stream, one n_releases block per budget in epsilon_mld_list order",
        "shared_private_upper_bounds_across_budgets": True,
        "epsilon_mld_list": list(releases),
        "epsilon_u": EPSILON_U,
        "epsilon_total_list": [epsilon + EPSILON_U for epsilon in releases],
        "lower_bound": LOWER_BOUND,
        "grid_growth": GRID_GROWTH,
        "threshold_budget_fraction": 0.5,
        "uqe_target": "n + 1/epsilon_u",
        "gamma": GAMMA,
        "noise_distribution": "standard Cauchy",
        "beta": "epsilon_mld / gamma",
        "alpha": "epsilon_mld / (4 * gamma)",
        "oracle_is_end_to_end_dp": False,
        "normalized_ratio_when_denominator_zero": "NaN (undefined)",
        "plots_enabled": bool(make_plots),
        "results_note": "Original seeded real-data experiment; preserves historical random-stream initialization and consumption order.",
    })


def main() -> None:
    parser = report_argument_parser(__doc__, OUTPUT_DIR)
    args = parser.parse_args()
    samples = ("personal", "household") if args.sample == "both" else (args.sample,)
    if args.plot_only:
        for sample in samples:
            output = args.output_dir / f"{sample}_report"
            replot_report(output, statistic="MLD", sample=sample)
            print(f"Regenerated figures from saved results in {output.resolve()}")
        return
    if args.releases < 1 or args.bootstrap < 2 or args.seed < 0:
        parser.error("Require releases>=1, bootstrap>=2, seed>=0.")
    for sample in samples:
        print(f"Starting {sample} income experiment", flush=True)
        report = build_report(
            load_cps(sample), n_releases=args.releases, n_bootstrap=args.bootstrap,
            seed=args.seed, return_details=True,
        )
        output = args.output_dir / f"{sample}_report"
        input_files = (
            {"personal_csv": str(INPUT_CSV), "income_column": INCOME_COLUMN}
            if sample == "personal" else
            {"household_csv": str(HOUSEHOLD_INPUT_CSV), "income_column": HOUSEHOLD_INCOME_COLUMN}
        )
        save_report(
            report, output, seed=args.seed, household=sample == "household",
            input_files=input_files, make_plots=not args.no_plots, save_details=args.save_details,
        )
        print(f"Saved MLD {sample}-income report to {output.resolve()}")


if __name__ == "__main__":
    main()
