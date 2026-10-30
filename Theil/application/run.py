"""Personal/household Theil experiments, saved results, and figures."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Theil.core import smooth_upper_bounds_theil, theil
from Theil.application.bootstrap import BOOTSTRAP_BATCH_SIZE, bootstrap_theil
from Theil.private_upper import clip_at_log_upper, sample_upper_bounds
from common.data import (
    DEFAULT_INPUT_CSV, DEFAULT_INCOME_COLUMN, HOUSEHOLD_INCOME_COLUMN, HOUSEHOLD_INPUT_CSV,
    load_cps,
)
from common.population import keyed_rng
from common.reporting import (
    add_plot_arguments, plot_real_accuracy, plot_real_releases, replot_report,
    save_real_paper_tables, save_real_plot_data, write_run_config,
)


@dataclass(frozen=True)
class RealDataConfig:
    epsilons: tuple[float, ...] = (0.25, 0.5, 1., 1.5, 2.)
    epsilon_u: float = 0.15
    lower_bound: float = 1.
    growth: float = 1.01
    n_releases: int = 10_000
    n_bootstrap: int = 2_000
    seed: int = 123


def build_report(x: np.ndarray, config: RealDataConfig = RealDataConfig(), *,
                 sample_index: int = 1, verbose: bool = False,
                 return_details: bool = False):
    """Return three summary tables and the releases; optionally the raw records."""
    x = np.sort(np.asarray(x, dtype=float))
    eps = np.asarray(config.epsilons, dtype=float)
    if eps.ndim != 1 or not len(eps) or np.any(~np.isfinite(eps)) or np.any(eps <= 0):
        raise ValueError("epsilons must be finite and positive.")
    if len(np.unique(eps)) != len(eps) or 1. not in eps:
        raise ValueError("Use distinct epsilons including 1 for the oracle comparison.")
    true_value = theil(x)
    bootstrap_values = bootstrap_theil(x, config.n_bootstrap, config.seed)
    sd = float(np.std(bootstrap_values, ddof=1))
    if verbose:
        print(f"bootstrap finished: n={x.size}, Theil={true_value:.9f}, SD={sd:.9f}", flush=True)
    # Upper bounds use the seed directly; release noises use keyed streams.
    uppers = sample_upper_bounds(x, config.epsilon_u, config.n_releases,
                                np.random.default_rng(config.seed), config.lower_bound, config.growth)
    unique, inverse = np.unique(uppers.log_upper, return_inverse=True)
    center_cache, count_cache, smooth_cache, global_cache = [], [], [], []
    for i, lu in enumerate(unique):
        clipped = clip_at_log_upper(x, float(lu), config.lower_bound)
        center_cache.append(theil(clipped))
        count_cache.append(int(np.sum(clipped < x)))
        bounds, global_sensitivity = smooth_upper_bounds_theil(
            clipped, eps/2, config.lower_bound, log_upper=float(lu))
        smooth_cache.append(bounds)
        global_cache.append(global_sensitivity)
        if verbose and ((i+1) % 100 == 0 or i+1 == len(unique)):
            print(f"upper-bound calibrations {i+1}/{len(unique)}", flush=True)
    centers = np.asarray(center_cache)[inverse]
    clipped_counts = np.asarray(count_cache)[inverse]
    smooth = np.asarray(smooth_cache)[inverse]
    global_sensitivities = np.asarray(global_cache)[inverse]
    releases, rows, raw_frames = {}, [], []
    for j, epsilon in enumerate(eps):
        noise = keyed_rng(config.seed, 0, sample_index, 300, j+1).standard_cauchy(config.n_releases)
        scale = smooth[:, j]/(epsilon/8)
        released = centers+scale*noise
        releases[float(epsilon)] = released
        error = np.abs(released-true_value)
        median_ae = float(np.median(error))
        q95_ae = float(np.quantile(error, .95))
        rows.append({
            "epsilon_theil": epsilon, "epsilon_total": config.epsilon_u+epsilon,
            "median_released_theil": float(np.median(released)),
            "median_absolute_error": median_ae, "q95_absolute_error": q95_ae,
            "median_AE_over_bootstrap_SD": median_ae/sd if sd else np.nan,
            "q95_AE_over_bootstrap_SD": q95_ae/sd if sd else np.nan,
            "fraction_AE_below_bootstrap_SD": float(np.mean(error < sd)),
        })
        if return_details:
            ratio = np.divide(smooth[:, j], global_sensitivities,
                              out=np.full(config.n_releases, np.nan),
                              where=global_sensitivities > 0)
            raw_frames.append(pd.DataFrame({
                "release_id": np.arange(1, config.n_releases+1), "n": len(x),
                "epsilon_theil": epsilon, "epsilon_total": config.epsilon_u+epsilon,
                "sample_value": true_value, "clipped_value": centers,
                "released_value": released, "log_private_u": uppers.log_upper,
                "clipped_count": clipped_counts, "clipped_percent": 100*clipped_counts/len(x),
                "global_sensitivity": global_sensitivities,
                "smooth_upper_bound": smooth[:, j], "smooth_to_global_ratio": ratio,
                "noise_scale": scale, "standard_noise": noise,
                "signed_clipping_error": centers-true_value,
                "signed_noise_error": released-centers,
                "signed_mechanism_error": released-true_value,
                "absolute_error": error, "absolute_clipping_error": np.abs(centers-true_value),
            }))
    maximum = float(x[-1])
    oracle_bounds, oracle_gs = smooth_upper_bounds_theil(x, np.array([.5]), config.lower_bound, maximum)
    oracle_bound = oracle_bounds[0]
    oracle_noise = keyed_rng(config.seed, 0, sample_index, 400).standard_cauchy(config.n_releases)
    oracle_signed_error = 8*oracle_bound*oracle_noise
    oracle_error = np.abs(oracle_signed_error)
    private_error = np.abs(releases[1.]-true_value)
    comparison = pd.DataFrame([
        {"upper_bound_strategy": strategy, "median_absolute_error": float(np.median(error)),
         "q95_absolute_error": float(np.quantile(error,.95))}
        for strategy, error in [("Private U, epsilon_u="+str(config.epsilon_u), private_error),
                                ("Observed U=max(X) (nonprivate benchmark)", oracle_error)]
    ])
    sorted_logs = np.sort(uppers.log_upper)
    middle = len(sorted_logs)//2
    median_log_u = (float(sorted_logs[middle]) if len(sorted_logs)%2 else
                    float(np.logaddexp(sorted_logs[middle-1], sorted_logs[middle])-math.log(2)))
    median_u = math.exp(median_log_u) if median_log_u < math.log(np.finfo(float).max) else f"exp({median_log_u})"
    clipping_error = float(np.median(np.abs(centers-true_value)))
    diagnostics = pd.DataFrame([{
        "n": len(x), "true_theil": true_value, "bootstrap_sd": sd,
        "n_bootstrap": config.n_bootstrap, "n_releases": config.n_releases,
        "epsilon_u": config.epsilon_u, "threshold_budget_fraction": .5, "grid_growth": config.growth,
        "target_count": len(x)+1/config.epsilon_u, "sample_max": maximum,
        "median_private_u": median_u, "median_log_private_u": median_log_u,
        "percent_u_below_sample_max": 100*float(np.mean(uppers.log_upper < math.log(maximum))),
        "median_clipped_count": float(np.median(clipped_counts)),
        "median_clipped_percent": 100*float(np.median(clipped_counts))/len(x),
        "median_absolute_clipping_error": clipping_error,
        "q95_absolute_clipping_error": float(np.quantile(np.abs(centers-true_value), .95)),
        "clipping_error_percent_of_bootstrap_sd": 100*clipping_error/sd if sd else np.nan,
        "upper_draws_beyond_float64": int(np.sum(uppers.log_upper > math.log(np.finfo(float).max))),
    }])
    tables = (pd.DataFrame(rows), comparison, diagnostics, releases)
    if not return_details:
        return tables
    details = {
        "raw_results": pd.concat(raw_frames, ignore_index=True),
        "bootstrap_results": pd.DataFrame({
            "bootstrap_id": np.arange(1, config.n_bootstrap+1), "n": len(x),
            "sample_value": true_value, "bootstrap_value": bootstrap_values,
            "signed_error": bootstrap_values-true_value,
        }),
        "oracle_results": pd.DataFrame({
            "release_id": np.arange(1, config.n_releases+1), "n": len(x),
            "epsilon_theil": 1., "upper_bound": maximum,
            "sample_value": true_value, "released_value": true_value+oracle_signed_error,
            "global_sensitivity": oracle_gs, "smooth_upper_bound": oracle_bound,
            "smooth_to_global_ratio": oracle_bound/oracle_gs if oracle_gs > 0 else np.nan,
            "noise_scale": 8*oracle_bound, "standard_noise": oracle_noise,
            "signed_mechanism_error": oracle_signed_error, "absolute_error": oracle_error,
        }),
    }
    return (*tables, details)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", choices=["personal", "household", "both"], default="both")
    parser.add_argument("--releases", type=int, default=10_000)
    parser.add_argument("--bootstrap", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent/"results")
    add_plot_arguments(parser)
    parser.add_argument("--save-details", action="store_true",
                        help="Also save raw CSVs and configuration in details/.")
    args = parser.parse_args()
    samples = ("personal", "household") if args.sample == "both" else (args.sample,)
    if args.plot_only:
        for sample in samples:
            output = args.output_dir / f"{sample}_report"
            replot_report(output, statistic="Theil", sample=sample)
            print(f"Regenerated figures from saved results in {output.resolve()}")
        return
    if args.releases < 1 or args.bootstrap < 2 or args.seed < 0:
        parser.error("Require releases>=1, bootstrap>=2, seed>=0.")
    config = RealDataConfig(n_releases=args.releases, n_bootstrap=args.bootstrap, seed=args.seed)
    for sample in samples:
        print(f"Starting {sample} income experiment", flush=True)
        sample_index = 1 if sample == "personal" else 2
        tables = build_report(load_cps(sample), config, sample_index=sample_index,
                              verbose=True, return_details=True)
        output = args.output_dir/f"{sample}_report"
        output.mkdir(parents=True, exist_ok=True)
        save_real_plot_data(tables[0], tables[2], tables[3], statistic="Theil",
                            sample=sample, output_dir=output)
        save_real_paper_tables(*tables[:3], output, "Theil", sample, config.seed)
        if not args.no_plots:
            plot_real_accuracy(tables[0], tables[2], epsilon_column="epsilon_theil",
                               statistic="Theil", sample=sample, output_dir=output)
            plot_real_releases(tables[3], float(tables[2].iloc[0].true_theil),
                               statistic="Theil", sample=sample, output_dir=output)
        print(tables[0].to_string(index=False), flush=True)
        if not args.save_details:
            continue
        output = output / "details"
        output.mkdir(exist_ok=True)
        for table, filename in zip(tables[:3], ["table_1_accuracy.csv", "table_2_upper_bound_comparison.csv",
                                              "table_3_private_upper_bound_and_clipping.csv"]):
            table.to_csv(output/filename, index=False)
        for name, table in tables[4].items():
            table.to_csv(output/f"{name}.csv", index=False)
        input_files = ({"personal_csv": str(DEFAULT_INPUT_CSV.resolve()),
                        "income_column": DEFAULT_INCOME_COLUMN}
                       if sample == "personal" else
                       {"household_csv": str(HOUSEHOLD_INPUT_CSV.resolve()),
                        "income_column": HOUSEHOLD_INCOME_COLUMN})
        write_run_config(output/"config.json", {**asdict(config), "statistic": "Theil", "sample": sample,
            "input_files": input_files, "income_shift": 1.,
            "sample_weighting": "unweighted statistic; positive survey weights used for eligibility",
            "n": int(tables[2].iloc[0].n), "sample_index": sample_index,
            "epsilon_total_list": [config.epsilon_u+epsilon for epsilon in config.epsilons],
            "threshold_budget_fraction": .5, "uqe_target": "n + 1/epsilon_u",
            "sensitivity": "paper smooth upper bound, not exact smooth sensitivity",
            "noise": "standard Cauchy; scale = 8 * smooth_upper_bound / epsilon_theil",
            "gamma": 2., "alpha": "epsilon_theil / 8", "beta": "epsilon_theil / 2",
            "ratio": "smooth_to_global_ratio is the paper smooth upper bound divided by exact GS; 0/0 is NaN",
            "bootstrap": "ordinary unweighted iid bootstrap, not a CPS survey-design standard error",
            "bootstrap_batch_size": BOOTSTRAP_BATCH_SIZE, "bootstrap_sd_ddof": 1,
            "quantile_method": "numpy.quantile method='linear' (default)",
            "oracle": "sample-maximum upper bound is a nonprivate benchmark; epsilon_theil=1",
            "oracle_epsilon_theil": 1., "oracle_is_end_to_end_dp": False,
            "rng": "U and bootstrap retain original seed streams; release and oracle use independent keyed streams",
            "rng_scheme": "numpy SeedSequence entropy keys; bootstrap and UQE intentionally preserve the same historical initial seed",
            "rng_keys": {
                "bootstrap": [config.seed], "uqe": [config.seed],
                "release": [config.seed, 0, sample_index, 300, "epsilon_index + 1 (zero-based order in epsilons)"],
                "oracle": [config.seed, 0, sample_index, 400, 0],
            },
            "shared_private_upper_bounds_across_budgets": True,
            "plots_enabled": not args.no_plots,
            "purpose": "public-CPS research evaluation; raw estimates, bootstrap and diagnostics are not private releases"})



if __name__ == "__main__":
    main()
