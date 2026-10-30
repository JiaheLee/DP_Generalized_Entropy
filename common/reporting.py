"""Result tables, plot caches and figures shared by the MLD and Theil experiments."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
import json
import math
import os
from pathlib import Path
import platform
import tempfile

import numpy as np
import pandas as pd


CAUCHY_MOMENT_NOTE = (
    "For nondegenerate raw Cauchy releases, ordinary population bias is undefined "
    "and population MSE is infinite. DP Monte Carlo mean error, empirical MSE "
    "and empirical RMSE describe this finite run only; no tail observations "
    "are discarded or output-clipped."
)
RESEARCH_OUTPUT_NOTE = (
    "These files contain nonprivate sample statistics and mechanism diagnostics "
    "for public-data/synthetic-data research. They are not a private release bundle."
)


def empirical_error_moments(errors: np.ndarray) -> tuple[float, float, float]:
    """Return mean signed error, mean squared error (denominator R), and RMSE."""
    values = np.asarray(errors, dtype=float)
    if values.ndim != 1 or not values.size or np.any(~np.isfinite(values)):
        raise ValueError("Moment summaries require a nonempty finite error vector.")
    scale = float(np.max(np.abs(values)))
    if scale == 0.0:
        return 0.0, 0.0, 0.0
    standardized = values / scale
    mean_error = scale * float(np.mean(standardized))
    rmse = scale * math.sqrt(float(np.mean(standardized ** 2)))
    mse = rmse * rmse
    if not math.isfinite(mse):
        raise FloatingPointError(
            "Empirical squared error exceeds float64 range; retain the raw "
            "draws and evaluate moments at higher precision."
        )
    return mean_error, mse, rmse


def summarize_simulation_moments(
    raw: pd.DataFrame, epsilon_column: str,
) -> pd.DataFrame:
    """Compare sample and DP estimates to the same fixed population target."""
    required = {
        "n", epsilon_column, "epsilon_total", "replication",
        "population_value", "sample_value", "released_value",
    }
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(f"Missing simulation moment fields: {sorted(missing)}")
    if raw.empty:
        raise ValueError("No simulation draws.")
    rows = []
    for (n, epsilon), group in raw.groupby(["n", epsilon_column], sort=True):
        if group.replication.duplicated().any():
            raise ValueError("Duplicate replication within a simulation cell.")
        if group.population_value.nunique() != 1 or group.epsilon_total.nunique() != 1:
            raise ValueError("A simulation cell must have one target and total budget.")
        target = group.population_value.to_numpy(dtype=float)
        sample = group.sample_value.to_numpy(dtype=float) - target
        private = group.released_value.to_numpy(dtype=float) - target
        sample_bias, sample_mse, sample_rmse = empirical_error_moments(sample)
        private_mean, private_mse, private_rmse = empirical_error_moments(private)
        rows.append({
            "n": int(n), epsilon_column: float(epsilon),
            "epsilon_total": float(group.epsilon_total.iloc[0]),
            "replications": len(group), "population_value": float(target[0]),
            "sample_bias": sample_bias, "sample_mse": sample_mse,
            "sample_rmse": sample_rmse,
            "dp_mc_mean_error": private_mean, "dp_empirical_mse": private_mse,
            "dp_empirical_rmse": private_rmse,
            "dp_median_signed_error": float(np.median(private)),
        })
    return pd.DataFrame(rows)


def write_run_config(path: str | Path, config: dict) -> None:
    """Save supplied settings plus runtime versions and metric interpretation."""
    packages = {}
    for name in ("numpy", "pandas", "scipy", "matplotlib"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    payload = {
        **config,
        "report_schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime": {"python": platform.python_version(), **packages},
        "quantile_method": "linear",
        "empirical_mse_denominator": "R (mean squared error; not a sample-variance correction)",
        "dp_moment_interpretation": CAUCHY_MOMENT_NOTE,
        "research_output_note": RESEARCH_OUTPUT_NOTE,
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temporary.replace(destination)



def add_plot_arguments(parser):
    """Plot-only mode never falls back to rerunning an experiment."""
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--no-plots", action="store_true",
                       help="Save tables and plot_data.npz without rendering figures.")
    modes.add_argument("--plot-only", action="store_true",
                       help="Regenerate figures from saved results; do not run the experiment.")


def simulation_arguments(description, output_dir, calibration_path, sample_sizes):
    """The same experiment controls apply to MLD and Theil."""
    parser = argparse.ArgumentParser(description=description)
    for name, kind, default in (("replications", int, 1000), ("seed", int, 123),
                               ("workers", int, 1), ("epsilon-u", float, .15),
                               ("growth", float, 1.01)):
        parser.add_argument("--" + name, type=kind, default=default)
    parser.add_argument("--sample-sizes", type=int, nargs="+", default=list(sample_sizes))
    parser.add_argument("--epsilons", type=float, nargs="+", default=[.5, 1., 2.])
    parser.add_argument("--calibration", type=Path, default=calibration_path)
    parser.add_argument("--output-dir", type=Path, default=output_dir)
    add_plot_arguments(parser)
    parser.add_argument("--save-details", action="store_true",
                        help="Also save raw CSVs and configuration in details/.")
    args = parser.parse_args()
    if args.plot_only:
        return args
    if args.replications < 1 or args.workers < 1 or args.seed < 0:
        parser.error("Require positive replication/worker counts and seed >= 0.")
    if min(args.sample_sizes) < 2 or len(set(args.sample_sizes)) != len(args.sample_sizes):
        parser.error("Sample sizes must be distinct integers >= 2.")
    scaled_budgets = [args.epsilon_u / 2, *(e / 8 for e in args.epsilons)]
    if any(not math.isfinite(e) or e <= 0 for e in scaled_budgets) or len(set(args.epsilons)) != len(args.epsilons):
        parser.error("Budgets must be positive and representable; second-stage budgets must be distinct.")
    if not math.isfinite(args.growth) or args.growth <= 1:
        parser.error("Grid growth must be finite and greater than 1.")
    return args


def run_replications(task, replications, workers=1):
    """Collect one named DataFrame per replication, with identical worker streams."""
    def collect(results):
        frames = []
        for r, frame in enumerate(results, 1):
            frames.append(frame)
            print(f"replication {r}/{replications}", flush=True)
        return pd.concat(frames, ignore_index=True)
    indices = range(1, replications + 1)
    if workers == 1:
        return collect(map(task, indices))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return collect(pool.map(task, indices, chunksize=1))


def summarize_simulation_quantiles(raw, epsilon_column, ratio_column):
    """One shared implementation of the two manuscript panels."""
    panel_a, panel_b = [], []
    for (n, epsilon), group in raw.groupby(["n", epsilon_column], sort=True):
        common = {"n": int(n), epsilon_column: float(epsilon),
                  "epsilon_total": float(group.epsilon_total.iloc[0]), "replications": len(group)}
        a = dict(common)
        for component in ("total", "mechanism", "clipping", "noise"):
            field = f"absolute_{component}_error"
            a[f"median_{field}"] = float(np.median(group[field]))
            a[f"q95_{field}"] = float(np.quantile(group[field], .95))
        a["q90_absolute_total_error"] = float(np.quantile(group.absolute_total_error, .9))
        for field in ("noise_scale", "n_clipped", "fraction_clipped"):
            a[f"median_{field}"] = float(np.median(group[field]))
        ratios = group[ratio_column].dropna()
        panel_a.append(a)
        panel_b.append({**common,
            "median_absolute_sampling_error": float(np.median(group.absolute_sampling_error)),
            "q95_absolute_sampling_error": float(np.quantile(group.absolute_sampling_error, .95)),
            f"median_{ratio_column}": float(np.median(ratios)) if len(ratios) else np.nan,
            "ratio_valid_replications": len(ratios),
            "ratio_zero_gs_replications": len(group) - len(ratios),
        })
    return pd.DataFrame(panel_a), pd.DataFrame(panel_b)


def save_simulation_report(raw, args, statistic, population, config):
    """Always retain plot inputs; full per-draw diagnostic records are opt-in."""
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    name = statistic.lower()
    epsilon = f"epsilon_{name}"
    ratio = {"MLD": "ss_over_gs", "Theil": "smooth_bound_over_global_bound"}[statistic]
    panel_a, panel_b = summarize_simulation_quantiles(raw, epsilon, ratio)
    _save_plot_data(output, {"kind": "simulation", "statistic": statistic},
                    panel_a=panel_a.to_records(index=False),
                    panel_b=panel_b.to_records(index=False))
    if args.save_details:
        (output / "details").mkdir(exist_ok=True)
        raw.to_csv(output / "details/raw_results.csv", index=False)
    moments = summarize_simulation_moments(raw, epsilon)
    save_simulation_paper_tables(panel_a, panel_b, moments, args, statistic, population, ratio)
    if not args.no_plots:
        plot_simulation_results(panel_a, panel_b, epsilon_column=epsilon,
            ratio_column=f"median_{ratio}", statistic=statistic, output_dir=output,
            ratio_label={"MLD": "exact SS / exact GS",
                         "Theil": "smooth upper bound / exact GS"}[statistic])
    if not args.save_details:
        return
    output = output / "details"
    panel_a.to_csv(output / "table_2_panel_a.csv", index=False)
    panel_b.to_csv(output / "table_2_panel_b.csv", index=False)
    pd.DataFrame([population]).to_csv(output / f"population_{name}.csv", index=False)
    settings = {key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()}
    write_run_config(output / "config.json", {**settings, **config, "statistic": statistic,
        "population": population, "plots_enabled": not args.no_plots,
        "lower_bound": 1., "income_shift": 1., "gamma": 2.,
        "alpha": "epsilon / 8", "beta": "epsilon / 2", "threshold_budget_fraction": .5,
        "noise_distribution": "standard Cauchy", "raw_columns": list(raw.columns),
        "zero_gs_ratio": "NaN; excluded from ratio median and counted separately"})
    moments.to_csv(output / "table_3_moments.csv", index=False)


def _save_plot_data(output_dir, metadata, **arrays):
    """Atomically store full-precision plot inputs without Python/pickle objects."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "plot_data.npz"
    metadata = {"schema_version": 1, **metadata}
    with tempfile.NamedTemporaryFile(dir=output, suffix=".npz", delete=False) as handle:
        temporary = Path(handle.name)
        try:
            np.savez_compressed(handle, metadata=json.dumps(metadata, allow_nan=False), **arrays)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def save_real_plot_data(accuracy, diagnostics, releases, *, statistic, sample, output_dir):
    """Retain every release for exact boxplots, even without --save-details."""
    epsilons = sorted(releases)
    _save_plot_data(output_dir, {
        "kind": "real", "statistic": statistic, "sample": sample,
        "true_value": float(diagnostics.iloc[0][f"true_{statistic.lower()}"]),
    }, accuracy=accuracy.to_records(index=False),
       diagnostics=diagnostics[["bootstrap_sd", "median_absolute_clipping_error"]].to_records(index=False),
       epsilons=np.asarray(epsilons, dtype=float),
       **{f"releases_{i}": np.asarray(releases[e], dtype=float) for i, e in enumerate(epsilons)})


def replot_report(results_dir, *, output_dir=None, statistic=None, sample=None):
    """Redraw figures from saved results without rerunning any experiment."""
    source = Path(results_dir)
    output = source if output_dir is None else Path(output_dir)
    cache_path = source / "plot_data.npz"
    if cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as cache:
            metadata = json.loads(str(cache["metadata"].item()))
            if metadata.get("schema_version") != 1:
                raise ValueError(f"Unsupported plot data version: {cache_path}")
            if statistic is not None and metadata["statistic"] != statistic:
                raise ValueError(f"Expected {statistic} results in {source}.")
            if sample is not None and metadata.get("sample") != sample:
                raise ValueError(f"Expected {sample} results in {source}.")
            statistic = metadata["statistic"]
            if metadata["kind"] == "real":
                accuracy = pd.DataFrame.from_records(cache["accuracy"])
                diagnostics = pd.DataFrame.from_records(cache["diagnostics"])
                releases = {float(e): cache[f"releases_{i}"]
                            for i, e in enumerate(cache["epsilons"])}
                plot_real_accuracy(accuracy, diagnostics,
                    epsilon_column=f"epsilon_{statistic.lower()}", statistic=statistic,
                    sample=metadata["sample"], output_dir=output)
                plot_real_releases(releases, metadata["true_value"], statistic=statistic,
                    sample=metadata["sample"], output_dir=output)
                return
            if metadata["kind"] != "simulation":
                raise ValueError(f"Unknown plot data kind in {cache_path}.")
            panel_a = pd.DataFrame.from_records(cache["panel_a"])
            panel_b = pd.DataFrame.from_records(cache["panel_b"])
    else:
        main_path = source / "main_table.csv"
        if not main_path.exists():
            raise FileNotFoundError(f"No saved plot data in {source}. Run the experiment once first.")
        panel_a = pd.read_csv(main_path, float_precision="round_trip")
        if "n" not in panel_a.columns or sample is not None:
            raise FileNotFoundError(
                f"Missing {cache_path}. Legacy real-data summary tables cannot reconstruct "
                "boxplots; run the updated experiment once to save all plot inputs. "
                "No experiment was rerun.")
        inferred = "MLD" if "epsilon_mld" in panel_a.columns else "Theil"
        if statistic is not None and statistic != inferred:
            raise ValueError(f"Expected {statistic} results in {source}.")
        statistic = inferred
        panel_b = pd.read_csv(source / "appendix_tables.csv", float_precision="round_trip")
    ratio = {"MLD": "ss_over_gs", "Theil": "smooth_bound_over_global_bound"}[statistic]
    plot_simulation_results(panel_a, panel_b, epsilon_column=f"epsilon_{statistic.lower()}",
        ratio_column=f"median_{ratio}", statistic=statistic, output_dir=output,
        ratio_label={"MLD": "exact SS / exact GS",
                     "Theil": "smooth upper bound / exact GS"}[statistic])


def _pyplot():
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "dp_inequality_mpl"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _save_figure(fig, directory: str | Path, basename: str, formats=("png",)) -> None:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for extension in formats:
        fig.savefig(directory / f"{basename}.{extension}", dpi=600, bbox_inches="tight",
                    facecolor="white")


def plot_real_accuracy(
    accuracy: pd.DataFrame, diagnostics: pd.DataFrame, *,
    epsilon_column: str, statistic: str, sample: str, output_dir: str | Path,
) -> None:
    """Plot median and Q95 AE relative to the ordinary-bootstrap reference."""
    plt = _pyplot()
    diag = diagnostics.iloc[0]
    sd = float(diag.bootstrap_sd)
    frame = accuracy.sort_values(epsilon_column)
    budget = frame[epsilon_column]
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    if math.isfinite(sd) and sd > 0:
        denominator = sd
        ax.axhline(1., color="black", ls=":", label="Ordinary-bootstrap SD")
        ylabel = "Absolute error / ordinary-bootstrap SD"
    else:
        denominator = 1.
        ylabel = "Absolute error"
        ax.text(.02, .97, "Bootstrap SD is zero; showing unnormalized errors.",
                transform=ax.transAxes, va="top", fontsize=8)
    ax.plot(budget, frame.median_absolute_error / denominator, "o-",
            color="#0072B2", label="Median AE")
    ax.plot(budget, frame.q95_absolute_error / denominator, "s--",
            color="#D55E00", label="95th percentile AE")
    clipping = float(diag.median_absolute_clipping_error) / denominator
    ax.axhline(clipping, color="#009E73", ls="-.", label="Median clipping AE")
    plotted = np.r_[frame.median_absolute_error, frame.q95_absolute_error, clipping]
    if np.all(plotted > 0):
        ax.set_yscale("log")
    ax.set(xlabel=rf"$\epsilon_{{\mathrm{{{statistic}}}}}$", ylabel=ylabel)
    ax.set_xticks(budget, [f"{value:g}" for value in budget])
    ax.grid(axis="y", alpha=.2)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    _save_figure(fig, output_dir, f"{sample}_{statistic.lower()}_accuracy")
    plt.close(fig)


def plot_real_releases(releases, true_value, *, statistic, sample, output_dir):
    """Show the original release boxplots, hiding fliers only in the figure."""
    plt = _pyplot()
    epsilons = sorted(releases)
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.boxplot([releases[epsilon] for epsilon in epsilons],
               tick_labels=[f"{epsilon:g}" for epsilon in epsilons],
               whis=1.5, showfliers=False, patch_artist=True,
               boxprops={"facecolor": "#8CB9D9", "edgecolor": "#244A64"},
               medianprops={"color": "#7A1F1F", "linewidth": 1.5})
    ax.axhline(true_value, color="black", ls="--",
               label=f"Nonprivate sample {statistic} = {true_value:.3f}")
    ax.set(xlabel=rf"$\epsilon_{{\mathrm{{{statistic}}}}}$", ylabel=f"Released {statistic}")
    ax.grid(axis="y", alpha=.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    _save_figure(fig, output_dir, f"{sample}_{statistic.lower()}_boxplots")
    plt.close(fig)


def plot_simulation_results(
    panel_a: pd.DataFrame, panel_b: pd.DataFrame, *,
    epsilon_column: str, ratio_column: str, statistic: str,
    ratio_label: str, output_dir: str | Path,
) -> None:
    """Save total-error curves and correctly labelled sensitivity-ratio curves."""
    plt = _pyplot()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for epsilon, group in panel_a.groupby(epsilon_column, sort=True):
        group = group.sort_values("n")
        for ax, field in zip(axes, ("median_absolute_total_error", "q95_absolute_total_error")):
            ax.plot(group.n, group[field], marker="o", label=f"stage-2 epsilon = {epsilon:g}")
    reference = panel_b.drop_duplicates("n").sort_values("n")
    axes[0].plot(reference.n, reference.median_absolute_sampling_error, "k--",
                 label="Median sampling error")
    axes[1].plot(reference.n, reference.q95_absolute_sampling_error, "k--",
                 label="95th percentile sampling error")
    for ax in axes:
        ax.set(xscale="log", xlabel="Sample size n", ylabel="Absolute error")
        if all(np.all(np.asarray(line.get_ydata()) > 0) for line in ax.lines):
            ax.set_yscale("log")
        ax.grid(alpha=.2)
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    _save_figure(fig, output_dir, f"{statistic.lower()}_simulation_errors")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    for epsilon, group in panel_b.groupby(epsilon_column, sort=True):
        group = group.sort_values("n")
        ax.plot(group.n, group[ratio_column], marker="o", label=f"stage-2 epsilon = {epsilon:g}")
    ax.set(xscale="log", xlabel="Sample size n", ylabel=f"Median {ratio_label}")
    if np.all(panel_b[ratio_column] > 0):
        ax.set_yscale("log")
    ax.grid(alpha=.2)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    _save_figure(fig, output_dir, f"{statistic.lower()}_sensitivity_ratio")
    plt.close(fig)


def load_simulation_panels(results_dir):
    """Return ``(statistic, panel_a, panel_b)`` from a saved simulation report."""
    source = Path(results_dir)
    cache_path = source / "plot_data.npz"
    if cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as cache:
            metadata = json.loads(str(cache["metadata"].item()))
            if metadata.get("schema_version") != 1 or metadata.get("kind") != "simulation":
                raise ValueError(f"{cache_path} does not hold simulation plot data.")
            return (metadata["statistic"], pd.DataFrame.from_records(cache["panel_a"]),
                    pd.DataFrame.from_records(cache["panel_b"]))
    main_path = source / "main_table.csv"
    if not main_path.exists():
        raise FileNotFoundError(f"No saved simulation results in {source}.")
    panel_a = pd.read_csv(main_path, float_precision="round_trip")
    if "n" not in panel_a.columns:
        raise ValueError(f"{main_path} is not a simulation summary table.")
    panel_b = pd.read_csv(source / "appendix_tables.csv", float_precision="round_trip")
    return ("MLD" if "epsilon_mld" in panel_a.columns else "Theil"), panel_a, panel_b


# Okabe-Ito hues plus distinct markers, so budgets stay apart in grayscale and for CVD.
_EPSILON_STYLES = (("#0072B2", "o"), ("#D55E00", "s"), ("#009E73", "^"), ("#CC79A7", "D"))
_SIMULATION_COLUMNS = (
    ("Median absolute error", "median_absolute_total_error", "median_absolute_sampling_error"),
    ("95th percentile of absolute error", "q95_absolute_total_error",
     "q95_absolute_sampling_error"),
)
_PAPER_STYLE = {
    "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8.5,
    "axes.linewidth": .6, "axes.edgecolor": "0.25",
    "xtick.direction": "out", "ytick.direction": "out",
    "xtick.major.width": .6, "ytick.major.width": .6,
    "xtick.minor.width": .4, "ytick.minor.width": .4,
    "xtick.major.size": 3, "ytick.major.size": 3,
    "xtick.minor.size": 1.8, "ytick.minor.size": 1.8,
}


def plot_simulation_figure(reports, output_dir, basename="simulation_errors"):
    """One publication grid: a row per index, median and Q95 columns, one shared legend.

    ``reports`` is a sequence of ``(statistic, panel_a, panel_b)`` tuples, e.g. from
    :func:`load_simulation_panels`. Every row must use the same stage-2 budgets.
    """
    plt = _pyplot()
    budgets = [sorted(panel_a[f"epsilon_{statistic.lower()}"].unique())
               for statistic, panel_a, _ in reports]
    if any(b != budgets[0] for b in budgets) or len(budgets[0]) > len(_EPSILON_STYLES):
        raise ValueError("All reports need the same (at most four) stage-2 budgets.")
    with plt.rc_context(_PAPER_STYLE):
        fig, axes = plt.subplots(len(reports), 2, figsize=(6.5, 2.45 * len(reports) + .45),
                                 sharex=True, squeeze=False)
        for row, (statistic, panel_a, panel_b) in enumerate(reports):
            epsilon_column = f"epsilon_{statistic.lower()}"
            reference = panel_b.drop_duplicates("n").sort_values("n")
            for column, (title, total, sampling) in enumerate(_SIMULATION_COLUMNS):
                ax = axes[row, column]
                for (color, marker), epsilon in zip(_EPSILON_STYLES, budgets[0]):
                    group = panel_a[panel_a[epsilon_column] == epsilon].sort_values("n")
                    ax.plot(group.n, group[total], color=color, marker=marker, lw=1.3, ms=4.8,
                            mec="white", mew=.6, zorder=3, label=rf"$\varepsilon_f = {epsilon:g}$")
                ax.plot(reference.n, reference[sampling], color="0.2", ls=(0, (4, 2)),
                        lw=1.1, zorder=2, label="Sampling error (non-private)")
                ax.set(xscale="log", yscale="log")
                ax.grid(which="major", color="0.9", lw=.5)
                ax.set_axisbelow(True)
                ax.spines[["top", "right"]].set_visible(False)
                ax.margins(x=.06, y=.08)
                letter = "abcdefghijklmnopqrstuvwxyz"[2 * row + column]
                ax.text(.98, .97, f"({letter}) {statistic}", transform=ax.transAxes,
                        ha="right", va="top", fontsize=9)
                if row == 0:
                    ax.set_title(title, pad=6)
                if row == len(reports) - 1:
                    ax.set_xlabel("Sample size $n$")
                if column == 0:
                    ax.set_ylabel("Absolute error")
        _finish_paper_grid(fig, *axes[0, 0].get_legend_handles_labels(), output_dir, basename)
        plt.close(fig)


def _finish_paper_grid(fig, handles, labels, output_dir, basename):
    """Place one shared legend under the grid (none if no handles), then save PNG and PDF."""
    bottom = 0.
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=len(handles), frameon=False,
                   bbox_to_anchor=(.5, 0.), handlelength=2.2, columnspacing=1.6)
        bottom = .17 / fig.get_figheight()
    fig.tight_layout(rect=(0, bottom, 1, 1), h_pad=1.2, w_pad=1.5)
    _save_figure(fig, output_dir, basename, formats=("png", "pdf"))


def load_real_releases(results_dir):
    """Return ``(statistic, sample, true_value, {epsilon: releases})`` from a real-data cache."""
    cache_path = Path(results_dir) / "plot_data.npz"
    if not cache_path.exists():
        raise FileNotFoundError(f"Missing {cache_path}. Run the experiment once first.")
    with np.load(cache_path, allow_pickle=False) as cache:
        metadata = json.loads(str(cache["metadata"].item()))
        if metadata.get("schema_version") != 1 or metadata.get("kind") != "real":
            raise ValueError(f"{cache_path} does not hold real-data plot data.")
        releases = {float(e): cache[f"releases_{i}"] for i, e in enumerate(cache["epsilons"])}
    return metadata["statistic"], metadata["sample"], metadata["true_value"], releases


_INDEX_SYMBOLS = {"MLD": "M", "Theil": "T"}


def plot_release_figure(reports, output_dir, basename="release_boxplots"):
    """Release boxplots on one grid: rows are samples, columns are indices.

    ``reports`` holds :func:`load_real_releases` tuples in row-major order, two per
    row. Whiskers span 1.5 IQR and fliers are hidden, as in ``plot_real_releases``.
    """
    plt = _pyplot()
    face, edge, median_color = "#A9CBE3", "#2B5577", "#C0392B"
    rows = math.ceil(len(reports) / 2)
    # Wider, flatter panels and type one step smaller than the simulation grid.
    style = {**_PAPER_STYLE, "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
             "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7.5}
    with plt.rc_context(style):
        fig, axes = plt.subplots(rows, 2, figsize=(7.2, 2.15 * rows + .1), squeeze=False)
        for index, (statistic, sample, true_value, releases) in enumerate(reports):
            row, column = divmod(index, 2)
            ax = axes[row, column]
            epsilons = sorted(releases)
            ax.boxplot([releases[e] for e in epsilons], tick_labels=[f"{e:g}" for e in epsilons],
                       whis=1.5, showfliers=False, patch_artist=True, widths=.55,
                       boxprops={"facecolor": face, "edgecolor": edge, "linewidth": .8},
                       whiskerprops={"color": edge, "linewidth": .8},
                       capprops={"color": edge, "linewidth": .8},
                       medianprops={"color": median_color, "linewidth": 1.3})
            ax.axhline(true_value, color="0.2", ls=(0, (4, 2)), lw=1., zorder=4)
            ax.grid(axis="y", color="0.9", lw=.5)
            ax.set_axisbelow(True)
            ax.spines[["top", "right"]].set_visible(False)
            letter = "abcdefghijklmnopqrstuvwxyz"[index]
            symbol = _INDEX_SYMBOLS.get(statistic, statistic)
            ax.text(.98, .97, f"({letter}) {sample.capitalize()} income, "
                    rf"${symbol}(X) = {true_value:.3f}$", transform=ax.transAxes,
                    ha="right", va="top", fontsize=7.5)
            if row == 0:
                ax.set_title(statistic, pad=6)
            if row == rows - 1:
                ax.set_xlabel(r"$\varepsilon_f$")
            if column == 0:
                ax.set_ylabel("Released value")
        for ax in axes.flat[len(reports):]:
            ax.set_visible(False)
        # No legend: the paper caption explains the boxes and the dashed line.
        _finish_paper_grid(fig, [], [], output_dir, basename)
        plt.close(fig)


def _save_summary_csv(frame, path):
    """Save computed values without rounding for presentation."""
    frame.to_csv(path, index=False)


def save_real_paper_tables(accuracy, comparison, diagnostics, output, statistic, sample, seed):
    """Save one budget table and a tidy appendix with clipping/oracle summaries."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    _save_summary_csv(accuracy, output / "main_table.csv")
    summary = diagnostics.assign(statistic=statistic, sample=sample, seed=seed).melt(
        var_name="metric", value_name="value")
    summary.insert(0, "section", "sample_and_clipping")
    summary.insert(1, "upper_bound_strategy", "")
    oracle = comparison.melt(id_vars="upper_bound_strategy", var_name="metric", value_name="value")
    oracle.insert(0, "section", "oracle_comparison_epsilon2_1")
    _save_summary_csv(pd.concat([summary, oracle], ignore_index=True), output / "appendix_tables.csv")


def save_simulation_paper_tables(panel_a, panel_b, moments, args, statistic, population, ratio):
    """Save one row per (n, budget), keeping moments beside the primary errors."""
    output = args.output_dir
    epsilon = f"epsilon_{statistic.lower()}"
    keys = ["n", epsilon]
    main_errors = panel_a[keys + ["median_absolute_total_error", "q95_absolute_total_error"]]
    main = main_errors.merge(moments, on=keys, validate="one_to_one")
    _save_summary_csv(main, output / "main_table.csv")
    appendix = panel_a.drop(columns=["median_absolute_total_error", "q95_absolute_total_error"])
    appendix = appendix.merge(panel_b.drop(columns=["epsilon_total", "replications"]),
                              on=keys, validate="one_to_one")
    appendix["median_clipped_percent"] = 100 * appendix.median_fraction_clipped
    appendix["epsilon_u"] = args.epsilon_u
    appendix["grid_growth"] = args.growth
    appendix["seed"] = args.seed
    for key, value in population.items():
        appendix[key] = value
    appendix["sensitivity_ratio_definition"] = (
        "exact SS / exact GS" if statistic == "MLD" else "paper smooth upper bound / exact GS")
    appendix["dp_moment_interpretation"] = CAUCHY_MOMENT_NOTE
    _save_summary_csv(appendix, output / "appendix_tables.csv")
