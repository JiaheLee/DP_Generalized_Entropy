"""Repeated-sampling Theil experiment using the existing calibrated GB2 DGP."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
import sys

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.reporting import (
    simulation_arguments, run_replications, save_simulation_report, replot_report,
    summarize_simulation_quantiles,
)

from Theil.core import smooth_upper_bounds_theil, theil
from common.population import CALIBRATION_PATH, GB2Population, SAMPLE_STREAMS, keyed_rng
from Theil.private_upper import clip_at_log_upper, sample_upper_bounds


# Named output columns; no positional indices are used by the report code.
RAW_COLUMNS = (
    "n", "epsilon_theil", "absolute_total_error", "absolute_sampling_error",
    "absolute_mechanism_error", "smooth_bound_over_global_bound",
    "replication", "epsilon_total", "population_value", "sample_value",
    "clipped_value", "released_value", "log_private_u", "global_sensitivity",
    "smooth_upper_bound", "noise_scale", "standard_noise",
    "signed_sampling_error", "signed_clipping_error", "signed_noise_error",
    "signed_mechanism_error", "signed_total_error", "absolute_clipping_error",
    "absolute_noise_error", "n_clipped", "fraction_clipped",
)


@dataclass(frozen=True)
class SimulationConfig:
    sample_sizes: tuple[int, ...] = (1_000, 10_000, 100_000, 1_000_000)
    epsilons: tuple[float, ...] = (.5, 1., 2.)
    replications: int = 1_000
    epsilon_u: float = .15
    growth: float = 1.01
    seed: int = 123


def simulate_replication(replication: int, *, config: SimulationConfig,
                         population: GB2Population, population_theil: float) -> pd.DataFrame:
    rows = []
    eps = np.array(config.epsilons)
    for n in config.sample_sizes:
        stream = SAMPLE_STREAMS.get(n, n+10)
        x = population.draw(n, keyed_rng(config.seed, replication, stream, 100))
        sample_theil = theil(x)
        x.sort()
        lu = float(sample_upper_bounds(x, config.epsilon_u, 1,
                   keyed_rng(config.seed, replication, stream, 200), growth=config.growth).log_upper[0])
        clipped = clip_at_log_upper(x, lu)
        center = theil(clipped)
        n_clipped = int(np.count_nonzero(clipped < x))
        smooth, global_sensitivity = smooth_upper_bounds_theil(clipped, eps/2, 1., log_upper=lu)
        for j, epsilon in enumerate(eps):
            noise = keyed_rng(config.seed, replication, stream, 300, j+1).standard_cauchy()
            scale = smooth[j]/(epsilon/8)
            released = center+scale*noise
            rows.append([n, epsilon, abs(released-population_theil),
                         abs(sample_theil-population_theil), abs(released-sample_theil),
                         smooth[j]/global_sensitivity if global_sensitivity > 0 else np.nan,
                         replication, epsilon+config.epsilon_u, population_theil,
                         sample_theil, center, released, lu, global_sensitivity,
                         smooth[j], scale, noise, sample_theil-population_theil,
                         center-sample_theil, released-center, released-sample_theil,
                         released-population_theil, abs(center-sample_theil),
                         abs(released-center), n_clipped, n_clipped/n])
    return pd.DataFrame(rows, columns=RAW_COLUMNS)


def summarize(raw, config):
    if not raw.groupby(["n", "epsilon_theil"]).size().eq(config.replications).all():
        raise ValueError("Incomplete simulation cell.")
    return summarize_simulation_quantiles(raw, "epsilon_theil", "smooth_bound_over_global_bound")


def main() -> None:
    args = simulation_arguments(__doc__, Path(__file__).resolve().parent / "results",
                                CALIBRATION_PATH, SimulationConfig().sample_sizes)
    if args.plot_only:
        replot_report(args.output_dir, statistic="Theil")
        print(f"Regenerated figures from saved results in {args.output_dir.resolve()}")
        return
    config = SimulationConfig(sample_sizes=tuple(sorted(args.sample_sizes)),
                              epsilons=tuple(args.epsilons), epsilon_u=args.epsilon_u,
                              growth=args.growth, replications=args.replications, seed=args.seed)
    population = GB2Population.load(args.calibration)
    moments = population.moments()
    task = partial(simulate_replication, config=config, population=population,
                   population_theil=moments["population_theil_shifted"])
    raw = run_replications(task, args.replications, args.workers)
    save_simulation_report(raw, args, "Theil", {**asdict(population), **moments}, {
        "calibration_source": str(args.calibration.resolve()),
        "sensitivity_ratio": "paper smooth upper bound / exact GS (not exact SS/GS)",
        "sample_streams": {str(n): SAMPLE_STREAMS.get(n, n+10) for n in config.sample_sizes},
        "random_streams": "SeedSequence(seed, replication, sample_stream, stage, epsilon_index)",
    })


if __name__ == "__main__":
    main()
