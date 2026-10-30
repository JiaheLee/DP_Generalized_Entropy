"""Regenerate figures from saved results without running any experiments."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from common.reporting import (
    load_real_releases, load_simulation_panels, plot_release_figure, plot_simulation_figure,
    replot_report,
)


PROJECT_DIR = Path(__file__).resolve().parent
SIMULATION_REPORTS = ("MLD/simulation/results", "Theil/simulation/results")
# Row-major: personal row, then household row; MLD left, Theil right.
RELEASE_REPORTS = (
    "MLD/application/results/personal_report", "Theil/application/results/personal_report",
    "MLD/application/results/household_report", "Theil/application/results/household_report",
)
DEFAULT_REPORTS = (
    "MLD/application/results/personal_report", "MLD/application/results/household_report",
    "Theil/application/results/personal_report", "Theil/application/results/household_report",
    "MLD/simulation/results", "Theil/simulation/results",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dirs", type=Path, nargs="*",
                        help="Report directories; defaults to all six project reports.")
    parser.add_argument("--output-dir", type=Path,
                        help="Collect figures in this directory instead of replacing report PNGs.")
    parser.add_argument("--combined-simulation", action="store_true",
                        help="Only draw the MLD+Theil simulation grid (PNG and PDF) into "
                             "--output-dir, default output/.")
    parser.add_argument("--combined-releases", action="store_true",
                        help="Only draw the 2x2 real-data release boxplots (PNG and PDF) into "
                             "--output-dir, default output/.")
    args = parser.parse_args()
    if args.combined_simulation and args.combined_releases:
        parser.error("Choose one of --combined-simulation and --combined-releases.")
    if args.combined_releases:
        sources = args.results_dirs or [PROJECT_DIR / path for path in RELEASE_REPORTS]
        output = args.output_dir or PROJECT_DIR / "output"
        plot_release_figure([load_real_releases(s) for s in sources], output)
        print(f"Saved release_boxplots.png/.pdf -> {output.resolve()}")
        return 0
    if args.combined_simulation:
        sources = args.results_dirs or [PROJECT_DIR / path for path in SIMULATION_REPORTS]
        output = args.output_dir or PROJECT_DIR / "output"
        plot_simulation_figure([load_simulation_panels(s) for s in sources], output)
        print(f"Saved simulation_errors.png/.pdf -> {output.resolve()}")
        return 0
    reports = args.results_dirs or [PROJECT_DIR / path for path in DEFAULT_REPORTS]
    failed = False
    for source in reports:
        try:
            replot_report(source, output_dir=args.output_dir)
        except (OSError, ValueError, KeyError) as error:
            print(f"Cannot replot {source}: {error}", file=sys.stderr)
            failed = True
            continue
        output = source if args.output_dir is None else args.output_dir
        print(f"Regenerated figures from {source} -> {output.resolve()}")
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
