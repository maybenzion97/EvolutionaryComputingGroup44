"""Figures and statistics for the report.

Reads results/<condition>/seed_XX/ and writes to figures/:
  fitness_curve        best fitness per generation (mean and std over seeds)
  diversity_curve      population diversity per generation
  selection_strength   different parents per generation
  closest_per_target   the closest body to each target at the end of the run
  final_fitness        final best fitness, one dot per seed
  summary.csv          final-generation mean and std per condition
  stats.csv            Mann-Whitney U tests (Holm-corrected) and A12 effect sizes

Usage (from the repository root):
    uv run python assignments/assignment_1/analysis.py
"""

import argparse
from pathlib import Path

import pandas as pd

from conditions import final_generation, find_conditions
from plots import make_all_figures
from records import FIGURES_DIR, RESULTS_DIR, check_consistent, load_histories
from stats import run_statistics, summarise


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Make the figures and statistics for the report."
    )
    parser.add_argument("--results", type=Path, default=RESULTS_DIR)
    parser.add_argument("--figures", type=Path, default=FIGURES_DIR)
    parser.add_argument(
        "--control", help="condition to use as the control, e.g. tournament_k40"
    )
    args = parser.parse_args()
    args.figures.mkdir(parents=True, exist_ok=True)

    data = load_histories(args.results)
    check_consistent(data)
    conditions = find_conditions(data, args.control)
    final = final_generation(data)

    seeds = final.groupby("condition")["seed"].nunique()
    print("runs:", ", ".join(f"{c.name} ({seeds[c.name]} seeds)" for c in conditions))
    if (seeds < 5).any():
        print("warning: the assignment asks for at least 5 seeds per condition")

    make_all_figures(data, final, conditions, args.figures)

    summary = summarise(data, final, conditions)
    summary.to_csv(args.figures / "summary.csv")
    stats = run_statistics(final, conditions)
    stats.to_csv(args.figures / "stats.csv", index=False)

    with pd.option_context(
        "display.width", 200, "display.max_columns", 30, "display.precision", 3
    ):
        print("\nfinal generation, mean and std over seeds:")
        print(summary)
        if not stats.empty:
            print(
                "\nMann-Whitney U tests (p_holm is corrected within each hypothesis; "
                "a12_a_better > 0.5 means a beats b):"
            )
            columns = [
                "hypothesis",
                "metric",
                "a",
                "b",
                "mean_a",
                "mean_b",
                "p",
                "p_holm",
                "a12_a_better",
            ]
            print(stats[columns].to_string(index=False))
    print(f"\nsaved to {args.figures}")


if __name__ == "__main__":
    main()
