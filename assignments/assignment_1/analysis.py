"""Figures and statistics for the report.

Reads results/<condition>/seed_XX/ and writes to figures/:
  fitness_curve        best fitness against evaluations (mean and std over seeds)
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
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")  
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from problem import TARGET_SIZES
from records import FIGURES_DIR, RESULTS_DIR, check_consistent, load_histories

# Each condition gets a colour and a marker by its role in the experiment.
COLORS = {
    "baseline": "#2a78d6",
    "lexicase": "#eb6834",
    "control": "#1baf7a",
    "random": "#52514e",
    "other": "#898781",
}
MARKERS = {
    "baseline": "o",
    "lexicase": "s",
    "control": "^",
    "random": "D",
    "other": "v",
}
ROLE_ORDER = ["baseline", "lexicase", "control", "random", "other"]
COLUMN_WIDTH = 3.33  


@dataclass
class Condition:
    name: str
    role: str
    label: str
    short_label: str 

    @property
    def color(self) -> str:
        return COLORS[self.role]

    @property
    def marker(self) -> str:
        return MARKERS[self.role]


def find_conditions(data: pd.DataFrame, control: str | None) -> list[Condition]:
    """Work out each condition's role from its config.

    The tournament with k = 2 is the baseline. The control is the one other
    tournament size, or the one given with --control.
    """
    runs = data.drop_duplicates("condition").set_index("condition")
    tournaments = runs[runs["selection"] == "tournament"]
    larger = [name for name, row in tournaments.iterrows() if row["k"] != 2]
    if control is None:
        if len(larger) > 1:
            raise SystemExit(
                f"several tournament sizes found ({', '.join(larger)}); pick one with --control"
            )
        control = larger[0] if larger else None

    conditions = []
    for name, row in runs.iterrows():
        if row["selection"] == "lexicase":
            conditions.append(Condition(name, "lexicase", "Lexicase", "Lexicase"))
        elif row["selection"] == "random":
            conditions.append(
                Condition(name, "random", "Random search", "Random\nsearch")
            )
        elif int(row["k"]) == 2:
            conditions.append(
                Condition(name, "baseline", "Tournament (k = 2)", "Tournament\nk = 2")
            )
        elif name == control:
            k = int(row["k"])
            conditions.append(
                Condition(
                    name,
                    "control",
                    f"Tournament (k = {k}, matched)",
                    f"Tournament\nk = {k}",
                )
            )
        else:
            conditions.append(Condition(name, "other", name, name))
    conditions.sort(key=lambda c: (ROLE_ORDER.index(c.role), c.name))
    return conditions


def final_generation(data: pd.DataFrame) -> pd.DataFrame:
    last = data.groupby(["condition", "seed"])["generation"].transform("max")
    return data[data["generation"] == last]


# --- figures ---


def set_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 8,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.labelcolor": "0.3",
            "xtick.color": "0.45",
            "ytick.color": "0.45",
            "axes.edgecolor": "0.75",
            "axes.linewidth": 0.75,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.color": "0.9",
            "grid.linewidth": 0.5,
            "lines.linewidth": 1.5,
            "legend.frameon": False,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,  
        }
    )


def save(fig, figures: Path, name: str) -> None:
    fig.savefig(figures / f"{name}.pdf")
    fig.savefig(figures / f"{name}.png", dpi=200)
    plt.close(fig)


def legend_above(ax) -> None:
    ax.legend(
        loc="lower left",
        bbox_to_anchor=(0, 1.02),
        ncol=2,
        borderaxespad=0,
        handlelength=1.6,
        columnspacing=1.2,
    )


def plot_mean_and_std(
    ax, data: pd.DataFrame, x: str, y: str, conditions: list[Condition]
) -> None:
    for condition in conditions:
        per_x = data[data["condition"] == condition.name].groupby(x)[y]
        mean = per_x.mean()
        std = per_x.std().fillna(0.0)  # std is undefined with a single seed
        ax.fill_between(
            mean.index,
            mean - std,
            mean + std,
            color=condition.color,
            alpha=0.12,
            linewidth=0,
        )
        ax.plot(
            mean.index, mean.to_numpy(), color=condition.color, label=condition.label
        )


def plot_fitness_curve(
    data: pd.DataFrame, conditions: list[Condition], figures: Path
) -> None:
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, 2.3))
    plot_mean_and_std(ax, data, "evaluations", "best_fitness", conditions)
    ax.set_xlabel("evaluations")
    ax.set_ylabel("best fitness (lower is better)")
    ax.xaxis.set_major_formatter(mpl.ticker.StrMethodFormatter("{x:,.0f}"))
    legend_above(ax)
    save(fig, figures, "fitness_curve")


def plot_per_generation(
    data: pd.DataFrame,
    conditions: list[Condition],
    column: str,
    ylabel: str,
    name: str,
    figures: Path,
) -> None:
    eas = [c for c in conditions if c.role != "random"]
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, 2.1))
    plot_mean_and_std(ax, data.dropna(subset=[column]), "generation", column, eas)
    ax.set_xlabel("generation")
    ax.set_ylabel(ylabel)
    ax.set_ylim(bottom=0)
    legend_above(ax)
    save(fig, figures, name)


def plot_closest_per_target(
    final: pd.DataFrame, conditions: list[Condition], figures: Path
) -> None:
    eas = [c for c in conditions if c.role != "random"]
    columns = [f"closest_{size}" for size in TARGET_SIZES]
    rows = np.arange(len(TARGET_SIZES))
    offsets = np.linspace(-0.22, 0.22, len(eas)) if len(eas) > 1 else [0.0]

    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, 2.4))
    for offset, condition in zip(offsets, eas):
        values = final[final["condition"] == condition.name][columns]
        ax.errorbar(
            values.mean().to_numpy(),
            rows + offset,
            xerr=values.std().fillna(0.0).to_numpy(),
            fmt=condition.marker,
            color=condition.color,
            markersize=4.5,
            markeredgecolor="white",
            markeredgewidth=0.8,
            elinewidth=1,
            label=condition.label,
        )
    ax.set_yticks(rows, [f"{size}-node target" for size in TARGET_SIZES])
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    ax.set_xlim(left=0)
    ax.set_xlabel("closest body in final population (edit distance)")
    legend_above(ax)
    save(fig, figures, "closest_per_target")


def plot_final_fitness(
    final: pd.DataFrame, conditions: list[Condition], figures: Path
) -> None:
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, 2.3))
    jitter = np.random.default_rng(0)
    for i, condition in enumerate(conditions):
        values = final[final["condition"] == condition.name]["best_fitness"].to_numpy()
        ax.boxplot(
            values,
            positions=[i],
            widths=0.5,
            showfliers=False,
            medianprops={"color": "black", "linewidth": 1},
            boxprops={"color": "0.75"},
            whiskerprops={"color": "0.75"},
            capprops={"color": "0.75"},
        )
        x = i + jitter.uniform(-0.12, 0.12, len(values))
        ax.scatter(
            x,
            values,
            s=18,
            color=condition.color,
            marker=condition.marker,
            edgecolors="white",
            linewidths=0.8,
            zorder=3,
        )
    ax.set_xticks(range(len(conditions)), [c.short_label for c in conditions])
    ax.set_ylabel("final best fitness (lower is better)")
    save(fig, figures, "final_fitness")


# --- statistics ---


def a12(x: np.ndarray, y: np.ndarray) -> float:
    """Vargha-Delaney A12: chance that a run from x has a higher value than a run from y."""
    greater = (x[:, None] > y[None, :]).sum()
    equal = (x[:, None] == y[None, :]).sum()
    return float((greater + 0.5 * equal) / (len(x) * len(y)))


def holm(p_values: list[float]) -> list[float]:
    """Holm-Bonferroni correction for multiple tests."""
    order = np.argsort(p_values)
    m = len(p_values)
    adjusted = np.empty(m)
    largest = 0.0
    for rank, i in enumerate(order):
        largest = max(largest, (m - rank) * p_values[i])
        adjusted[i] = min(largest, 1.0)
    return adjusted.tolist()


def compare(
    final: pd.DataFrame, hypothesis: str, metric: str, a: str, b: str
) -> dict | None:
    x = final[final["condition"] == a][metric].to_numpy(dtype=float)
    y = final[final["condition"] == b][metric].to_numpy(dtype=float)
    if len(x) < 2 or len(y) < 2:
        return None
    u, p = mannwhitneyu(x, y, alternative="two-sided")
    return {
        "hypothesis": hypothesis,
        "metric": metric,
        "a": a,
        "b": b,
        "n_a": len(x),
        "n_b": len(y),
        "mean_a": x.mean(),
        "std_a": x.std(ddof=1),
        "mean_b": y.mean(),
        "std_b": y.std(ddof=1),
        "U": u,
        "p": p,
        "a12": a12(x, y),
    }


def run_statistics(final: pd.DataFrame, conditions: list[Condition]) -> pd.DataFrame:
    name_of = {c.role: c.name for c in conditions if c.role != "other"}
    eas = [c.name for c in conditions if c.role != "random"]
    lexicase = name_of.get("lexicase")
    tournaments = [name_of[role] for role in ("baseline", "control") if role in name_of]

    tests = []
    if "random" in name_of:
        for ea in eas:
            tests.append(("H0 EA vs random", "best_fitness", ea, name_of["random"]))
    if lexicase:
        for other in tournaments:
            tests.append(("H1 diversity", "diversity", lexicase, other))
        for other in tournaments:
            for size in TARGET_SIZES:
                tests.append(("H2 specialists", f"closest_{size}", lexicase, other))
        for other in tournaments:
            tests.append(("H3 combined fitness", "best_fitness", lexicase, other))
    if "control" in name_of and "baseline" in name_of:
        tests.append(
            (
                "H4 control diversity",
                "diversity",
                name_of["control"],
                name_of["baseline"],
            )
        )

    rows = []
    for test in tests:
        result = compare(final, *test)
        if result is not None:
            rows.append(result)
    stats = pd.DataFrame(rows)
    if not stats.empty:
        stats["p_holm"] = stats.groupby("hypothesis")["p"].transform(
            lambda p: holm(p.tolist())
        )
    return stats


def summarise(
    data: pd.DataFrame, final: pd.DataFrame, conditions: list[Condition]
) -> pd.DataFrame:
    columns = ["best_fitness", "best_size", "mean_size", "diversity"]
    columns += [f"closest_{size}" for size in TARGET_SIZES]
    summary = final.groupby("condition")[columns].agg(["mean", "std"])
    summary[("distinct_parents_over_run", "mean")] = data.groupby("condition")[
        "distinct_parents"
    ].mean()
    summary[("seeds", "n")] = final.groupby("condition")["seed"].nunique()
    return summary.loc[[c.name for c in conditions]]


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

    set_plot_style()
    plot_fitness_curve(data, conditions, args.figures)
    plot_per_generation(
        data,
        conditions,
        "diversity",
        "diversity (mean pairwise dist.)",
        "diversity_curve",
        args.figures,
    )
    plot_per_generation(
        data,
        conditions,
        "distinct_parents",
        "different parents per generation",
        "selection_strength",
        args.figures,
    )
    plot_closest_per_target(final, conditions, args.figures)
    plot_final_fitness(final, conditions, args.figures)

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
                "\nMann-Whitney U tests (p_holm is corrected within each hypothesis):"
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
                "a12",
            ]
            print(stats[columns].to_string(index=False))
    print(f"\nsaved to {args.figures}")


if __name__ == "__main__":
    main()
