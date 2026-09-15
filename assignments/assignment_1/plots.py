"""The figures for the report, sized for one column of the GECCO template."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from conditions import Condition
from problem import TARGET_SIZES

COLUMN_WIDTH = 3.33


def set_plot_style() -> None:
    plt.switch_backend("Agg")  # files only, no window
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


def save(fig: Figure, figures: Path, name: str) -> None:
    fig.savefig(figures / f"{name}.pdf")
    fig.savefig(figures / f"{name}.png", dpi=200)
    plt.close(fig)


def legend_above(ax: Axes) -> None:
    ax.legend(
        loc="lower left",
        bbox_to_anchor=(0, 1.02),
        ncol=2,
        borderaxespad=0,
        handlelength=1.6,
        columnspacing=1.2,
    )


def plot_mean_and_std(
    ax: Axes, data: pd.DataFrame, x: str, y: str, conditions: list[Condition]
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
    # random search logs each batch of pop_size bodies as one generation
    plot_mean_and_std(ax, data, "generation", "best_fitness", conditions)
    ax.set_xlabel("generation")
    ax.set_ylabel("best fitness (lower is better)")
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
    for offset, condition in zip(offsets, eas, strict=True):
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


def make_all_figures(
    data: pd.DataFrame,
    final: pd.DataFrame,
    conditions: list[Condition],
    figures: Path,
) -> None:
    set_plot_style()
    plot_fitness_curve(data, conditions, figures)
    plot_per_generation(
        data,
        conditions,
        "diversity",
        "diversity (mean pairwise dist.)",
        "diversity_curve",
        figures,
    )
    plot_per_generation(
        data,
        conditions,
        "distinct_parents",
        "different parents per generation",
        "selection_strength",
        figures,
    )
    plot_closest_per_target(final, conditions, figures)
    plot_final_fitness(final, conditions, figures)
