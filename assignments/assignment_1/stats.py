"""Statistical tests for the hypotheses and the summary table."""

from typing import Any, cast

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from conditions import Condition
from problem import TARGET_SIZES

# Metrics where a higher value is better; for fitness and distances lower wins.
HIGHER_IS_BETTER = {"diversity"}


def a12(x: np.ndarray, y: np.ndarray) -> float:
    """Vargha-Delaney A12: chance that a run from x is higher than a run from y."""
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
) -> dict[str, Any] | None:
    """Mann-Whitney U test of condition a against condition b on one metric.

    a12_a_better is the chance that a run of a beats a run of b, taking into
    account whether lower or higher is better for this metric.
    """
    x = final[final["condition"] == a][metric].to_numpy(dtype=float)
    y = final[final["condition"] == b][metric].to_numpy(dtype=float)
    if len(x) < 2 or len(y) < 2:
        return None
    u, p = mannwhitneyu(x, y, alternative="two-sided")
    lower_is_better = metric not in HIGHER_IS_BETTER
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
        "lower_is_better": lower_is_better,
        "a12_a_better": a12(y, x) if lower_is_better else a12(x, y),
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
    summary = cast(
        pd.DataFrame, final.groupby("condition")[columns].agg(["mean", "std"])
    )
    summary[("distinct_parents_over_run", "mean")] = data.groupby("condition")[
        "distinct_parents"
    ].mean()
    summary[("seeds", "n")] = final.groupby("condition")["seed"].nunique()
    return summary.loc[[c.name for c in conditions]]
