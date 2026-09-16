import numpy as np
import pandas as pd

from stats import a12, compare


def final_table(
    metric: str, a_values: list[float], b_values: list[float]
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "condition": ["a"] * len(a_values) + ["b"] * len(b_values),
            metric: a_values + b_values,
        }
    )


def test_a12_counts_ties_as_half():
    assert a12(np.array([1.0, 2.0]), np.array([2.0])) == 0.25


def test_lower_fitness_counts_as_better():
    table = final_table("best_fitness", [1.0, 2.0, 3.0], [4.0, 5.0, 6.0])
    result = compare(table, "H3", "best_fitness", "a", "b")
    assert result is not None
    assert result["lower_is_better"]
    assert result["a12_a_better"] == 1.0


def test_higher_diversity_counts_as_better():
    table = final_table("diversity", [4.0, 5.0, 6.0], [1.0, 2.0, 3.0])
    result = compare(table, "H1", "diversity", "a", "b")
    assert result is not None
    assert not result["lower_is_better"]
    assert result["a12_a_better"] == 1.0
