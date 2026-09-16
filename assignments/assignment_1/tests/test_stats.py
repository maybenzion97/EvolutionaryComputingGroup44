import numpy as np
import pandas as pd

from stats import a12, compare, diversity_over_run


def final_table(
    metric: str, a_values: list[float], b_values: list[float]
) -> pd.DataFrame:
    """One row per condition and seed, the shape analysis.py passes to compare."""
    return pd.DataFrame(
        {
            "condition": ["a"] * len(a_values) + ["b"] * len(b_values),
            "seed": list(range(len(a_values))) + list(range(len(b_values))),
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
    assert result["a_better_in"] == 3


def test_higher_diversity_counts_as_better():
    table = final_table("diversity", [4.0, 5.0, 6.0], [1.0, 2.0, 3.0])
    result = compare(table, "H1", "diversity", "a", "b")
    assert result is not None
    assert not result["lower_is_better"]
    assert result["a12_a_better"] == 1.0
    assert result["a_better_in"] == 3


def test_runs_are_paired_by_seed():
    """The test uses each seed's own difference, not the two groups as a whole.

    The two conditions overlap heavily (9 to 11 against 10 to 12), so comparing
    them as unpaired samples shows little. Every seed is worse under a by exactly
    one, which is what pairing is meant to see.
    """
    table = pd.DataFrame(
        {
            "condition": ["a", "a", "a", "b", "b", "b"],
            "seed": [0, 1, 2, 0, 1, 2],
            "best_fitness": [10.0, 11.0, 12.0, 9.0, 10.0, 11.0],
        }
    )
    result = compare(table, "H3", "best_fitness", "a", "b")
    assert result is not None
    assert result["a_better_in"] == 0
    assert result["mean_diff"] == 1.0


def test_pairing_follows_the_seed_not_the_row_order():
    """Rows out of order must not change the result."""
    rows = pd.DataFrame(
        {
            "condition": ["a", "a", "a", "b", "b", "b"],
            "seed": [2, 0, 1, 1, 2, 0],
            "diversity": [6.0, 4.0, 5.0, 2.0, 3.0, 1.0],
        }
    )
    result = compare(rows, "H1", "diversity", "a", "b")
    assert result is not None
    assert result["mean_diff"] == 3.0
    assert result["a_better_in"] == 3


def test_seeds_without_a_partner_are_dropped():
    table = final_table("best_fitness", [1.0, 2.0, 3.0], [4.0, 5.0])
    result = compare(table, "H3", "best_fitness", "a", "b")
    assert result is not None
    assert result["n_pairs"] == 2


def test_diversity_over_run_skips_the_initial_population():
    """Generation 0 is shared by every condition, so it is left out."""
    data = pd.DataFrame(
        {
            "condition": ["a"] * 3,
            "seed": [0, 0, 0],
            "generation": [0, 1, 2],
            "diversity": [99.0, 4.0, 6.0],
        }
    )
    averaged = diversity_over_run(data)
    assert len(averaged) == 1
    assert averaged.iloc[0]["diversity_over_run"] == 5.0


def test_identical_runs_give_no_result():
    table = final_table("best_fitness", [1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    assert compare(table, "H3", "best_fitness", "a", "b") is None
