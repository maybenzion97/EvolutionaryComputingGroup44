import random

import pytest

from problem import TARGETS, evaluate_genome, random_genome
from tree_edit_distance import mean_plus_std_tree_edit_distance


@pytest.mark.parametrize("init_size", ["uniform", "full"])
def test_fitness_matches_the_course_function(init_size):
    random.seed(0)
    for _ in range(15):
        genome = random_genome(init_size)
        expected = mean_plus_std_tree_edit_distance(genome.to_networkx(), TARGETS)
        assert evaluate_genome(genome).fitness == pytest.approx(expected, abs=1e-9)


def test_unknown_init_size_is_rejected():
    with pytest.raises(ValueError):
        random_genome("medium")
