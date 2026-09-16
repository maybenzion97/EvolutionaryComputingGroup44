"""Measurements of a population that are not part of the fitness."""

import random

import numpy as np
from ariel.ec.genotypes.tree.tree_genome import TreeGenome

from tree_edit_distance import tree_edit_distance

DIVERSITY_SAMPLE = 20  # bodies used to estimate population diversity


def diversity(genomes: list[TreeGenome]) -> float:
    """Mean edit distance between all pairs of the given bodies."""
    graphs = [genome.to_networkx() for genome in genomes]
    distances = []
    for i, a in enumerate(graphs):
        for b in graphs[i + 1 :]:
            distances.append(tree_edit_distance(a, b))
    return float(np.mean(distances))


def diversity_rng(seed: int) -> random.Random:
    # a separate generator, so that measuring diversity does not change the run
    return random.Random(10_000 + seed)


def sample_for_diversity[T](items: list[T], rng: random.Random) -> list[T]:
    return rng.sample(items, min(DIVERSITY_SAMPLE, len(items)))
