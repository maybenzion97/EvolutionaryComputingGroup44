"""The optimisation problem: target bodies, body budget, fitness and diversity."""

import random
from pathlib import Path
from typing import Literal, NamedTuple

import networkx as nx
import numpy as np

from ariel.body_phenotypes.robogen_lite.decoders._blueprint import load_graph_from_json
from ariel.ec.genotypes.tree.operators import random_tree
from ariel.ec.genotypes.tree.tree_genome import TreeGenome

from tree_edit_distance import tree_edit_distance

TARGET_DIR = Path(__file__).parent / "target_bodies"
TARGETS: list[nx.DiGraph] = [
    load_graph_from_json(path) for path in sorted(TARGET_DIR.glob("*.json"))
]
TARGET_SIZES = [target.number_of_nodes() for target in TARGETS]  # 7, 11, 15, 19, 25

MAX_MODULES = 20  # not counting the core
MAX_NODES = MAX_MODULES + 1

DIVERSITY_SAMPLE = 20  # bodies used to estimate population diversity

InitSize = Literal["uniform", "full"]


class Evaluation(NamedTuple):
    fitness: float
    dists: list[float]  # distance to each target, in the order of TARGETS
    size: int


def seed_everything(seed: int) -> None:
    # the tree operators use `random`
    random.seed(seed)
    np.random.seed(seed)


def random_genome(init_size: InitSize) -> TreeGenome:
    """Random body with 20 modules ("full") or 1-20 modules ("uniform").

    random_tree stops early when no free faces are left, so a body can end up
    smaller than requested.
    """
    if init_size == "full":
        n_modules = MAX_MODULES
    elif init_size == "uniform":
        n_modules = random.randint(1, MAX_MODULES)
    else:
        raise ValueError(f"unknown init_size: {init_size}")
    return random_tree(max_modules=n_modules)


def evaluate_genome(genome: TreeGenome) -> Evaluation:
    """Fitness = mean + std of the edit distances to the targets (lower is better).

    Same value as mean_plus_std_tree_edit_distance, but we also keep the
    per-target distances because lexicase selection needs them.
    """
    body = genome.to_networkx()
    dists = [tree_edit_distance(body, target) for target in TARGETS]
    fitness = float(np.mean(dists) + np.std(dists))
    return Evaluation(fitness, dists, len(genome.nodes))


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


def sample_for_diversity(items: list, rng: random.Random) -> list:
    return rng.sample(items, min(DIVERSITY_SAMPLE, len(items)))
