"""The optimisation problem: target bodies, body budget and fitness."""

import random
from pathlib import Path
from typing import Literal, NamedTuple, cast

import numpy as np
from ariel.body_phenotypes.robogen_lite.decoders._blueprint import load_graph_from_json
from ariel.ec import Individual, JSONType
from ariel.ec.genotypes.tree.operators import random_tree
from ariel.ec.genotypes.tree.tree_genome import TreeGenome

from tree_edit_distance import tree_edit_distance

TARGET_DIR = Path(__file__).parent / "target_bodies"
TARGETS = [load_graph_from_json(path) for path in sorted(TARGET_DIR.glob("*.json"))]
TARGET_SIZES = [target.number_of_nodes() for target in TARGETS]  # 7, 11, 15, 19, 25

MAX_MODULES = 20  # not counting the core
MAX_NODES = MAX_MODULES + 1

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


def evaluation_tags(result: Evaluation) -> dict[JSONType, JSONType]:
    """What the EA stores in an individual's tags after evaluating it."""
    return {"dists": result.dists, "size": result.size}


def distances_of(ind: Individual) -> list[float]:
    # ariel types tags as generic JSON; evaluation_tags() decides what is in them
    return cast("list[float]", ind.tags["dists"])


def evaluation_of(ind: Individual) -> Evaluation:
    return Evaluation(ind.fitness, distances_of(ind), cast("int", ind.tags["size"]))
