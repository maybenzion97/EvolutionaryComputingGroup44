"""Parent selection: tournament and lexicase.

Both functions take the current parents and return one of them. Lexicase uses
the distances to the individual targets, which the EA stores in each
individual's tags.
"""

import random
from collections.abc import Callable, Sequence

from ariel.ec import Individual

Selector = Callable[[list[Individual]], Individual]


def tournament(parents: list[Individual], k: int = 2) -> Individual:
    """Pick k different parents at random and return the one with the lowest fitness."""
    return min(random.sample(parents, k), key=lambda ind: ind.fitness)


def lexicase(
    parents: list[Individual], order: Sequence[int] | None = None
) -> Individual:
    """Go through the targets in a random order, each time keeping only the
    parents that are closest to that target.

    `order` fixes the order of the targets; the tests use it.
    """
    n_targets = len(parents[0].tags["dists"])
    if order is None:
        order = random.sample(range(n_targets), n_targets)

    candidates = parents
    for target in order:
        best = min(ind.tags["dists"][target] for ind in candidates)
        candidates = [ind for ind in candidates if ind.tags["dists"][target] == best]
        if len(candidates) == 1:
            break
    return random.choice(candidates)


def make_selector(name: str, k: int = 2) -> Selector:
    if name == "tournament":
        return lambda parents: tournament(parents, k)
    if name == "lexicase":
        return lexicase
    raise ValueError(f"unknown selection: {name}")
