"""The worked example from our proposal: four bodies and their distances to the
7, 11, 15, 19 and 25-node targets."""

from ariel.ec import Individual

from selection import lexicase, tournament


def body(fitness: float, dists: list[float]) -> Individual:
    ind = Individual()
    ind.fitness = fitness
    ind.tags = {"dists": dists}
    return ind


def example_bodies() -> dict[str, Individual]:
    return {
        "A": body(12.41, [9, 10, 11, 12, 13]),
        "B": body(16.27, [5, 8, 12, 15, 18]),
        "C": body(11.97, [12, 12, 11, 10, 9]),
        "D": body(15.53, [5, 9, 12, 14, 17]),
    }


def test_lexicase_starting_with_the_smallest_target_picks_b():
    bodies = example_bodies()
    # T7 keeps B and D (both 5), then T11 keeps B (8 < 9)
    assert lexicase(list(bodies.values()), order=[0, 1, 2, 3, 4]) is bodies["B"]


def test_lexicase_starting_with_the_largest_target_picks_c():
    bodies = example_bodies()
    assert lexicase(list(bodies.values()), order=[4, 0, 1, 2, 3]) is bodies["C"]


def test_tournament_picks_the_lower_fitness():
    bodies = example_bodies()
    assert tournament([bodies["A"], bodies["C"]], k=2) is bodies["C"]
