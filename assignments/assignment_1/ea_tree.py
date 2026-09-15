"""Evolve tree-encoded bodies with tournament or lexicase parent selection.

Each generation: select parents -> subtree crossover (p = 0.7) -> one random
tree mutation per child -> drop children with more than 21 nodes -> evaluate
-> the children replace the parents, except for the best parent (elitism).

Usage (from the repository root):
    uv run python assignments/assignment_1/ea_tree.py --selection lexicase --seed 0
    uv run python assignments/assignment_1/ea_tree.py --selection tournament --k 2 --seed 0

Results are written to results/<condition>/seed_XX/.
"""


import argparse
import copy
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from ariel.ec import EA, EAOperation, Individual, Population
from ariel.ec.genotypes.tree.operators import (
    crossover_subtree,
    mutate_hoist,
    mutate_replace_node,
    mutate_shrink,
    mutate_subtree_replacement,
)
from ariel.ec.genotypes.tree.tree_genome import TreeGenome

from problem import (
    MAX_NODES,
    TARGET_SIZES,
    Evaluation,
    InitSize,
    diversity,
    diversity_rng,
    evaluate_genome,
    random_genome,
    sample_for_diversity,
    seed_everything,
)
from records import RESULTS_DIR, HistoryWriter, ensure_fresh, generation_row, save_json
from selection import Selector, make_selector


def mutate_subtree(genome: TreeGenome) -> None:
    mutate_subtree_replacement(genome, max_modules=3)


MUTATIONS = [mutate_replace_node, mutate_shrink, mutate_hoist, mutate_subtree]


@dataclass
class RunConfig:
    selection: str = "tournament"  # or "lexicase"
    k: int = 2  # tournament size, not used by lexicase
    seed: int = 0
    pop_size: int = 100
    generations: int = 100
    p_crossover: float = 0.7
    init_size: InitSize = "uniform"
    out_dir: Path | None = None
    overwrite: bool = False

    @property
    def condition(self) -> str:
        if self.selection == "lexicase":
            return "lexicase"
        return f"tournament_k{self.k}"

    def output_dir(self) -> Path:
        if self.out_dir is not None:
            return self.out_dir
        return RESULTS_DIR / self.condition / f"seed_{self.seed:02d}"


@dataclass
class RunState:
    config: RunConfig
    select: Selector
    history: HistoryWriter
    diversity_rng: random.Random
    generation: int = 0
    distinct_parents: int | None = None
    parent_picks: int | None = None


def make_individual(genome: TreeGenome) -> Individual:
    ind = Individual()
    ind.genotype = genome.to_dict()
    return ind


def copy_genome(ind: Individual) -> TreeGenome:
    # from_dict reuses the stored edge list, so copy it before mutating
    return TreeGenome.from_dict(copy.deepcopy(ind.genotype))


def evaluation_of(ind: Individual) -> Evaluation:
    return Evaluation(ind.fitness, ind.tags["dists"], ind.tags["size"])


def record(individuals: list[Individual], run: RunState) -> None:
    sample = sample_for_diversity(individuals, run.diversity_rng)
    row = generation_row(
        generation=run.generation,
        evaluations=(run.generation + 1) * run.config.pop_size,
        population=[evaluation_of(ind) for ind in individuals],
        diversity_value=diversity(
            [TreeGenome.from_dict(ind.genotype) for ind in sample]
        ),
        distinct_parents=run.distinct_parents,
        parent_picks=run.parent_picks,
    )
    run.history.write(row)


# The EA steps below run in this order every generation.


def reproduce(population: Population, run: RunState) -> Population:
    pop_size = run.config.pop_size
    parents = population.alive.to_list()
    children = []
    picks = []

    while len(children) < pop_size:
        parent_a = run.select(parents)
        parent_b = run.select(parents)
        picks += [id(parent_a), id(parent_b)]
        a, b = copy_genome(parent_a), copy_genome(parent_b)
        if random.random() < run.config.p_crossover:
            a, b = crossover_subtree(a, b)
        for genome in (a, b):
            random.choice(MUTATIONS)(genome)
            if len(genome.nodes) <= MAX_NODES:
                children.append(make_individual(genome))

    population.extend(children[:pop_size])
    run.generation += 1
    # Selection strength: different parents among the first pop_size picks.
    # Every generation makes at least pop_size picks.
    run.distinct_parents = len(set(picks[:pop_size]))
    run.parent_picks = len(picks)
    return population


def evaluate(population: Population, run: RunState) -> Population:
    for ind in population.unevaluated:
        result = evaluate_genome(TreeGenome.from_dict(ind.genotype))
        ind.fitness = result.fitness
        ind.tags = {"dists": result.dists, "size": result.size}
    return population


def survivor_selection(population: Population, run: RunState) -> Population:
    """The children replace the parents; the best parent takes the place of the worst child."""
    alive = population.alive.to_list()
    # Children made in this generation have not been saved yet, so the EA engine
    # has not set their time_of_birth.
    parents = [ind for ind in alive if ind.time_of_birth != -1]
    children = [ind for ind in alive if ind.time_of_birth == -1]
    n = run.config.pop_size
    if len(parents) != n or len(children) != n:
        raise RuntimeError(
            f"expected {n} parents and {n} children, got {len(parents)} and {len(children)}"
        )

    elite = min(parents, key=lambda ind: ind.fitness)
    for parent in parents:
        if parent is not elite:
            parent.alive = False
    worst_child = max(children, key=lambda ind: ind.fitness)
    worst_child.alive = False
    return population


def log_generation(population: Population, run: RunState) -> Population:
    record(population.alive.to_list(), run)
    return population


def describe(ind: Individual) -> dict:
    return {
        "fitness": ind.fitness,
        "size": ind.tags["size"],
        "dists": dict(zip(TARGET_SIZES, ind.tags["dists"])),
        "genotype": ind.genotype,
    }


def save_final(individuals: list[Individual], out: Path) -> None:
    """Save the best body and, for each target, the body closest to it."""
    best = min(individuals, key=lambda ind: ind.fitness)
    save_json(out / "best_body.json", describe(best))

    specialists = {}
    for j, size in enumerate(TARGET_SIZES):
        closest = min(individuals, key=lambda ind: ind.tags["dists"][j])
        specialists[f"target_{size}"] = describe(closest)
    save_json(out / "specialists.json", specialists)


def run_experiment(
    cfg: RunConfig, extra_steps: list[EAOperation] | None = None
) -> Path:
    """Run one EA and return its output folder.

    `extra_steps` run after survivor selection; calibrate_k.py uses this.
    """
    start = time.perf_counter()
    out = cfg.output_dir()
    ensure_fresh(out, cfg.overwrite)
    out.mkdir(parents=True, exist_ok=True)
    save_json(out / "config.json", {**asdict(cfg), "condition": cfg.condition})

    seed_everything(cfg.seed)
    run = RunState(
        config=cfg,
        select=make_selector(cfg.selection, cfg.k),
        history=HistoryWriter(out / "history.csv"),
        diversity_rng=diversity_rng(cfg.seed),
    )

    initial = Population(
        [make_individual(random_genome(cfg.init_size)) for _ in range(cfg.pop_size)]
    )
    evaluate(initial, run)
    record(initial.to_list(), run)  # generation 0

    steps = [
        EAOperation(reproduce, run=run),
        EAOperation(evaluate, run=run),
        EAOperation(survivor_selection, run=run),
        *(extra_steps or []),
        EAOperation(log_generation, run=run),
    ]
    ea = EA(
        initial,
        steps,
        num_steps=cfg.generations,
        is_maximisation=False,
        quiet=True,
        db_file_path=out / "database.db",
        db_handling="delete",
    )
    ea.run()
    run.history.close()

    ea.fetch_population()  # the last generation, from the database
    final = ea.population.to_list()
    save_final(final, out)

    best = min(ind.fitness for ind in final)
    seconds = time.perf_counter() - start
    print(
        f"{cfg.condition} seed {cfg.seed}: best fitness {best:.3f} ({seconds:.0f}s) -> {out}"
    )
    return out


def parse_args() -> RunConfig:
    parser = argparse.ArgumentParser(
        description="Evolve bodies with tournament or lexicase selection."
    )
    parser.add_argument(
        "--selection", choices=["tournament", "lexicase"], default="tournament"
    )
    parser.add_argument("--k", type=int, default=2, help="tournament size")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pop-size", type=int, default=100)
    parser.add_argument("--generations", type=int, default=100)
    parser.add_argument("--p-crossover", type=float, default=0.7)
    parser.add_argument("--init-size", choices=["uniform", "full"], default="uniform")
    parser.add_argument(
        "--out-dir", type=Path, help="output folder, e.g. for test runs"
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="replace an existing run"
    )
    args = parser.parse_args()
    return RunConfig(
        selection=args.selection,
        k=args.k,
        seed=args.seed,
        pop_size=args.pop_size,
        generations=args.generations,
        p_crossover=args.p_crossover,
        init_size=args.init_size,
        out_dir=args.out_dir,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    run_experiment(parse_args())
