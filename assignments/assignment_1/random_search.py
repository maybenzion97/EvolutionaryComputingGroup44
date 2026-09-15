"""Random search baseline with the same number of evaluations as the EA.

Bodies are sampled in batches of pop_size, one batch per "generation", so the
CSV lines up with the EA's. best_fitness is the best body found so far; the
other columns describe the current batch.

Usage (from the repository root):
    uv run python assignments/assignment_1/random_search.py --seed 0
"""

import argparse
import time
from pathlib import Path

from problem import (
    TARGET_SIZES,
    InitSize,
    diversity,
    diversity_rng,
    evaluate_genome,
    random_genome,
    sample_for_diversity,
    seed_everything,
)
from records import RESULTS_DIR, HistoryWriter, ensure_fresh, generation_row, save_json


def run_random_search(
    seed: int,
    pop_size: int = 100,
    generations: int = 100,
    init_size: InitSize = "uniform",
    out_dir: Path | None = None,
    overwrite: bool = False,
) -> Path:
    start = time.perf_counter()
    if out_dir is None:
        out_dir = RESULTS_DIR / "random" / f"seed_{seed:02d}"
    ensure_fresh(out_dir, overwrite)
    config = {
        "condition": "random",
        "selection": "random",
        "seed": seed,
        "pop_size": pop_size,
        "generations": generations,
        "init_size": init_size,
    }
    save_json(out_dir / "config.json", config)

    seed_everything(seed)
    rng = diversity_rng(seed)
    history = HistoryWriter(out_dir / "history.csv")
    best = None
    best_genotype = None

    for generation in range(generations + 1):
        genomes = [random_genome(init_size) for _ in range(pop_size)]
        batch = [evaluate_genome(genome) for genome in genomes]
        for genome, result in zip(genomes, batch):
            if best is None or result.fitness < best.fitness:
                best = result
                best_genotype = genome.to_dict()

        row = generation_row(
            generation=generation,
            evaluations=(generation + 1) * pop_size,
            population=batch,
            diversity_value=diversity(sample_for_diversity(genomes, rng)),
            best=best,
        )
        history.write(row)
    history.close()

    best_body = {
        "fitness": best.fitness,
        "size": best.size,
        "dists": dict(zip(TARGET_SIZES, best.dists)),
        "genotype": best_genotype,
    }
    save_json(out_dir / "best_body.json", best_body)
    seconds = time.perf_counter() - start
    print(
        f"random seed {seed}: best fitness {best.fitness:.3f} ({seconds:.0f}s) -> {out_dir}"
    )
    return out_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Random search baseline.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pop-size", type=int, default=100)
    parser.add_argument("--generations", type=int, default=100)
    parser.add_argument("--init-size", choices=["uniform", "full"], default="uniform")
    parser.add_argument(
        "--out-dir", type=Path, help="output folder, e.g. for test runs"
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="replace an existing run"
    )
    args = parser.parse_args()
    run_random_search(
        args.seed,
        args.pop_size,
        args.generations,
        args.init_size,
        args.out_dir,
        args.overwrite,
    )
