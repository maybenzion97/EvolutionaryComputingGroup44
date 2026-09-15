"""Find the tournament size that selects as strongly as lexicase.

Selection strength is the number of different parents among pop_size picks
(fewer means stronger); ea_tree.py logs the same number as distinct_parents.
We run lexicase EAs and, in every generation, let lexicase and tournaments of
several sizes pick parents from the same population. The random state is
restored afterwards, so the measurement does not change the runs.

Usage (from the repository root):
    uv run python assignments/assignment_1/calibrate_k.py
"""

import argparse
import csv
import random

import numpy as np

from ariel.ec import EAOperation, Population

from ea_tree import RunConfig, run_experiment
from records import RESULTS_DIR, save_json
from selection import Selector, lexicase, make_selector

# Lexicase turned out to be very selective, so we also try large tournaments.
CANDIDATE_K = [2, 5, 10, 15, 20, 25, 30, 40, 50]


def measure_selection(
    population: Population,
    selectors: dict[str, Selector],
    counts: dict[str, list[int]],
) -> Population:
    parents = population.alive.to_list()
    state = random.getstate()
    for name, select in selectors.items():
        picked = {id(select(parents)) for _ in range(len(parents))}
        counts[name].append(len(picked))
    random.setstate(state)
    return population


def main() -> None:
    parser = argparse.ArgumentParser(description="Choose k for the control condition.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[100, 101, 102])
    parser.add_argument("--generations", type=int, default=50)
    args = parser.parse_args()

    selectors = {"lexicase": lexicase}
    for k in CANDIDATE_K:
        selectors[f"tournament_k{k}"] = make_selector("tournament", k)
    counts = {name: [] for name in selectors}

    out_root = RESULTS_DIR / "_calibration"
    measure = EAOperation(measure_selection, selectors=selectors, counts=counts)
    for seed in args.seeds:
        cfg = RunConfig(
            selection="lexicase",
            seed=seed,
            generations=args.generations,
            out_dir=out_root / f"seed_{seed}",
            overwrite=True,  
        )
        run_experiment(cfg, extra_steps=[measure])

    means = {name: float(np.mean(values)) for name, values in counts.items()}
    best_k = min(
        CANDIDATE_K, key=lambda k: abs(means[f"tournament_k{k}"] - means["lexicase"])
    )

    print("\nmean number of different parents per generation:")
    for name, value in means.items():
        note = "  <- closest to lexicase" if name == f"tournament_k{best_k}" else ""
        print(f"  {name:15s} {value:6.1f}{note}")
    print(f"\ncontrol condition: tournament with k = {best_k}")
    print(f"next: K_CONTROL={best_k} bash assignments/assignment_1/run_all.sh")

    with (out_root / "calibration.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["selector", "mean_distinct_parents", "std_distinct_parents"])
        for name, values in counts.items():
            writer.writerow([name, np.mean(values), np.std(values)])
    summary = {
        "seeds": args.seeds,
        "generations": args.generations,
        "mean_distinct_parents": means,
        "chosen_k": best_k,
    }
    save_json(out_root / "calibration.json", summary)


if __name__ == "__main__":
    main()
