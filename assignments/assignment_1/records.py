"""Writing and reading experiment results."""

import csv
import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from problem import TARGET_SIZES, Evaluation

HERE = Path(__file__).parent
RESULTS_DIR = HERE / "results"
FIGURES_DIR = HERE / "figures"

HISTORY_COLUMNS = [
    "generation",
    "evaluations",
    "population_size",
    "best_fitness",
    "mean_fitness",
    "std_fitness",
    "best_size",
    "mean_size",
    "diversity",
    "distinct_parents",  # distinct individuals among the first pop_size picks
    "parent_picks",  # all parent picks, including those for discarded children
    *[f"closest_{size}" for size in TARGET_SIZES],
    *[f"best_dist_{size}" for size in TARGET_SIZES],
]


def generation_row(
    generation: int,
    evaluations: int,
    population: list[Evaluation],
    diversity_value: float,
    distinct_parents: int | None = None,
    parent_picks: int | None = None,
    best: Evaluation | None = None,
) -> dict[str, Any]:
    """One CSV row for a population.

    `best` replaces the population's best body (random search reports the best
    body found so far instead).
    """
    fitness = np.array([e.fitness for e in population])
    dists = np.array([e.dists for e in population])
    sizes = [e.size for e in population]
    if best is None:
        best = population[int(np.argmin(fitness))]

    row: dict[str, Any] = {
        "generation": generation,
        "evaluations": evaluations,
        "population_size": len(population),
        "best_fitness": best.fitness,
        "mean_fitness": fitness.mean(),
        "std_fitness": fitness.std(),
        "best_size": best.size,
        "mean_size": float(np.mean(sizes)),
        "diversity": diversity_value,
        "distinct_parents": distinct_parents,
        "parent_picks": parent_picks,
    }
    for j, size in enumerate(TARGET_SIZES):
        row[f"closest_{size}"] = dists[:, j].min()
        row[f"best_dist_{size}"] = best.dists[j]
    return row


class HistoryWriter:
    """Writes history.csv, one row per generation."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.file = path.open("w", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.file, fieldnames=HISTORY_COLUMNS)
        self.writer.writeheader()

    def write(self, row: dict[str, Any]) -> None:
        self.writer.writerow(row)
        self.file.flush()

    def close(self) -> None:
        self.file.close()


def body_summary(result: Evaluation, genotype: Any) -> dict[str, Any]:
    """One body as saved in best_body.json and specialists.json."""
    return {
        "fitness": result.fitness,
        "size": result.size,
        "dists": dict(zip(TARGET_SIZES, result.dists, strict=True)),
        "genotype": genotype,
    }


def ensure_fresh(out: Path, overwrite: bool) -> None:
    if (out / "history.csv").exists() and not overwrite:
        raise SystemExit(f"{out} already has results (use --out-dir or --overwrite)")


def save_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


def load_histories(results: Path) -> pd.DataFrame:
    """All history.csv files under results/<condition>/seed_XX/ in one table.

    Adds the condition, selection scheme and tournament size from each run's
    config.json. Folders starting with "_" (e.g. _calibration) are skipped.
    """
    frames = []
    for path in sorted(results.glob("*/seed_*/history.csv")):
        if path.parent.parent.name.startswith("_"):
            continue
        config = json.loads((path.parent / "config.json").read_text(encoding="utf-8"))
        frame = pd.read_csv(path)
        frame["condition"] = config["condition"]
        frame["selection"] = config.get("selection", config["condition"])
        frame["k"] = config.get("k")
        frame["seed"] = path.parent.name
        frames.append(frame)
    if not frames:
        raise SystemExit(f"no results found in {results}")
    return pd.concat(frames, ignore_index=True)


def check_consistent(data: pd.DataFrame) -> None:
    """Stop if the runs differ in length or population size.

    This catches short test runs that were accidentally left in results/.
    """
    runs = data.groupby(["condition", "seed"]).agg(
        generations=("generation", "max"),
        pop_size=("population_size", "first"),
    )
    shapes = runs.value_counts()
    if len(shapes) > 1:
        generations, pop_size = cast("tuple[int, int]", shapes.idxmax())
        odd = runs[
            (runs["generations"] != generations) | (runs["pop_size"] != pop_size)
        ]
        raise SystemExit(
            f"most runs have {generations} generations and population {pop_size}, "
            f"but these do not:\n{odd.to_string()}"
        )
