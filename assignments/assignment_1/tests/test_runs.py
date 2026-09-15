"""Short EA and random-search runs (population 20, 5 generations)."""

import csv
import shutil
from pathlib import Path

import pytest

from ea_tree import RunConfig, run_experiment
from random_search import run_random_search
from records import check_consistent, load_histories

POP = 20
GENERATIONS = 5


def read_history(folder: Path) -> list[dict[str, str]]:
    with (folder / "history.csv").open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


@pytest.fixture(scope="module")
def results(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("results")
    for selection in ["tournament", "lexicase"]:
        cfg = RunConfig(
            selection=selection, seed=0, pop_size=POP, generations=GENERATIONS
        )
        cfg.out_dir = root / cfg.condition / "seed_00"
        run_experiment(cfg)
    run_random_search(
        seed=0,
        pop_size=POP,
        generations=GENERATIONS,
        out_dir=root / "random" / "seed_00",
    )
    return root


@pytest.mark.parametrize("condition", ["tournament_k2", "lexicase"])
def test_ea_run(results: Path, condition: str):
    folder = results / condition / "seed_00"
    rows = read_history(folder)
    best = [float(row["best_fitness"]) for row in rows]

    assert len(rows) == GENERATIONS + 1  # generation 0 is the initial population
    assert all(int(row["population_size"]) == POP for row in rows)
    assert best == sorted(best, reverse=True)  # elitism: the best never gets worse
    assert int(rows[-1]["evaluations"]) == (GENERATIONS + 1) * POP
    for name in ["database.db", "best_body.json", "specialists.json"]:
        assert (folder / name).exists()
    for row in rows[1:]:
        assert 0 < int(row["distinct_parents"]) <= POP <= int(row["parent_picks"])


def test_random_search_has_the_same_budget(results: Path):
    rows = read_history(results / "random" / "seed_00")
    best = [float(row["best_fitness"]) for row in rows]
    assert int(rows[-1]["evaluations"]) == (GENERATIONS + 1) * POP
    assert best == sorted(best, reverse=True)


def test_same_seed_gives_the_same_history(tmp_path: Path):
    folders = []
    for i in range(2):
        cfg = RunConfig(
            selection="lexicase",
            seed=3,
            pop_size=POP,
            generations=3,
            out_dir=tmp_path / str(i),
        )
        folders.append(run_experiment(cfg))
    assert (folders[0] / "history.csv").read_text() == (
        folders[1] / "history.csv"
    ).read_text()


def test_existing_results_are_not_overwritten(results: Path):
    cfg = RunConfig(selection="lexicase", seed=0, pop_size=POP, generations=GENERATIONS)
    cfg.out_dir = results / "lexicase" / "seed_00"
    with pytest.raises(SystemExit):
        run_experiment(cfg)
    with pytest.raises(SystemExit):
        run_random_search(
            seed=0,
            pop_size=POP,
            generations=GENERATIONS,
            out_dir=results / "random" / "seed_00",
        )


def test_overwrite_replaces_a_run(tmp_path: Path):
    cfg = RunConfig(
        selection="lexicase",
        seed=3,
        pop_size=POP,
        generations=2,
        out_dir=tmp_path / "run",
    )
    run_experiment(cfg)
    cfg.overwrite = True
    run_experiment(cfg)  # would stop with SystemExit without overwrite
    assert (tmp_path / "run" / "history.csv").exists()


def test_analysis_rejects_a_shorter_run(results: Path, tmp_path: Path):
    check_consistent(load_histories(results))  # all runs have the same size

    mixed = tmp_path / "results"
    shutil.copytree(results, mixed)
    short = RunConfig(
        selection="lexicase",
        seed=1,
        pop_size=POP,
        generations=2,
        out_dir=mixed / "lexicase" / "seed_01",
    )
    run_experiment(short)
    with pytest.raises(SystemExit):
        check_consistent(load_histories(mixed))
