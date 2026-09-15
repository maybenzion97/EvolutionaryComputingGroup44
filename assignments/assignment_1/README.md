# Assignment 1: lexicase vs tournament parent selection

**Research question.** When one robot body has to match five target bodies of
different sizes, does selecting parents one target at a time (lexicase) instead
of on the combined mean + std score (tournament) change what the EA finds, in
terms of population diversity and bodies that are close to single targets? And
does that help or hurt the combined fitness?

Only parent selection differs between the conditions. Everything else is the
same: tree encoding, at most 20 modules, population 100, 100 generations
(10,100 evaluations), subtree crossover (p = 0.7), one random tree mutation per
child, and generational replacement that keeps the best parent.

Run all commands from the **repository root**.

## Files

| File | What it does |
|---|---|
| `tree_edit_distance.py` | Distance between two bodies (from the course, unchanged) |
| `problem.py` | Targets, body budget, fitness, random bodies, and what the EA stores per individual |
| `metrics.py` | Population diversity |
| `selection.py` | Tournament and lexicase selection |
| `ea_tree.py` | The EA, built on `ariel.ec.EA` |
| `random_search.py` | Random search with the same number of evaluations |
| `calibrate_k.py` | Finds the tournament size that selects as strongly as lexicase |
| `records.py` | Writing and reading the result files |
| `run_all.sh` | The final experiments: 10 seeds × 4 conditions |
| `analysis.py` | Makes the figures and statistics for the report, using: |
| `conditions.py` | which condition is the baseline, the control, lexicase and random search |
| `plots.py` | the figures |
| `stats.py` | the hypothesis tests and the summary table |
| `tests/` | Tests for the code above |
| `lint.toml` | Ruff settings for this folder (see below) |
| `A1_template_2026.py` | Course template (reference only) |

## Workflow

```bash
# 0. Tests (a few seconds)
uv run pytest assignments/assignment_1/tests

# 1. Choose k for the control condition (~1 minute)
uv run python assignments/assignment_1/calibrate_k.py

# 2. Final experiments (~10 minutes), with the k from step 1
K_CONTROL=40 bash assignments/assignment_1/run_all.sh

# 3. Figures and statistics
uv run python assignments/assignment_1/analysis.py
```

For a quick test run, give it its own output folder so it doesn't end up among
the final results (the scripts refuse to overwrite an existing run unless you
pass `--overwrite`):

```bash
uv run python assignments/assignment_1/ea_tree.py --selection lexicase --seed 0 \
    --pop-size 20 --generations 10 --out-dir assignments/assignment_1/results/_debug/lexicase
```

To run the final experiments again over the existing results:
`OVERWRITE=1 K_CONTROL=40 bash assignments/assignment_1/run_all.sh`

`analysis.py` uses the tournament with k = 2 as the baseline and the other
tournament size as the control. If `results/` contains more than one other
tournament size, choose the control with `--control tournament_k40`.

## Linting and formatting

The repository's own ruff settings skip `assignments/`, so this folder has its
own settings in `lint.toml`. The file is deliberately not called `ruff.toml`,
because the pre-commit hooks would then pick it up and rewrite the folder.

```bash
uv run ruff check --no-fix --config assignments/assignment_1/lint.toml assignments/assignment_1
uv run ruff format --config assignments/assignment_1/lint.toml assignments/assignment_1
```

The course files (`tree_edit_distance.py`, `A1_template_2026.py`) are excluded.
Editors don't pick up `lint.toml` automatically.

## Output

```
results/
  tournament_k2/seed_00/   config.json  history.csv  best_body.json  specialists.json  database.db
  lexicase/seed_00/        ...
  tournament_k40/seed_00/  ...
  random/seed_00/          config.json  history.csv  best_body.json
  _calibration/            calibration.csv  calibration.json
figures/
  fitness_curve  diversity_curve  selection_strength  closest_per_target  final_fitness  (.pdf and .png)
  summary.csv  stats.csv
```

`database.db` is ariel's record of every individual (~12 MB per run). It is not
committed; `run_all.sh` recreates it.

### Columns of `history.csv`

| Column | Meaning |
|---|---|
| `generation`, `evaluations` | Generation 0 is the initial population; evaluations = (generation + 1) × population size |
| `best_fitness`, `mean_fitness`, `std_fitness` | Fitness = mean + std of the edit distances to the 5 targets (lower is better). For random search `best_fitness` is the best found so far |
| `best_size`, `mean_size` | Number of nodes in a body, including the core |
| `diversity` | Mean edit distance between 20 randomly chosen bodies |
| `distinct_parents` | Different bodies among the first `pop_size` parent picks (fewer = stronger selection) |
| `parent_picks` | All parent picks in the generation; more than `pop_size` when children were too big and discarded |
| `closest_7` to `closest_25` | Smallest distance of any body in the population to that target |
| `best_dist_7` to `best_dist_25` | Distance of the best body to each target |

### Columns of `stats.csv`

One row per test: `hypothesis`, `metric`, the two conditions `a` and `b`, their
means and standard deviations, the Mann-Whitney `U` and `p`, `p_holm` (corrected
within each hypothesis), `lower_is_better` for the metric, and `a12_a_better`:
the chance that a run of `a` beats a run of `b` (0.5 means no difference).

## Design choices and limitations

Source material for the Methods and Discussion sections of the report.

- **Tournament without replacement.** A tournament of size k compares k
  different bodies. Sizes below 1 or above the population size are rejected.
- **Calibrating the control.** `calibrate_k.py` lets lexicase and tournaments of
  several sizes pick parents from the same lexicase populations: lexicase gave
  8.6 distinct parents per 100 picks and k = 40 gave 8.7. The final runs confirm
  the match on each condition's own populations: 8.83 for lexicase and 8.77 for
  k = 40 (`summary.csv`).
- **Generational survivors with one elite.** In a pilot, (μ + λ) survivor
  selection on the combined score hid most of the difference between the
  selection schemes, so parent selection is kept as the main source of
  selection pressure.
- **Initial bodies get 1 to 20 modules.** `random_tree(20)` always builds full
  21-node bodies, which would make random search an unfairly weak baseline.
- **The 25-node target cannot be matched exactly.** Bodies have at most 21
  nodes, so their distance to it is at least 4. No run came close to that
  limit: the closest any body got was 6.0 (lexicase), 10.5 (k = 2) and 11.5
  (k = 40).
- **Oversized children are discarded, but this almost never happens.** Only 0
  to 2 of the 1,000 generations per condition discarded any child, and the
  largest best body had 17 nodes, so the size limit adds no real pressure
  towards small bodies.
- **Diversity is estimated** from 20 randomly chosen bodies (190 pairs) to keep
  runs fast.
- **Lexicase with five targets is very selective**: it usually ends at the best
  body for one target, so only about 9 different bodies reproduce per
  generation.

## Reproducibility

- Every run seeds `random` and `numpy` with `--seed`; the same seed gives the same `history.csv` (tested).
- Final experiments use seeds 0 to 9; calibration uses seeds 100 to 102.
- Diversity is measured with its own random generator, so it does not affect the runs.
- Initial bodies (and random-search bodies) get 1 to 20 modules, chosen uniformly (`--init-size uniform`).
- The ariel framework (`src/ariel`) is not modified.
