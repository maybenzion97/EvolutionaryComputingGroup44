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
| `A1_template_2026.py`, `Assignment1.html` | Course template and assignment text (reference only) |
| `problem.py` | Targets, body budget, fitness, random bodies, and what the EA stores per individual |
| `metrics.py` | Population diversity |
| `selection.py` | Tournament and lexicase selection |
| `ea_tree.py` | The EA, built on `ariel.ec.EA` |
| `random_search.py` | Random search with the same number of evaluations |
| `calibrate_k.py` | Finds the tournament size that selects as strongly as lexicase |
| `records.py` | Writing and reading the result files |
| `run_all.sh` | The final experiments: 10 seeds × 4 conditions |
| `analysis.py` | Makes the figures and statistics, using the three files below |
| `conditions.py` | Which condition is the baseline, the control, lexicase and random search |
| `plots.py` | The five figures |
| `stats.py` | The hypothesis tests and the summary table |
| `tests/` | 18 tests for the code above |
| `lint.toml` | Ruff settings for this folder (see below) |

## Reproducing the results

### Only the analysis, from what is in git (a few seconds)

The per-run CSVs are committed, so the figures and statistics can be rebuilt
without running any evolution:

```bash
uv run python assignments/assignment_1/analysis.py
```

This reads `results/*/seed_*/` and overwrites everything in `figures/`. Use it
to check a number, or after changing a plot or a test.

### Everything, from scratch (about 8.5 minutes)

```bash
# 0. Tests                                                    (4 seconds)
uv run pytest assignments/assignment_1/tests

# 1. Choose k for the control condition                       (18 seconds)
uv run python assignments/assignment_1/calibrate_k.py

# 2. Final experiments: 40 runs, with the k from step 1       (8 minutes)
K_CONTROL=40 bash assignments/assignment_1/run_all.sh

# 3. Figures and statistics                                   (1 second)
uv run python assignments/assignment_1/analysis.py
```

The scripts do not overwrite existing runs by default, so delete `results/` and
`figures/` first, or add `OVERWRITE=1` to step 2:

```bash
OVERWRITE=1 K_CONTROL=40 bash assignments/assignment_1/run_all.sh
```

Every run is seeded, so this rebuild is exact: after deleting both folders and
running the four commands, all 43 `history.csv` files, `stats.csv`,
`summary.csv` and the five figures came out byte-identical to the committed
ones, and the calibration chose k = 40 again.

### One run on its own

Give test runs their own output folder so they never mix with the final
results:

```bash
uv run python assignments/assignment_1/ea_tree.py --selection lexicase --seed 0 \
    --pop-size 20 --generations 10 --out-dir assignments/assignment_1/results/_debug/lexicase
```

| Script | Options |
|---|---|
| `ea_tree.py` | `--selection tournament\|lexicase` (default tournament), `--k 2`, `--seed 0`, `--pop-size 100`, `--generations 100`, `--p-crossover 0.7`, `--init-size uniform\|full`, `--out-dir`, `--overwrite` |
| `random_search.py` | `--seed 0`, `--pop-size 100`, `--generations 100`, `--init-size uniform\|full`, `--out-dir`, `--overwrite` |
| `calibrate_k.py` | `--seeds 100 101 102`, `--generations 50` |
| `analysis.py` | `--results`, `--figures`, `--control` (e.g. `--control tournament_k40`) |

`analysis.py` treats the tournament with k = 2 as the baseline and the other
tournament size as the control. If `results/` holds more than one other
tournament size, name the control with `--control`.

## What each result file contains

```
results/
  tournament_k2/seed_00/   config.json  history.csv  best_body.json  specialists.json  database.db
  lexicase/seed_00/        (same files)
  tournament_k40/seed_00/  (same files)
  random/seed_00/          config.json  history.csv  best_body.json
  _calibration/            calibration.csv  calibration.json  seed_100/ seed_101/ seed_102/
figures/
  fitness_curve  diversity_curve  selection_strength  closest_per_target  final_fitness  (.pdf and .png)
  summary.csv  stats.csv
```

| File | Contents | In git |
|---|---|---|
| `config.json` | Every setting of that run: selection scheme, k, seed, population, generations, crossover probability, starting-size mode. `analysis.py` reads it to tell the conditions apart | yes (~200 B) |
| `history.csv` | One row per generation: the data behind every figure and statistic (see the columns below) | yes (~15 KB) |
| `best_body.json` | The best body of the run: its fitness, size, distance to each target, and the genome itself (nodes with type and rotation, edges with parent, child and face) | yes (~2.5 KB) |
| `specialists.json` | The same description for the body closest to each of the five targets, taken from the final population. EA runs only: random search has no population to keep | yes (~10 KB) |
| `database.db` | ariel's own record: one row per body ever created (10,100 per run), with its genome, fitness, tags, and the generations it was born and last seen in. We do not use it for the report; it is the full archive for later inspection | **no** (~10 MB per run, 343 MB in total; `run_all.sh` recreates it) |
| `calibration.csv`, `calibration.json` | Mean and std of the number of different parents each selector picks, and the chosen k. The evidence for k = 40 | yes |
| `figures/*.pdf`, `figures/*.png` | The five report figures. PDF for LaTeX, PNG for quick viewing | yes |
| `summary.csv` | Final-generation mean and std per condition. Two header rows: the metric, then `mean`/`std` | yes |
| `stats.csv` | One row per hypothesis test (see the columns below) | yes |


### Columns of `history.csv`

| Column | Meaning |
|---|---|
| `generation`, `evaluations` | Generation 0 is the initial population; evaluations = (generation + 1) × population size |
| `population_size` | Bodies in the population (100; 20 in short test runs) |
| `best_fitness`, `mean_fitness`, `std_fitness` | Fitness = mean + std of the edit distances to the 5 targets (lower is better). For random search, `best_fitness` is the best found so far and the other two describe the current batch |
| `best_size`, `mean_size` | Number of nodes in a body, including the core |
| `diversity` | Mean edit distance between 20 randomly chosen bodies |
| `distinct_parents` | Different bodies among the first `pop_size` parent picks (fewer = stronger selection). Empty for generation 0 and for random search |
| `parent_picks` | All parent picks in the generation; above `pop_size` when children were too big and discarded |
| `closest_7` to `closest_25` | Smallest distance of any body in the population to that target |
| `best_dist_7` to `best_dist_25` | Distance of the best body to each target |

### Columns of `stats.csv`

`hypothesis`, `metric`, the two conditions `a` and `b`, their sizes `n_a`/`n_b`,
their means and standard deviations, the Mann-Whitney `U` and `p`,
`lower_is_better` for the metric, `a12_a_better` (the chance that a run of `a`
beats a run of `b`; 0.5 means no difference) and `p_holm` (`p` corrected within
each hypothesis).


```bash
uv run ruff check --no-fix --config assignments/assignment_1/lint.toml assignments/assignment_1
uv run ruff format --config assignments/assignment_1/lint.toml assignments/assignment_1
```

## Design choices and known limits

Why the settings are what they are, with the numbers taken from the committed
results.

- **Population 100, 100 generations.** Within the range the assignment
  recommends (50 to 100 bodies over 100 generations).
- **Binary tournament (k = 2) as the baseline.** The most common tournament
  size, with weak selection pressure.
- **Crossover probability 0.7.** A common value in the usual 0.6 to 0.9 range.
  It was not tuned: the research question is about selection, so the variation
  operators stay at standard settings and are identical in every condition.
- **One mutation per child, chosen uniformly from ariel's four tree
  mutations.** Every child differs from its parents even when crossover is
  skipped (30% of the time), and no single operator is favoured.
- **Subtree replacement adds a branch of 1 to 3 modules.** This is ariel's
  built-in limit (the code passes `max_modules=3`, which matches it). Small
  branches keep the mutation a local change and rarely push a body over the size
  limit.
- **Tournament without replacement.** A tournament of size k compares k
  different bodies. Sizes below 1 or above the population size are rejected.
- **Initial bodies get 1 to 20 modules.** `random_tree(20)` always builds full
  21-node bodies, which would make random search an unfairly weak baseline.
- **Calibrating the control.** `calibrate_k.py` lets lexicase and tournaments of
  several sizes pick parents from the same lexicase populations: lexicase gave
  8.6 different parents per 100 picks and k = 40 gave 8.7. The final runs
  confirm the match on each condition's own populations: 8.83 for lexicase and
  8.77 for k = 40 (`summary.csv`).
- **Lexicase with five targets is very selective.** It usually ends at the best
  body for one target, so only about 9 different bodies reproduce per
  generation.
- **Diversity is estimated** from 20 randomly chosen bodies (190 pairs) to keep
  runs fast.


## Reproducibility notes

- Every run seeds `random` and `numpy` with `--seed`, and the tree operators
  draw from that same seeded generator. The same seed gives the same
  `history.csv` (there is a test for it).
- Final experiments use seeds 0 to 9; calibration uses seeds 100 to 102.
- Diversity is measured with its own generator, so logging it cannot change a
  run.
- Library versions are pinned by `uv.lock`, which is what makes the byte-for-byte
  rebuild dependable.
- The ariel framework (`src/ariel`) is not modified.
