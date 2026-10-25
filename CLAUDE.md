# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Crop-yield prediction from WOFOST-simulated growing seasons. Parts of the repo:

1. **Data pipeline** (`src/data_pipeline/`, `src/data_checks/`) — picks locations, pulls soil (SoilGrids) and weather (ERA5-Land) from Google Earth Engine, runs WOFOST via PCSE, and turns the runs into one training table per crop.
2. **Models**: `nanotabicl/`, self-contained plain-PyTorch NanoTabICLv2 (TabICLv2) training/evaluation. Does **not** use Hydra/Lightning; it only reads the processed Parquet table.
3. **Template skeleton**: `src/train.py`, `src/eval.py`, `src/utils/`, `configs/` from the lightning-hydra-template, kept basic. The old TNP-D model/datamodule code was removed; `configs/data/tnp.yaml`, `configs/model/tnp.yaml` and `configs/experiment/tnp_synthetic.yaml` are kept for reference but point to removed modules, so `src/train.py` needs a new model/datamodule before it runs.
4. **Analysis**: `analysis/plot_results.py` makes the result figures (`analysis/figures/`) from `docs/results_*.csv`; `analysis/scaling/` has the EU grid (scaling) figures, described in `docs/scaling_report.md`.

`docs/manual.md` is a full walkthrough of the pipeline end to end (with file/function pointers and known simplifications); `docs/data_dictionary.md` describes every column; `nanotabicl/README.md` covers the model side in detail. Read these before changing behavior in those areas, and keep them in sync when you do.

## Environment & commands

Uses `uv` (not conda); `uv.lock` and `.python-version` are committed, and `uv run` handles the `.venv`. Extras: `train` (torch/lightning), `data` (pcse, earthengine, zarr…), `nanotabicl` (needs `train` too for torch), `dev`.

```bash
uv sync --extra train --extra nanotabicl          # NanoTabICL work
uv sync --extra data                              # data pipeline
uv add <pkg> / uv add --optional <extra> <pkg>    # never hand-edit deps without `uv lock && uv sync`

uv run pytest tests/test_nanotabicl.py            # tests (synthetic table fixture, tiny model; no data/ needed)
uv run pytest tests/test_nanotabicl.py -k walk_forward   # single test
uv run pytest -k "not slow"                       # `make test`
pre-commit run -a                                 # `make format`: black, isort, flake8, docformatter, codespell, ...
```

`test_official_checkpoint_conversion_is_exact` is skipped unless the `tabicl` package is installed. pytest runs with `--doctest-modules` and `--strict-markers`.

Everything is run as modules from the repo root: `uv run python -m <package.module> ...`.

### Pipeline order (maize shown; wheat is analogous)

```bash
uv run python -m src.data_pipeline.locations.extract_locations_from_zarr --zarr-path ... --crop maize -n 0 --seed 42
uv run python -m src.data_pipeline.wofost.generate_wofost_dataset --locations-csv data/raw/locations/locations_maize.csv   # CPU job: scripts/slurm/run_cpu.sh
uv run python -m src.data_checks.coverage_audit   --manifest data/raw/wofost/dataset_manifest.csv --locations-csv data/raw/locations/locations_maize.csv
uv run python -m src.data_checks.simulation_audit --manifest data/raw/wofost/dataset_manifest.csv --locations-csv data/raw/locations/locations_maize.csv
uv run python -m src.data_pipeline.wofost.process_wofost_dataset --manifest data/raw/wofost/dataset_manifest.csv --locations-csv data/raw/locations/locations_maize.csv
uv run python -m nanotabicl.checkpoint             # download+convert official weights (needs internet: run on a login node)
uv run python -m nanotabicl.train --init official --data crop --crop-episodes structured --table data/processed/wofost_maize_daily.parquet ...
uv run python -m nanotabicl.evaluate --checkpoint <out>/best.pt --table ... --years 2019 2020   # --checkpoint official = zero-shot
sbatch scripts/slurm/run_gpu.sh                    # GPU training + test eval on an A100 (MODE=structured|random)
```

## Setup prerequisites (one-time, per machine)

- **Earth Engine:** authenticate from inside Python in the uv env (`ee.Authenticate()`); the shell `earthengine authenticate` is unreliable. `ensure_ee_initialized()` uses the Cloud project `$EE_PROJECT` (default `helical-kayak-399407`, display name "cropfm"; use the project **ID**). Credentials live in `~/.config/earthengine/credentials`; if the notebook auth page lists no projects, run `ee.Authenticate(auth_mode="localhost")` on a laptop and copy that file over.
- **Crop parameters:** a local clone, not fetched at runtime, from the `herman-berghuijs` fork (has the C4/maize fix): `git clone -b wofost81 https://github.com/herman-berghuijs/WOFOST_crop_parameters.git data/crop_parameters/wofost81`. `data/` is gitignored.

## Key concepts that span files

- **Unit = season**: one location × sowing year × sowing-date jitter. The batch runner simulates every sowing year 2005–2020 × `--num-jitters` (3) offsets within ±10 days, so 48 seasons per location. Offsets are pre-drawn from one seeded RNG, so results don't depend on `--workers`. Rows with the same coordinates go to the same worker because soil/weather caches are one file per coordinate. Failed seasons are recorded in `dataset_manifest.csv` with `status="failed"`; the manifest is rewritten after each location.
- **`location_index` = row number of `locations_{crop}.csv`**; processing joins `country` back on it, so don't reorder/filter that CSV after a run.
- **Crop calendar**: simulations sow at the location's WorldCereal SOS ± jitter and run from SOS − 60 days to at least EOS (soil water balance on bare soil before sowing / after maturity), so soil moisture covers CY-Bench's feature window.
- **Processed table** `wofost_{crop}_daily.parquet` (`--align calendar`, default): one row per season, `year` = harvest year, `sowing_year` separate, daily series over `[SOS − 60, EOS]` as `<var>_dNNN` columns in CY-Bench units (NaN after a row's EOS); a sibling `..._columns.json` lists the column groups. Current maize run: `/projects/prjs1788/tnp-crop/` (home is nearly full).
- **Features = CY-Bench's tabular features** (`nanotabicl/cybench_features.py`, spec `docs/cybench_features.md`): the model's input format must stay identical to what AgML-CY-Bench-Modeling builds (names, order, statistics); `tests/test_cybench_features.py` checks it against the benchmark's own code. `vpd`, `ndvi`, `drainage_class` are left out (not in WOFOST).
- **Year split by sowing year:** train 2005–2016, val 2017–2018, test 2019–2020. Val/test are always walk-forward (context = all earlier years of the region's points, queries = points at year T).
- **Regions = countries**; regions with < `--min-points` (10) points are dropped. Training tables: `structured` (target year + 5–11 earlier context years per point) or `random` (TabICL-style random context/query split). All tables in a batch share their shape (no padding).
- **Regression objective** matches official TabICL: 999 quantiles, pinball loss, target standardized per table with context-row mean/std (inverted for predictions).
- **GPU memory:** each cell is a 128-dim token (maize `eos`: 153 features). `--micro-batch-size` and `--checkpoint-activations` must not change the update (there is a test for this); on a 40 GB A100 use `--micro-batch-size 1 --checkpoint-activations`.
- Run outputs: `--out-dir` gets `config.json`, `log.csv`, `best.pt` (lowest val RMSE), `last.pt`; evaluate writes `metrics_<years>.json` and `predictions_<years>.csv`.

## Conventions

- `nanotabicl/model.py` and `nanotabicl/prior.py` are vendored unchanged from soda-inria/nanotabicl @ `4a7f9c7` (BSD-3). Don't edit them; put changes in our own files (`checkpoint.py`, `crop_tables.py`, `prior_tables.py`, `train.py`, `evaluate.py`).
- `checkpoint.py`'s conversion of the official TabICLv2 weights is verified to be exact (all weights randomised, real grouping settings; `NanoTabICL` with `official_feature_grouping` fixes the vendored model's off-by-one column grouping); keep it that way.
- `NanoTabICLRegressor(n_estimators=k)` = TabICL's test-time ensemble (`ensemble.py`), tested equal to `TabICLRegressor(n_estimators=k)`; compare against default TabICL with `n_estimators=8`.
- On the HPC cluster, run heavy jobs through SLURM (`scripts/slurm/`), not on login nodes; logs go to `slurm_logs/`.
- **Do not run heavy computations on the login node.** For heavy things (training, simulations, dataset processing, audits, test suites with real data) either submit a SLURM job yourself (`sbatch`, CPU: like `scripts/slurm/run_cpu.sh`, GPU: `run_gpu.sh`) and read its log in `slurm_logs/`, or ask the user to run it with `srun`.
- Home (`/gpfs/home2`) is nearly full (quota and inodes): write large outputs (simulations, processed tables) to `/projects/prjs1788/tnp-crop/`.
- The benchmark is `../AgML-CY-Bench-Modeling` (WUR-AI/AgML-CY-Bench-Modeling, run with poetry). Ask before changing it; changes go on a local branch there, never pushed.
