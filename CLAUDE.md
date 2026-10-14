# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Crop-yield prediction from WOFOST-simulated growing seasons. Two independent halves share one repo:

1. **Data pipeline** (`src/data_pipeline/`, `src/data_checks/`) — picks locations, pulls soil (SoilGrids) and weather (ERA5-Land) from Google Earth Engine, runs WOFOST via PCSE, and turns the runs into one training table per crop. Shared with `main`.
2. **Models**
   - `nanotabicl/` (the focus of the `tabICL` branch): self-contained plain-PyTorch NanoTabICLv2 (TabICLv2) training/evaluation. Does **not** use Hydra/Lightning; it only reads the processed Parquet table.
   - `src/train.py`, `src/models/`, `src/data/`, `configs/` — the TNP-D model on the lightning-hydra-template (Hydra configs, Lightning modules). Currently only a synthetic smoke test (`configs/experiment/tnp_synthetic.yaml`).

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

TNP side: `python src/train.py experiment=tnp_synthetic` (Hydra overrides as usual).

## Setup prerequisites (one-time, per machine)

- **Earth Engine:** authenticate from inside Python in the uv env (`ee.Authenticate()`); the shell `earthengine authenticate` is unreliable. `ensure_ee_initialized()` uses project `"cropfm"`.
- **Crop parameters:** a local clone, not fetched at runtime, from the `herman-berghuijs` fork (has the C4/maize fix): `git clone -b wofost81 https://github.com/herman-berghuijs/WOFOST_crop_parameters.git data/crop_parameters/wofost81`. `data/` is gitignored.

## Key concepts that span files

- **Unit = season**: one location × sowing year × sowing-date jitter. The batch runner simulates every sowing year 2005–2020 × `--num-jitters` (3) offsets within ±10 days, so 48 seasons per location. Offsets are pre-drawn from one seeded RNG, so results don't depend on `--workers`. Rows with the same coordinates go to the same worker because soil/weather caches are one file per coordinate. Failed seasons are recorded in `dataset_manifest.csv` with `status="failed"`; the manifest is rewritten after each location.
- **`location_index` = row number of `locations_{crop}.csv`**; processing joins `country` back on it, so don't reorder/filter that CSV after a run.
- **Processed table** `data/processed/wofost_{crop}_daily.parquet`: one row per season, `year` = harvest year, `sowing_year` separate, daily series kept unaggregated as `<var>_dNNN` columns; a sibling `..._columns.json` lists id/static/time-series/target columns. Aggregation happens downstream (`CropTables`, `--bucket-days`, 7 = weekly; sums for prec/rad/et0/cwb, means otherwise).
- **Year split by sowing year:** train 2005–2016, val 2017–2018, test 2019–2020. Val/test are always walk-forward (context = all earlier years of the region's points, queries = points at year T).
- **Regions = countries**; regions with < `--min-points` (10) points are dropped. Training tables: `structured` (target year + 5–11 earlier context years per point) or `random` (TabICL-style random context/query split). All tables in a batch share their shape (no padding).
- **Regression objective** matches official TabICL: 999 quantiles, pinball loss, target standardized per table with context-row mean/std (inverted for predictions).
- **GPU memory:** weekly features are wide (346 maize / 508 wheat), each cell is a 128-dim token. `--micro-batch-size` and `--checkpoint-activations` must not change the update (there is a test for this); on a 40 GB A100 use `--micro-batch-size 1 --checkpoint-activations`.
- Run outputs: `--out-dir` gets `config.json`, `log.csv`, `best.pt` (lowest val RMSE), `last.pt`; evaluate writes `metrics_<years>.json` and `predictions_<years>.csv`.

## Conventions

- `nanotabicl/model.py` and `nanotabicl/prior.py` are vendored unchanged from soda-inria/nanotabicl @ `4a7f9c7` (BSD-3). Don't edit them; put changes in our own files (`checkpoint.py`, `crop_tables.py`, `prior_tables.py`, `train.py`, `evaluate.py`).
- `checkpoint.py`'s conversion of the official TabICLv2 weights is verified to be exact; keep it that way.
- On the HPC cluster, run heavy jobs through SLURM (`scripts/slurm/`), not on login nodes; logs go to `slurm_logs/`.
- **Do not run heavy computations on the login node.** If something heavy needs to run (training, simulations, dataset processing, audits, full test suites with real data), ask the user to run the script; they will run it with `srun`.
