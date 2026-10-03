# NanoTabICL on WOFOST crop tables

Self-contained research code: NanoTabICLv2 (a ~170-line TabICLv2), a
regression training loop, and tables built from the WOFOST simulations.
It doesn't use the Hydra/Lightning setup in `src/` (that's the TNP side);
it only reads the processed table `src/data_pipeline` produces.

| File | What | From |
|---|---|---|
| `model.py` | `NanoTabICLv2` | vendored unchanged from [soda-inria/nanotabicl](https://github.com/soda-inria/nanotabicl) @ `4a7f9c7` (BSD-3, `LICENSE_nanotabicl`) |
| `prior.py` | TabICLv2 "nanoprior" (synthetic tables) | same |
| `checkpoint.py` | official TabICLv2 weights -> `NanoTabICLv2`; save/load; model sizes | ours |
| `crop_tables.py` | context/query tables from the processed WOFOST table | ours |
| `prior_tables.py` | batches of prior tables for regression | ours, on `prior.py` |
| `train.py` | (continued) pretraining loop | ours, in the style of [nanoTabPFN](https://github.com/automl/nanoTabPFN)'s `train.py` |
| `evaluate.py` | walk-forward evaluation on val/test years | ours |

Tests: `uv run pytest tests/test_nanotabicl.py`.

## Why this layout

- nanotabicl ships the model and prior only: no pretrained checkpoint and no
  training code ("we refer to nanoTabPFN and the nanoTabPFN speedrun"). The
  pretrained TabICLv2 weights are the official ones on the Hugging Face Hub
  (`jingang/TabICL`), for the `tabicl` package's model. `checkpoint.py`
  converts them: same architecture, renamed parameters. The conversion is
  checked to reproduce the official model's outputs exactly (float64, see the
  tests). The `tabicl` package itself isn't needed at runtime.
- The training loop follows nanoTabPFN's plain `train.py` rather than the
  speedrun repo (modded-nanotabpfn), which is classification-only, TabPFN-
  specific and heavily optimised (Muon, compiled blocks, custom
  dataloaders). Those tricks can be ported later if speed matters.
- Regression objective = the official one (`tabicl/train/_run.py`): 999
  quantiles at levels `linspace(0, 1, 1001)[1:-1]`, pinball loss, target
  standardised per table with its context rows' mean/std.

## Data: tables from the WOFOST simulations

Input = `process_wofost_dataset.py` output, one row per location × sowing
year × jitter (static soil features from the per-location GEE soil files,
daily weather/crop series, yield):

```bash
uv run python -m src.data_pipeline.wofost.process_wofost_dataset --manifest data/raw/wofost/dataset_manifest.csv \
    --locations-csv data/raw/locations/locations_wheat.csv      # -> data/processed/wofost_wheat_daily.parquet
```

`CropTables` turns each row into a feature vector: `latitude, longitude,
awc, bulk_density` plus every daily series aggregated into `--bucket-days`
buckets (7 = weekly; sums for prec/rad/et0/cwb, means otherwise). A training
table is: a country (uniform), `n_points` of its points, a target year T and
per point `n_context_years` earlier years, one jitter per (point, year):
context rows = those cells with their yields, query rows = the points at T.
All tables in a batch share `n_points` × `n_context_years`, so batches need
no padding. Year split by sowing year: train 2005–2016 (T and context),
val 2017–2018 and test 2019–2020 (walk-forward: context = every earlier
year of the country's points, queries = the points at T).

## Running

```bash
uv sync --extra train --extra nanotabicl

# 1. the pretrained checkpoint (needs internet once; cached in ~/.cache/huggingface).
#    On a cluster, run this on a login node before submitting jobs.
uv run python -m nanotabicl.checkpoint          # downloads + converts tabicl-regressor-v2-20260212.ckpt, one forward pass

# 2a. continue pretraining the official weights on WOFOST tables
uv run python -m nanotabicl.train --init official --data crop \
    --table data/processed/wofost_wheat_daily.parquet --lr 3e-5 --steps 5000 --out-dir logs/nanotabicl/continue_wheat

# 2b. from scratch on WOFOST tables, the TabICLv2 prior, or both
uv run python -m nanotabicl.train --init scratch --size small --data mix --prior-fraction 0.5 \
    --table data/processed/wofost_wheat_daily.parquet --lr 1e-3 --steps 50000 --num-workers 8 --out-dir logs/nanotabicl/scratch_mix

# 3. walk-forward test evaluation (2019-2020); --checkpoint official = zero-shot baseline
uv run python -m nanotabicl.evaluate --checkpoint logs/nanotabicl/continue_wheat/best.pt --table data/processed/wofost_wheat_daily.parquet
uv run python -m nanotabicl.evaluate --checkpoint official --table data/processed/wofost_wheat_daily.parquet
```

`--init` also takes a path: a manually downloaded official `.ckpt`
(`huggingface-cli download jingang/TabICL tabicl-regressor-v2-20260212.ckpt`)
or one of our `.pt` checkpoints (to resume). `--size base` = the official
dimensions (~29M parameters) for from-scratch runs; `small` = the
nanotabicl README's small regression model.

Each run writes `config.json`, `log.csv` (loss, lr, val RMSE/R² vs the
per-point-mean baseline), `best.pt` (lowest val RMSE) and `last.pt` to
`--out-dir`. Evaluation writes `metrics_<years>.json` and
`predictions_<years>.csv` (one row per query: true/predicted yield and the
baseline).

## Notes

- `NanoTabICLv2` standardises the features inside the model using the context
  rows; we standardise the target per table and invert it for predictions.
- The regression checkpoint has bias-free LayerNorms; nano's LayerNorms get
  zero biases at conversion (same function) which then train normally.
- Prior tables are generated on the fly. The prior's ExtraTrees filtering
  makes it CPU-bound (~0.3 s per small table), so use `--num-workers`.
