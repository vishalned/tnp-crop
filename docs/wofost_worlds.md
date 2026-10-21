# Sampled WOFOST worlds

More diverse simulated pretraining data, varied in a controlled way, so that simulated regions differ the
way real ones do (variety, crop physiology, irrigation, sowing). The original pipeline is unchanged and
is dataset A.

## World = location + hidden settings, simulated for every year

A **world** is one location with one set of settings, simulated for every year 2005–2020 with those same
settings. A region's yield/feature relation is then the same in all its years, which is what the model
learns from a region's other years. `worlds_per_location` worlds (default 10) are drawn per location.

| Level | Setting | Values |
|---|---|---|
| world | cultivar | uniform over `Grain_maize_201`…`205` (they differ only in TSUM1/TSUM2; `Maize_VanHeemst_1988` = whole cob and `Fodder_maize_nl` are left out) |
| world | production | `PP` (potential = irrigated) with probability = the region's irrigated share, else `WLP` (water-limited, multi-layer water balance, as before) |
| world | crop-parameter factors | one factor per parameter, Latin hypercube over a location's worlds |
| world | sowing shift | uniform integer in ±10 days around the crop calendar's SOS |
| year | sowing jitter | uniform integer in ±3 days on top of the shift |

None of these is a model input. They are saved per world in `worlds.parquet` (`world_id`, location,
cultivar, production, irrigated share, `factor_<P>`, sowing shift, seed, per-year sowing offsets, kept /
reject reasons).

**Seeding.** Every draw comes from its own random stream, seeded by (seed, location, component). So the
same seed gives the same worlds whatever the worker count or location subset. Datasets that differ in
one setting also share all other draws: B and C have the same cultivars and sowing dates.

## Crop-parameter factors

| Parameter | Default range | Note |
|---|---|---|
| TSUM1, TSUM2 | [0.9, 1.1] | narrower: the cultivar already varies them |
| AMAX_REF | [0.85, 1.15] | WOFOST 8.1 maize has no `AMAXTB`; AMAX = min(AMAX_REF, AMAX_SLP × leaf N) |
| EFFTB | [0.85, 1.15] | table over daily mean **temperature**; value column scaled |
| SPAN | [0.85, 1.15] | |
| SLATB | [0.85, 1.15] | table over development stage; value column scaled |
| RDMCR | [0.85, 1.15] | rounded to 5 cm (`rdmcr_step_cm`), ≤ soil depth (200 cm) |

- Tables are PCSE's flat `[x0, y0, x1, y1, …]` lists. Only the y values are multiplied
  (`worlds.scale_crop_parameters`), and the overrides go through `ParameterProvider.set_override`.
- **RDMCR and the soil layers.** The multi-layer water balance needs the max rooting depth on a soil-layer
  boundary (0/5/15/30/60/100/200 cm). The original pipeline rounds RDMCR down to one, which would turn
  the factor into a 60-or-100 cm switch. With an RDMCR override, the layer containing it is instead
  split into two identical layers (`pcse_runner.split_soil_layer_at`). After the run, the per-layer
  outputs are merged back to the six standard layers (`merge_split_layers`: water amounts add up,
  volumetric contents are thickness-weighted). So `ssm`/`rsm` use the same layers as before.

## Irrigation (PP)

- PCSE's `Wofost81_PP` has no soil water balance, so on its own it would give no `ssm`/`rsm`. An
  irrigated world-year is therefore run twice:
  - PP for the crop (yield, LAI/fpar, phenology);
  - the rain-fed water-limited model with the same settings for the soil columns. That matches
    GLDAS, which doesn't model irrigation either.
- PP's classic soil bucket still needs scalar soil parameters (SM0, SMFCF, SMW, CRAIRC); they are taken
  from the topsoil's pF curve. In PP the soil stays at field capacity, so they don't limit growth.
- **Irrigated share:** read from `irrigation_share_csv` (`region_id,irrigated_share`, region = the
  locations CSV's `country` column by default). **There is no MIRCA2000 data yet**, so datasets D/E use a
  placeholder `default_irrigated_share: 0.5` for every region: each world is irrigated (PP) or rain-fed (WLP)
  with equal probability.

## Filtering broken worlds

After a world is simulated it is dropped (and logged) if more than 50% of its years:
- did not reach maturity by the crop calendar's EOS (`max_immature_fraction`);
- had a yield below 0.5 t/ha (`min_yield_t_ha`, `max_low_yield_fraction`); or
- failed to simulate (`max_failed_fraction`).

`world_rejections.csv` gives the rejection rate per country and reason. The manifest gets `world_kept`,
and `process_wofost_dataset` leaves rejected worlds out.

## Datasets (`configs/wofost_worlds/`)

| Dataset | Settings |
|---|---|
| A | the original setup (`sowing_scheme: legacy`: `generate_wofost_dataset`, 3 jitters per year in ±10 days, `Grain_maize_201`, WLP); reproduces `wofost_calendar` |
| A_w | worlds, sowing shift + jitter only (isolates the sowing change from the cultivar) |
| B | A_w + random cultivar |
| C | B + crop-parameter factors |
| D | C + irrigation (placeholder share) |
| E | D + `extra_locations_csvs` (none yet; new locations need GEE downloads) |

All have the same locations (`max_locations: 1585` = the cached ones; E: all), years and seed.

**Why staged.** Each dataset adds one source of variation to the previous one, and the shared random
streams keep everything else identical. So a change in a check or in downstream model performance from
one dataset to the next can be attributed to that one addition (e.g. C vs B = the parameter factors
only). E is the intended full pretraining dataset; A–D are ablation steps on the way. D (not E) is the
largest one that needs no new GEE downloads.

```bash
uv run python -m src.data_pipeline.wofost.generate_worlds_dataset --config configs/wofost_worlds/dataset_C.yaml \
    [--location-indices 67 172] [--worlds-per-location 3] [--workers 16] [-o <dir>]
uv run python -m src.data_pipeline.wofost.process_wofost_dataset --manifest <dir>/dataset_manifest.csv \
    --locations-csv data/raw/locations/locations_maize.csv --wofost-dir <dir> -o <dir>/processed
```

Outputs per dataset:
- `dataset_manifest.csv`: one row per run, with `world_id`, `cultivar`, `production`, `flowering_date`,
  `maturity_date`, `runtime_s` and `world_kept`;
- `worlds.parquet`, `world_rejections.csv`, `config.yaml`;
- the usual daily CSVs (file names end in the `world_id`).

## Training tables

- In the processed table every world is its own point: `location_index = site_index × 1000 +
  world_index`, with the location in `site_index` and the world in `world_id`. Dataset A keeps its
  original ids.
- `CropTables` uses at most **one world per site per training table**, and counts region sizes
  (`min_points`, points per table) in sites. Tables without `site_index` are sampled exactly as before.
- The walk-forward evaluation tables are unchanged: they use all worlds.
- The features are the same CY-Bench features as before (`cybench_features.py`).

## Sensitivity analysis and checks

- `src/data_pipeline/wofost/sensitivity_morris.py`: Morris (SALib) over the factor ranges.
  - Setup: rain-fed, default cultivar, 5 cached locations spread over latitude, 3 years.
  - Outputs: yield, days to flowering, days to maturity.
  - Writes `morris_per_site_year.csv`, `morris_summary.csv` and `morris_mu_star.png`.
- `src/data_checks/worlds_checks.py`:
  1. A reproduces the reference manifest;
  2. flowering/maturity timing per country vs the calendar;
  3. yield per dataset and country;
  4. feature–yield correlation agreement with CY-Bench (from `nanotabicl.distribution_gap` on each
     dataset's processed table);
  5. runs, rejections and runtime, with CPU hours for 100k/300k runs.
- Test job: `scripts/slurm/run_worlds_test.sh` (7 cached locations, 3 worlds; every step above).
- Unit tests: `tests/test_worlds.py`.
