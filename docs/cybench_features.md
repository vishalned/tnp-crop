# CY-Bench tabular features, and how WOFOST maps onto them

Spec of the feature table that [AgML-CY-Bench-Modeling](https://github.com/WUR-AI/AgML-CY-Bench-Modeling)
("the benchmark", cloned at `../AgML-CY-Bench-Modeling`, commit `1af21b1`) builds for its tabular
models (ridge, RF, XGBoost, LightGBM, TabPFN, TabICL, TabDPT), read from the code; and the mapping of
each feature to the WOFOST/ERA5-Land data in this repo. Goal: NanoTabICL pretraining tables with
exactly these columns, in this order and form.

CY-Bench data used for reference: `/projects/prjs1788/CropBench/data/cybench/cybench-data/<crop>/<CC>/`.

## 1. Which configuration the benchmark uses for tabular models

The cluster jobs (`cybench/runs/slurm/models.txt` + `slurm_common.sh`) run every tabular model
(`feature_design = yes`) with:

| setting | value | config |
|---|---|---|
| temporal features | `dataset/temporal=feature_design` | `conf/dataset/temporal/feature_design.yaml` |
| static features | `non_temporal: geometric_transform` (dataset default) | `conf/dataset/non_temporal/geometric_transform.yaml` |
| normalizer | `normalizer: fit` (dataset default) | `conf/dataset/normalizer/fit.yaml` |
| horizon | `dataset.temporal.season.end_of_sequence=$PREDICTION_HORIZON`, default **`eos`**; the paper/dashboard also runs `middle-of-season` (and `quarter-of-season`, `early-season`) | `slurm_common.sh` |
| feature selection | `+feature_selection=mrmr` (k = 50, tuned 10–150 in HPO) | `conf/feature_selection/mrmr.yaml` |
| targets | `yield` (t/ha), harvest year; samples flagged `flag_yield_outlier` dropped | `conf/dataset/target/yield.yaml` |

Note: the dataset default `temporal: no_aggregate` (daily, unaggregated) is **not** what tabular
models get in the benchmark runs; `feature_design` is set explicitly.

**mRMR** is only applied when `+feature_selection=mrmr` is on the command line
(`run_experiments.py` → `util/feature_selection.apply_mrmr_at_origin`, refit at each forecast
origin on the training years). Switch it off by not passing it (and, for walk-forward, by using a
frozen screening dir without `optimal_feature_selection.yaml`). The feature table itself
(`DataFactory.build`) never depends on it.

## 2. How the table is built (`cybench/datasets/data_factory.py`, `alignment.py`)

Per country, `DataFactory.load_dfs`:

1. **Targets** — `yield_<crop>_<CC>.csv`, `harvest_year` → `year`; NaN and yield ≤ 0 dropped; years
   in `[min_year, max_year]` = [2000, 2024]; quality-flagged rows dropped (needs
   `yield_quality_<crop>_<CC>.csv`, made by `data_preparation/assess_yield_quality.py`; **not in our
   copy of the data**).
2. **Static features** (`load_non_temporal`), sources in this order, each transformed, merged on `adm_id`:
   - `crop_calendar`: `sos`, `eos` (day of year) → cyclical, period 365: `sos_sin, sos_cos, eos_sin, eos_cos`
     (from the **float** doy, not truncated);
   - `location`: `latitude, longitude` → unit sphere: `loc_x, loc_y, loc_z`;
   - `soil`: `awc, bulk_density, drainage_class` → `drainage_class` one-hot over classes 1–6:
     `awc, bulk_density, drainage_class_1 … drainage_class_6`.
   - NaN → column mean.
3. **Season window** (`alignment.compute_crop_season_window`), per region and year, from the crop
   calendar `crop_calendar_<crop>_<CC>.csv` (ESA WorldCereal SOS/EOS, region average, float doy):
   - `sos`, `eos` truncated to int; `sos_date`/`eos_date` = that doy in the year (if `sos > eos` the
     season started the previous year);
   - `start_of_sequence` = `sos − 60 days` (`start_of_sequence: sos-60`);
   - `end_of_sequence` = `eos` for horizon `eos`; otherwise `eos − cutoff`, with cutoff =
     `season_length // 2` (`middle-of-season`), `season_length // 4` (`quarter-of-season`),
     `3·season_length // 4` (`early-season`), or `N` (`eos-N`); `season_length = eos_date − sos_date`.
4. **Alignment** (`align_to_crop_season_window_numpy`): daily rows keep `start ≤ date ≤ end`; a date
   after `eos` belongs to the next year's season; a (region, year) is dropped for a source if its data
   doesn't span the window (±30 days tolerance for sparse sources).
5. **Created features** (`feature_design.py`, in config order, on the aligned daily rows):
   `gdd = clip(tavg − tbase, 0, upper_limit)` (maize: tbase 10, upper 35 — the cap is on the GDD,
   not on tavg; wheat: tbase 0, no cap); `cum_gdd`, `cum_prec`, `cum_cwb`, `cum_ndvi`, `cum_fpar` =
   running sums **from the window start** (sos − 60) per (region, year), over the source's own
   observation dates.
6. **Aggregation** (`_aggregate_time_series`): 30-day windows **anchored at `end_of_sequence`**:
   `window = (end_of_sequence − date).days // 30`, so window 0 = the 30 days ending at (and
   including) the cutoff, window 1 the 30 before, … The earliest window is usually partial (sums
   then cover fewer days). Statistics:

   | source (file) | variable | resolution | aggregated columns |
   |---|---|---|---|
   | meteo (AgERA5) | `tmin`, `tmax`, `tavg` (°C) | daily | `tmin_min`, `tmax_max`, `tavg_mean` |
   | | `prec` (mm/day) | daily | `prec_sum`, `prec_max` |
   | | `rad` (J m⁻² day⁻¹) | daily | `rad_mean` |
   | | `et0` (mm/day, FAO-56 PM) | daily | `et0_mean` |
   | | `vpd` (hPa, AgERA5 VPD at daily Tmax) | daily | `vpd_mean` |
   | | `cwb` = prec − et0 (mm/day) | daily | `cwb_mean` |
   | | `gdd` (created) | daily | `gdd_sum` |
   | | `cum_gdd`, `cum_prec`, `cum_cwb` (created) | daily | `cum_gdd_max`, `cum_prec_max`, `cum_cwb_mean` |
   | ndvi (MOD09CMG) | `ndvi` (0–1) | ~8-day (`0101, 0109, 0117, …`) | `ndvi_mean`, `cum_ndvi_max` |
   | fpar (JRC FPAR 500 m) | `fpar` (%, 0–100) | dekadal (days 1, 11, 21) | `fpar_mean`, `cum_fpar_max` |
   | soil_moisture (GLDAS) | `ssm` (kg m⁻², 0–10 cm), `rsm` (kg m⁻², root zone) | daily, from 2003 | `ssm_mean`, `rsm_mean` |

7. **Normalization** (`normalizer.py`, `fit`), applied to the aggregated frames and the statics
   **before** tabularization: every aggregated column above and `awc`, `bulk_density` →
   `(x − mean) / std`, mean/std per **variable** (pooled over all windows), fitted on the screening
   train+val years (`_resolve_normalizer_fit_years`), applied to all rows. The `logsinh` entries
   (`prec`, `et0`, `vpd`, `cwb`) never match an aggregated column name, so nothing is log-transformed;
   `sos_*`, `eos_*`, `loc_*`, `drainage_class_*` aren't listed → left as is. So per column the
   normalization is an affine map.
8. **Tabularization** (`_tabularize`): per source, pivot to one row per (`adm_id`, `year`), columns
   `<feature>_<window>`. `pivot_table` sorts columns: **alphabetically by feature, then by window
   number** (0 = latest). Sources are concatenated in config order (meteo, ndvi, fpar, soil_moisture),
   then the statics are appended. Missing windows → NaN; columns with NaN rate > 5 %
   (`max_column_nan_rate`) dropped. Remaining NaNs reach the model untouched (TabPFN/TabICL with
   `preprocess: none`).
9. Features are cast to float32; `X = dataset.x`, index (`adm_id`, `year`).

### Expected column order, maize DE, horizon `eos`

The DE crop calendar has sos 101–115 and eos 297–304, so the window `[sos − 60, eos]` is 248–261 days,
giving **9 windows (0–8)** in every region (window 8 is partial: 9–22 days). Derived from the code
(to be confirmed by running `DataFactory.build` — see the open points):

```
meteo (13 × 9):  cum_cwb_mean_0..8, cum_gdd_max_0..8, cum_prec_max_0..8, cwb_mean_0..8, et0_mean_0..8,
                 gdd_sum_0..8, prec_max_0..8, prec_sum_0..8, rad_mean_0..8, tavg_mean_0..8,
                 tmax_max_0..8, tmin_min_0..8, vpd_mean_0..8
ndvi  (2 × 9):   cum_ndvi_max_0..8, ndvi_mean_0..8
fpar  (2 × 9):   cum_fpar_max_0..8, fpar_mean_0..8
soil_moisture (2 × 9): rsm_mean_0..8, ssm_mean_0..8
statics (15):    sos_sin, sos_cos, eos_sin, eos_cos, loc_x, loc_y, loc_z,
                 awc, bulk_density, drainage_class_1..6
```

That gives 171 + 15 = **186 columns** with the benchmark's full config, unless some sparse NDVI window crosses the 5 % NaN
threshold. For `middle-of-season` the window is 154–161 days, giving 6 windows (0–5) and 114 + 15 columns.

With the features NanoTabICL uses (without `vpd`, `ndvi`, `drainage_class`, see §4) and 9 blocks:
meteo 12 × 9, fpar 2 × 9, soil moisture 2 × 9, and 9 statics (`sos_sin, sos_cos, eos_sin, eos_cos,
loc_x, loc_y, loc_z, awc, bulk_density`) = **153 columns**.

The number of windows depends on each country's season length, so the column set is per
(crop, country, horizon), not global.

## 3. Mapping to the WOFOST data

The WOFOST side (`docs/data_dictionary.md`): per-location ERA5-Land weather caches (full calendar
years, PCSE units) and the episode CSVs. Since the calendar re-simulation
(`scripts/slurm/run_wofost_cybench.sh`), every episode runs from the location's WorldCereal SOS − 60
days until at least its EOS, with sowing at SOS ± 10 days, so soil moisture exists for the whole
window. `process_wofost_dataset.py --align calendar` stores the daily series over
`[SOS − 60, EOS]` in CY-Bench units; `nanotabicl/cybench_features.py` builds the features.

| benchmark feature | WOFOST source | conversion | status |
|---|---|---|---|
| `tmin`, `tmax`, `tavg` | `TMIN`, `TMAX`, `TEMP` | none (°C) | ✓ same variable (ERA5-Land vs AgERA5) |
| `prec` | `RAIN` | × 10 (cm → mm/day) | ✓ |
| `rad` | `IRRAD` | none (J m⁻² day⁻¹) | ✓ |
| `et0` | `ET0` (PCSE Penman-Monteith) | × 10 (cm → mm/day) | ✓ close (AgERA5 FAO-56 PM vs PCSE PM) |
| `cwb` | `RAIN − ET0` | × 10 | ✓ same definition |
| `gdd`, `cum_*` | from the above | as the benchmark | ✓ |
| `fpar` | WOFOST `1 − exp(−k·LAI)` | × 100 (%), dekad dates only | ⚠ crop canopy only: 0 before sowing / after maturity; satellite FPAR is the whole pixel |
| `ssm` | `WC_layer0` + ½ `WC_layer1` | × 10 (cm → kg m⁻², 0–10 cm) | ⚠ simulated (multi-layer water balance) vs GLDAS |
| `rsm` | `WC_layer0..4` | × 10 (cm → kg m⁻², 0–100 cm) | ⚠ same |
| `sos`, `eos` | WorldCereal calendar at the point (`tc-maize-main_sos/eos` in the locations CSV) | none | ✓ same source as CY-Bench's calendar (point vs region mean) |
| `latitude`, `longitude` | locations CSV | spherical → `loc_x/y/z` | ✓ |
| `awc` | soil sidecar `awc` (m³ m⁻³, topsoil) | × 100 → cm m⁻¹ (WISE `TAWC` unit) | ⚠ SoilGrids + van Genuchten, topsoil only, vs WISE profile |
| `bulk_density` | soil sidecar (g cm⁻³) | none | ✓ (SoilGrids vs WISE) |
| `vpd`, `ndvi`, `drainage_class` | — | — | ✗ left out (decision 3) |
| target `yield` | `yield_t_per_ha` (TWSO) | none (t/ha) | ✓ unit; yield level differs (water-limited potential vs reported) |
| `year` | harvest year | — | ✓ |

What matters to NanoTabICL: the model standardizes every column with the context rows' mean/std
(`model.py`, first line of `forward`), so a per-column affine change (units, the benchmark's
`standard` normalization) has no effect on its input. Non-affine differences do matter: sampling dates
(`cum_fpar_max` counts dekads), partial blocks, zeros outside the season, NaN filling, and which columns
exist in which order (the model's feature grouping takes circular shifts over the column order).

## 4. Decisions

1. **Calendar bug fixed.** CropFM's `save2zarr.py` writes the six crop-calendar values in a fixed order
   but stores seven `variable_names` (starting with `aez_id`) cut to six, so every name was one place
   to the left of its value; `locations_maize.csv` had `sos_doy`/`eos_doy` = 309/15 (= maize EOS and
   winter-cereal SOS). `extract_locations_from_zarr.py` now uses the fixed column order and reads the
   real `aez_id`; `locations_maize.csv` was repaired from the Zarr (backup:
   `locations_maize.csv.bak_shifted_calendar`). Maize calendars are now SOS 102–107 / EOS 303–309
   (CY-Bench DE: 101–115 / 297–304). The winter-cereal SOS (day 15) still needs checking before wheat.
2. **Re-simulation with the calendar.** Sowing = WorldCereal SOS ± 10 days (was a fixed day 121);
   campaign from SOS − 60 days (bare-soil water balance before sowing) to max(EOS, sowing + 240 days)
   (a trailing empty campaign keeps the water balance running after maturity). Only the 1,585
   locations whose soil/weather were cached (Belgium, Denmark, part of France).
3. **Left out:** `vpd` (meteo), the `ndvi` source and `drainage_class`; there is no general mapping
   from WOFOST for them. The benchmark runs NanoTabICL is compared in must leave them out too
   (benchmark-side feature choice, to do).
4. **Location kept**: `loc_x/y/z`, as in the published baselines.
5. **Horizon `eos`** for the model's input format (9 blocks for these calendars; `middle-of-season`
   would be a separate format with 6 blocks).
6. **Normalization and NaN**: `CropTables` applies the benchmark's normalizer (mean/std per variable
   pooled over blocks, fitted on the train years; `awc`, `bulk_density` too), drops columns with > 5 %
   NaN, then sets NaN to 0. `NanoTabICLRegressor` gets the benchmark-normalized table and sets NaN to 0.
7. **Own implementation, checked against the benchmark**: `nanotabicl/cybench_features.py` re-implements
   the steps above; `tests/test_cybench_features.py` runs the benchmark's own functions
   (`compute_crop_season_window`, `align_to_crop_season_window`, `DataFactory._create_features`,
   `_aggregate_time_series`, `_tabularize`, `feature_transform`, `_drop_sparse_tabular_columns`) on the
   same rows and checks equal names and values (`eos` and `middle-of-season`).
8. **Still open**: benchmark-side feature choice (configs or an `exclude_features` option), the yield
   quality flags for CY-Bench (`assess_yield_quality.py` or `filter_samples=null`), the benchmark
   environment (poetry).

## 5. Check against CY-Bench maize DE (`nanotabicl/cybench_check.py`)

Run with `scripts/slurm/run_cybench_check.sh` on the calendar run (72,525 matured seasons from Belgium,
Denmark and France): CY-Bench maize DE built by the benchmark's own `DataFactory`
(`feature_design` + `geometric_transform`, `exclude_features=[vpd, ndvi, drainage_class]`, no
normalization) has **153 columns, equal in names and order** to the WOFOST table
(`tests/test_cybench_features.py::test_wofost_maize_feature_names_equal_cybench_de`).
Per-feature statistics: `figures/cybench_vs_wofost_maize_DE_eos.csv`; histograms:
`figures/cybench_vs_wofost_maize_DE_eos_{meteo,fpar_soil_moisture,static}.png`.

What differs, and why:

- **fpar (real concept gap)**: WOFOST fpar is exactly 0 before emergence and after maturity. That's
  blocks 0–1 (the last 60 days before EOS; the simulated crop matures in September–early October,
  while the calendar EOS is early November) and 6–8 (before sowing): 92–100 % zeros. CY-Bench FPAR is
  30–60 % there (whole-pixel satellite FPAR: other vegetation, cover crops, stubble). Mid-season WOFOST
  is higher (≈ 90 % vs ≈ 67 %), and `cum_fpar_max` is lower because of the zeros.
- **ssm (scale)**: WOFOST ≈ 37 kg/m² (0–10 cm water) vs CY-Bench ≈ 6. A 10 cm layer at 6 kg/m² would be
  6 % volumetric, implausibly dry for Germany, so CY-Bench's `ssm` is probably not "kg/m² over 0–10 cm"
  (different layer or unit). The model standardizes every column, so a pure scale difference is
  invisible to it; the shapes are similar (narrow, unimodal). `rsm` matches well (both ≈ 200–400 kg/m²).
- **temperature / GDD**: WOFOST is warmer (tmin +2–4 °C, `gdd_sum`, `cum_gdd_max` higher). Expected:
  the WOFOST points are mostly in France (and none in Germany), and ERA5-Land vs AgERA5.
- **statics**: `awc` 21 vs 15.5 cm/m and `bulk_density` 1.24 vs 1.44 g/cm³ (SoilGrids topsoil vs WISE
  profile); `loc_*`, `sos_*`, `eos_*` differ because they are other places and only 3 calendars
  (mostly SOS 102 / EOS 309) vs DE's region-level calendars.
- prec, rad, et0, cwb, rsm: similar ranges.
