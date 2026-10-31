# Organic-matter fix (OM = SOC / 0.58): data, training and CY-Bench results

Branch `soc_fix_rerun`. Densified EU grid only (12,382 maize locations, 2005-2020, 3 sowing jitters = 594,336 seasons).
Nothing else was rerun: not default TabICL (its results are copied from `exp_1_eu_grid_dense`), not exp_1, the 3-country
runs, the EU grid run or the SCM control.

## 1. The bug and the fix
`calculate_van_genuchten` (`src/data_pipeline/soil/utils_soil/generate_soil_files.py`) computed organic matter as
`SOC x 0.1 x 0.58`. Carbon is ~58 % of organic matter, so it must be `SOC x 0.1 / 0.58` (SOC in g/kg -> %). Organic matter was
~3x too low (1 / 0.58^2 = 2.97; checked: FSOMI in the new soil files is exactly 2.973x the old one in every layer).
The Wosten pedotransfer functions take organic matter as an input, so every soil water-retention curve, `awc`, the soil water
in the simulation and the simulated yields changed.

## 2. What was done
1. **Soil.** The raw SoilGrids values had never been saved (only the derived soil YAMLs, which were built with the bug and do not
   contain clay or silt), so SoilGrids was downloaded again from Earth Engine for all 12,382 locations (job 28767563, 42 min; 63
   locations returned no SoilGrids data, as before). This time the raw values are kept in one table:
   `/projects/prjs1788/tnp-crop/data/raw/soilgrids_raw/soilgrids_raw_maize_eu27_dense700_s42.parquet` (+ `.failed.csv`).
   The soil files are rederived from it offline. New default soil cache folder `soilgrids_gee_omfix` (the old `soilgrids_gee`
   holds buggy files and is not touched).
2. **Weather: nothing downloaded.** ERA5-Land weather and elevation come from the existing caches; `TNP_NO_GEE_WEATHER=1` turns any
   download into an error. Same settings as the old run (2005-2020, 3 jitters of +-10 days, seed 42), so every season is the same
   episode as before; only the soil differs. The weather features of the two tables are identical (checked, `feature_shift.csv`).
3. **Simulation** (array 28767565, 7 chunks of 1,800 locations, 4.1-5.3 h each) with the season files on the node-local disk, so the
   project's file limit is not touched: per chunk only 5 files are written to the project space (manifest, processed parquet + columns
   json, `maize.tar.zst`, `soil.tar.zst`; archive counts checked against the folders). Merged into
   `/projects/prjs1788/tnp-crop/wofost_eu_grid_dense_omfix/processed/wofost_maize_daily.parquet`.
4. **Training** `exp_1_eu_grid_dense_omfix`: WOFOST only, 3 seeds, same recipe as the old dense run (continued pretraining of the official
   TabICLv2 weights, LR 1e-5, 2000 steps, last.pt), then the CY-Bench evaluation on the 18 European countries (walk-forward, single pass).
5. **Comparison** `analysis/soc_fix/` (`compare_distributions.py`, `compare_results.py`).

Files written to the project space by this rerun: `wofost_eu_grid_dense_omfix/` 38 files / 48 GB, `nanotabicl/exp_1_eu_grid_dense_omfix/`
about 20 files / 0.7 GB, `cybench_output/exp_1_eu_grid_dense_omfix/` about 1,600 files / 7 MB, raw soil table 2 files.

## 3. What changed in the data
Seasons are matched one-to-one with the old table (same location, sowing year, jitter). Figures: `analysis/soc_fix/figures/`
(`soil_static.png`, `yield.png`, `features.png`).

- `awc` (available water capacity of the top layer) fell by 16 % on average (location-weighted), bulk density is unchanged, and the simulated
  yield fell by 18 % overall (season-weighted; 31-40 % in Bulgaria, Romania, Italy, Portugal and Spain, 4-5 % in Denmark and Poland).
- Only features that depend on the soil moved: `awc`, `ssm`, `rsm` (a little: +0.2 and +0.01 old standard deviations), `fpar`, `cum_fpar`.
  All weather features, the calendar and location features and `bulk_density` are identical.
- **Why awc fell although organic matter went up:** in the Wosten curves more organic matter makes the retention curve flatter (n falls,
  1.17 -> 1.13 for a median topsoil); the water held at wilting point (pF 4.2) rises more than at field capacity (pF 2.0), so the plant-available
  water (FC - WP) drops (0.203 -> 0.171). The new organic matter reaches up to 46 % in a few organic topsoils, outside what the Wosten functions were fitted on.
- Seasons: 579,360 succeed (old 581,040). The old 13,296 failures (no ERA5 data over sea, no SoilGrids data) are still failures. **1,680 seasons
  (35 locations) newly fail** with `AFGEN ... not strictly ascending`: with the higher organic matter their tabulated water-retention
  curve is no longer strictly monotonic (0.3 % of the seasons).

| country | locations | awc old | awc new | awc % | yield old (t/ha) | yield new (t/ha) | yield % |
|---|---|---|---|---|---|---|---|
| Austria | 701 | 20.79 | 17.19 | -17.31 | 10.11 | 8.4 | -16.89 |
| Belgium | 693 | 21.5 | 18.29 | -14.95 | 9.88 | 8.28 | -16.19 |
| Bulgaria | 713 | 17.95 | 14.85 | -17.23 | 3.83 | 2.32 | -39.56 |
| Croatia | 710 | 18.92 | 16.52 | -12.7 | 7.25 | 5.9 | -18.66 |
| Czechia | 893 | 21.56 | 17.82 | -17.35 | 10.69 | 9.34 | -12.65 |
| Denmark | 534 | 28.77 | 21.85 | -24.04 | 10.76 | 10.31 | -4.22 |
| France | 697 | 20.88 | 17.27 | -17.29 | 7.74 | 5.81 | -24.97 |
| Germany | 791 | 21.58 | 18.44 | -14.56 | 10.59 | 9.12 | -13.82 |
| Greece | 703 | 20.8 | 17.81 | -14.36 | 3.99 | 3.17 | -20.61 |
| Hungary | 698 | 19.26 | 16.58 | -13.89 | 7.96 | 6.77 | -14.93 |
| Italy | 697 | 19.12 | 15.68 | -18.02 | 5.84 | 3.91 | -33.09 |
| Lithuania | 261 | 26.77 | 19.01 | -28.98 | 11.05 | 9.39 | -15.07 |
| Netherlands | 292 | 23.84 | 19.83 | -16.8 | 11.97 | 10.25 | -14.3 |
| Poland | 723 | 22.16 | 20.06 | -9.49 | 11.03 | 10.46 | -5.16 |
| Portugal | 392 | 19.52 | 17.99 | -7.87 | 1.6 | 1.09 | -31.96 |
| Romania | 706 | 18.07 | 14.7 | -18.63 | 5.25 | 3.36 | -35.92 |
| Slovakia | 810 | 20.25 | 16.8 | -17.02 | 9.05 | 7.55 | -16.64 |
| Slovenia | 350 | 19.79 | 16.45 | -16.89 | 9.36 | 7.22 | -22.86 |
| Spain | 696 | 19.94 | 17.26 | -13.41 | 1.36 | 0.94 | -30.69 |

(countries with at least 1,000 seasons; `analysis/soc_fix/per_country.csv` has all.)

## 4. Training
Validation (WOFOST years 2017-2018) and test (2019-2020) are on each run's own table, so the numbers are not comparable between the two runs
(yields differ). Mean of 3 seeds: old dense table val RMSE 1.267 (R2 0.864), test RMSE 1.315 (R2 0.888, point-mean baseline RMSE 2.100);
OM-fix table val RMSE 1.438 (R2 0.817), test RMSE 1.282 (R2 0.892, baseline 1.957). The model still learns the simulator well.

## 5. CY-Bench results (18 European maize countries, walk-forward, single pass, mean over 3 pretraining seeds)
| run | NRMSE (mean over countries) | NRMSE (weighted by region-years) | anomaly r (mean) | anomaly r (weighted) |
|---|---|---|---|---|
| TabICL default | 0.2142 | 0.2043 | 0.2962 | 0.2318 |
| WOFOST, exp_1 (original points) | 0.2008 | 0.1946 | 0.3054 | 0.2407 |
| SCM control, exp_1 | 0.2141 | 0.2034 | 0.2644 | 0.2219 |
| WOFOST, EU grid | 0.2116 | 0.2016 | 0.3092 | 0.2552 |
| SCM control, EU grid | 0.2141 | 0.2031 | 0.2774 | 0.2292 |
| WOFOST, densified EU grid (old soil) | 0.2116 | 0.2003 | 0.3091 | 0.2823 |
| WOFOST, densified EU grid, OM fix | 0.2103 | 0.1967 | 0.327 | 0.2995 |

Per country, densified table old soil vs OM fix (seed sd of NRMSE is 0.001-0.009 per country):

| country | NRMSE old soil | NRMSE OM fix | diff | anomaly r old | anomaly r fix | diff r |
|---|---|---|---|---|---|---|
| AT | 0.1242 | 0.1253 | 0.0011 | 0.4051 | 0.3658 | -0.0393 |
| BE | 0.1789 | 0.1609 | -0.018 | 0.4949 | 0.5745 | 0.0796 |
| BG | 0.1852 | 0.1978 | 0.0126 | 0.5499 | 0.6163 | 0.0664 |
| CZ | 0.1839 | 0.1663 | -0.0176 | 0.2445 | 0.5251 | 0.2806 |
| DE | 0.1564 | 0.1537 | -0.0027 | 0.5088 | 0.4991 | -0.0097 |
| DK | 0.2587 | 0.2672 | 0.0085 | -0.1967 | -0.2396 | -0.0429 |
| EL | 0.2722 | 0.2746 | 0.0024 | 0.0631 | 0.0563 | -0.0068 |
| ES | 0.1813 | 0.1706 | -0.0107 | 0.0838 | 0.0695 | -0.0143 |
| FR | 0.1429 | 0.1328 | -0.0101 | 0.2831 | 0.3827 | 0.0996 |
| HR | 0.1604 | 0.181 | 0.0206 | 0.8496 | 0.8278 | -0.0218 |
| HU | 0.2309 | 0.2795 | 0.0486 | 0.5324 | 0.475 | -0.0574 |
| IT | 0.214 | 0.2141 | 0.0001 | 0.1389 | 0.1018 | -0.0371 |
| LT | 0.3417 | 0.2859 | -0.0558 | 0.05 | 0.1582 | 0.1082 |
| NL | 0.2165 | 0.2088 | -0.0077 | -0.0438 | 0.0195 | 0.0633 |
| PL | 0.1454 | 0.1425 | -0.0029 | 0.3755 | 0.3737 | -0.0018 |
| RO | 0.366 | 0.3574 | -0.0086 | 0.2225 | 0.2509 | 0.0284 |
| SE | 0.1875 | 0.2013 | 0.0138 | 0.5518 | 0.4113 | -0.1405 |
| SK | 0.2633 | 0.2651 | 0.0018 | 0.4497 | 0.4176 | -0.0321 |

Summary: the fix changes the CY-Bench results only a little. NRMSE falls by 0.0014 on average (0.2116 -> 0.2103) and by 0.0036 weighted by
region-years (0.2003 -> 0.1967); anomaly r rises from 0.309 to 0.327 (weighted 0.282 -> 0.300). It is better in 9 of 18 countries and worse in 9;
9 countries differ by more than twice the seed sd. The gains come from LT (-0.056), BE and CZ (-0.018), ES and FR (-0.010) and the loss is
mainly HU (+0.049), BG and HR. Against the other runs: the OM-fix densified run is the best of the scaled-up tables (NRMSE 0.2103 / weighted 0.1967, vs 0.2116 / 0.2003 for the
old-soil densified run and 0.2116 / 0.2016 for the EU grid) and slightly better than default TabICL (0.2142 / 0.2043), but the original exp_1 points
(0.2008 / 0.1946) are still the best, so the fix does not change the earlier conclusion that more WOFOST locations did not help.
Figure: `analysis/soc_fix/figures/comparison.png`.

## 6. Caveats
- **Seed 1 needed a rerun.** It crashed twice at step ~300 with `CUDA error: invalid argument`: it drew a 4096-row table and with
  `MICRO_BATCH=2` (the default in the working-tree `run_gpu.sh`; the old dense run used 1) row attention exceeds CUDA's 65,535 grid limit
  (the job still reported COMPLETED). Seed 1 was rerun with `MICRO_BATCH=1`; seeds 0 and 2 used 2. The micro-batch size does not change
  the update (there is a test), so the seeds are comparable.
- The NRMSE differences are small compared with the seed spread in many countries; read the pattern, not single countries.
- Default TabICL in these tables is the exp_1 reference (official `tabicl` package), copied, not rerun.

## 7. Jobs
soil 28767563; simulation 28767565 (0-6); merge 28767566; training 28767569 / 28788520 (seed 1, rerun) / 28767573; CY-Bench 28767570 / 28788521 / 28767574;
distribution comparison 28767577. Rerun everything with `bash scripts/slurm/submit_eu_grid_dense_omfix.sh` (`TEST=1` for a 6-location check).
