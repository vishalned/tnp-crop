# Cultivar variation (TSUM)

**Question / setup.** More variation in the prior: per location one factor f ~ U(0.8, 1.2) (seeded by `(tsum_seed, location_index)`, the same for all years and jitters of a location) multiplies both TSUM1 and TSUM2 of Grain_maize_201 (695 / 800 degree-days) in the simulation. Everything else is as in the densified OM-fix table (soil, weather, sowing dates, jitter). Pretraining: the recipe of exp_2a (log-uniform points per table), 3 seeds (rerun cleanly on 2026-11-04; the first seed-0 run is in `_replaced_runs/` of the project space).

**Simulate.** `bash scripts/slurm/submit_wofost_run.sh tsum` (-> `/projects/prjs1788/tnp-crop/wofost_eu_grid_dense_tsum/processed/`; the factor is stored in the table's `tsum_factor` metadata column). Code: `--tsum-factor-range LO HI --tsum-seed N` of `generate_wofost_dataset`, `tsum_factor` in `pcse_runner.build_parameter_provider`.

**Configs.** `exp_7_cultivar_tsum` (`paths.tables.tsum`)

**Figures and files.** `figures/cultivar_tsum.png` (CY-Bench, mean over 3 pretraining seeds); data comparison with the OM-fix table (`compare_data.py`, run in a job: two 5.6 GB tables): `figures/{usable,diversity,factor}.png`, `usable_per_country.csv`, `diversity_per_country.csv`, `summary.txt`.

**Result (3-seed mean).** Mean NRMSE 0.202 against 0.209 (log-uniform, same recipe), anomaly r 0.382 against 0.359, temporal r 0.674 against 0.611. Data: 94.0 % of seasons mature (old 97.0 %), usable locations 10,316 against 10,842; mean season length 129.6 d against 132.7 d, country-mean season-length sd 16.1 against 12.0 d.

Index of all experiments: `docs/experiments.md`.
