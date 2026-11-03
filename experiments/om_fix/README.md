# Organic-matter fix

**Question / setup.** OM = SOC x 0.1 / 0.58 instead of x 0.58 in the pedotransfer input (`src/data_pipeline/soil/utils_soil/generate_soil_files.py`), the densified EU grid simulated again (raw soil table `soilgrids_raw`), then exp_1 on it.

**Configs.** `exp_1_eu_grid_dense_omfix` (vs `exp_1_eu_grid_dense`, the old soil)

**Figures and files.** `figures/comparison.png` (CY-Bench, 3-seed mean); data comparison (`compare_distributions.py`, run in a job: two 5.6 GB tables): `figures/{soil_static,yield,features}.png`, `per_country.csv`, `feature_shift.csv`. Report: `docs/soc_fix_report.md`.

**Result.** Mean NRMSE 0.210 against 0.212 with the old soil, anomaly r 0.327 against 0.309; better in 9 of 18 countries on NRMSE: a small gain.

Index of all experiments: `docs/experiments.md`.
