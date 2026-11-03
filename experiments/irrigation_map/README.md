# Irrigation map

**Question / setup.** Each location is irrigated (potential-production simulation, `--model-name Wofost81_PP`) or rainfed as a draw from the MIRCA-OS 2015 maize map; soil moisture of irrigated rows from the rainfed twin. Pipeline: `src/data_pipeline/irrigation/` (assign, build), `scripts/slurm/submit_wofost_run.sh pp`.

**Configs.** `exp_3_irrigation_map`

**Figures and files.** `figures/irrigation_map_seed0.png` (CY-Bench); data plots `figures/irrigation_map.png`, `irrigation_yields.png` (`plot_irrigation.py`, needs the tables).

**Result.** NRMSE 0.208 / anomaly r 0.385 (seed 0). Simulated year-to-year agreement with reported yields falls in the south, where the mix is mostly fully irrigated.

Index of all experiments: `docs/experiments.md`.
