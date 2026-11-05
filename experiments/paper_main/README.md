# Main comparison (paper)

**Question / setup.** The chain of changes on the densified OM-fix table, each adding one thing, 3 pretraining seeds each, CY-Bench walk-forward on the 18 European maize countries (single pass): default TabICL; SCM control (prior tables with the shapes the log-uniform sampler draws, the one control for all); WOFOST rainfed with uniform points per table; + log-uniform points; + cultivar variation (TSUM1 and TSUM2 x U(0.8, 1.2) per location); + irrigation map (MIRCA-OS 2015: irrigated locations take the PP simulation with the same TSUM factor).

**Configs.** `exp_1_eu_grid_dense_omfix`, `exp_2a_loguniform_points`, `exp_2c_scm_control_loguniform`, `exp_7_cultivar_tsum`, `exp_8_irrigation_map_tsum`. Table of the last one: `bash scripts/slurm/submit_wofost_run.sh pp_tsum`, then `build_irrigation_table` with `--rainfed` the tsum and `--irrigated` the pp_tsum table (-> `wofost_eu_grid_dense_irrmap_tsum`).

**Figures and files.** `figures/main_comparison.png` (NRMSE, R², anomaly r, temporal r; per country and the mean), `figures/main_comparison.csv` (means).

**Result (mean over countries and 3 seeds; NRMSE / R² / anomaly r / temporal r).** Default TabICL 0.214 / -0.075 / 0.296 / 0.609; SCM control 0.215 / -0.090 / 0.251 / 0.542; WOFOST rainfed 0.210 / -0.037 / 0.327 / 0.636; + log-uniform 0.209 / -0.012 / 0.359 / 0.611; + TSUM 0.202 / 0.066 / 0.382 / 0.674; + irrigation map 0.200 / 0.079 / 0.392 / 0.702. Each step helps; the SCM control is no better than default TabICL.

Index of all experiments: `docs/experiments.md`.
