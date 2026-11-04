# Experiments

One row per experiment: the question, the Hydra config(s) in `configs/experiment/`, the result and where it is. Every run is `python -m nanotabicl.train experiment=<config> seed=<n>`
(or `bash scripts/slurm/submit_experiment.sh <config>`: training -> CY-Bench benchmark -> summary). Results are collected into `results/runs.csv` (`uv run python -m analysis.collect_results`) and the
bar plots are drawn from it (`uv run python -m analysis.make_plots`, `analysis/runs.yaml` + `experiments/<id>/plots.yaml`). The index in machine-readable form: `results/experiments.csv`.

Results are CY-Bench walk-forward on the 18 European maize countries, single pass: mean over the countries of NRMSE (lower is better) / anomaly r (higher is better); mean over the pretraining seeds
where there are 3, else seed 0. Default TabICL: 0.214 / 0.296.

| id | question | configs | seeds | result | status | where |
|---|---|---|---|---|---|---|
| `exp_1` | Does continued pretraining on WOFOST tables (FR, BE, DK) beat default TabICL on CY-Bench, and is it the data or just more training (SCM control)? | exp_1, exp_1_scm | 0-2 | see docs/results.md | done | docs/results.md; experiments/early_results |
| `exp_1_eu_grid` | The same with 2,000 locations spread over the EU maize area (one per 0.1 deg cell). | exp_1_eu_grid, exp_1_eu_grid_scm | 0-2 | WOFOST 0.212 / 0.309; SCM control 0.214 / 0.277; default TabICL 0.214 / 0.296 | done | docs/scaling_report.md; experiments/eu_grid_scaling |
| `exp_1_eu_grid_dense` | Equal country share: about 700 usable points in every country (12,382 locations). | exp_1_eu_grid_dense | 0-2 | 0.212 / 0.309 | done (soil with the organic-matter bug) | docs/country_share_report.md |
| `om_fix` | Does fixing OM = SOC / 0.58 (it was SOC x 0.58) in the pedotransfer input change the result? | exp_1_eu_grid_dense_omfix | 0-2 | 0.210 / 0.327 (old soil 0.212 / 0.309) | done; the base table of the experiments below | docs/soc_fix_report.md; experiments/om_fix |
| `sampling` | A: log-uniform points per table (many small tables); B: A plus 50 % SCM tables. | exp_2a_loguniform_points, exp_2b_loguniform_points_scm | 0-2 | A 0.209 / 0.359; B 0.211 / 0.308 | done | experiments/sampling |
| `irrigation_pilot` | How different are fully irrigated (PP) and rainfed simulations (ES, IT, DE, PL), against reported yields? | - | - | data check, no training | done | experiments/irrigation_pilot |
| `irrigation_map` | Train on a table where each location is irrigated (PP) or rainfed as the MIRCA-OS 2015 maize map says. | exp_3_irrigation_map | 0 | 0.208 / 0.385 (seed-0 references: OM fix 0.211 / 0.330, log-uniform 0.211 / 0.366) | done; one seed | experiments/irrigation_map |
| `obs_noise` | Noise on fPAR, ssm and rsm in the training tables (sigma and bg per table). | exp_4_obs_noise | 0 | 0.207 / 0.339 | done; one seed | experiments/obs_noise |
| `context_years` | Does the prior help when only the last 2, 3 or 5 years are context? (DE FR HU ES; Senegal with all years.) | - | 0 | WOFOST models ahead of default TabICL in anomaly r at every length; NRMSE gain small | done; needs the benchmark branch context-years | experiments/context_years |
| `permuted_yields` | Is the WOFOST gain real, or what any extra pretraining gives? Shuffle the yields within each training table. | sanity_permuted_yields | 0 | 0.241 / 0.020: clearly worse than default TabICL, so the gain comes from the feature -> yield link | done; one seed | experiments/permuted_yields |
| `regional_subsets` | Is the gain from the whole of Europe or from one part? Pretrain on NW, CE, south or FR/BE/DK only. | sanity_regions_nw, _ce, _south, _frbedk | 0 | NW 0.211 / 0.314; CE 0.211 / 0.334; south 0.214 / 0.316; FR/BE/DK 0.207 / 0.317; all 0.212 / 0.313 | done; one seed each | experiments/regional_subsets |
| `cultivar_tsum` | Scale TSUM1 and TSUM2 by one factor ~ U(0.8, 1.2) per location in the simulation: more diversity in the prior. | exp_7_cultivar_tsum | 0 | seed 0 done | 1 seed only: lower NRMSE and higher anomaly r than log-uniform | experiments/cultivar_tsum |
| `feature_importance` | Which feature groups do default TabICL, the SCM control and the WOFOST model use? | - | - | analysis/feature_importance/results | done | analysis/feature_importance |

## Reading the old reports and scripts

The reports in `docs/` (`results.md`, `scaling_report.md`, `country_share_report.md`, `soc_fix_report.md`) describe the runs as they were made, with the command-line flags and
scripts of that time (`submit_exp1.sh`, `run_gpu.sh` with `DATA=...`, ...). Those scripts were replaced by the Hydra configs and `submit_experiment.sh` / `submit_wofost_run.sh`; the old code is
in git: `git show archive/pre-cleanup:scripts/slurm/submit_exp1.sh`. `tests/test_experiment_configs.py` checks that every experiment config equals the settings the old runs saved.

## Adding an experiment

1. If it needs new behaviour (a new points sampler, a table transform, another batch source), add a small class to `nanotabicl/samplers.py`, `transforms.py` or `sources.py` and a file in the matching
   `configs/` group; keep it off by default and add a test. If it is only a different table or recipe, no code is needed.
2. `configs/experiment/<id>.yaml`: the groups it changes (`override /sampler: ...`, `/transform@transforms.<name>: ...`), `name: <id>/<tag>_seed${seed}` and the table (`paths.tables.*`).
3. `bash scripts/slurm/submit_experiment.sh <id>`; then `analysis.collect_results`, add the run to `analysis/runs.yaml` and the plot to `experiments/<id>/plots.yaml`, `analysis.make_plots`.
4. A README in `experiments/<id>/` (question, config, result, conclusion) and a row in `results/experiments.csv` and the table above.
