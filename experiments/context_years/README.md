# Short context

**Question / setup.** The context of every forecast year is cut to the last 2, 3 or 5 years (all regions kept) for default TabICL, the SCM control and two WOFOST models on DE, FR, HU, ES; Senegal with all years as a side check. The cap is an env variable `TNP_MAX_CONTEXT_YEARS` in the benchmark's `TabularFoundationModel.fit` (local branch `context-years` of AgML-CY-Bench-Modeling, not pushed); `scripts/slurm/submit_context_years.sh`.

**Configs.** -

**Figures and files.** `figures/context_years.png`, `figures/senegal.png`, `context_years.csv` (`plot_context_years.py`).

**Result.** The WOFOST models are ahead in anomaly r at every length (2 years: 0.25 against 0.14); the NRMSE gain is small and not uniform; none helps in Senegal.

Index of all experiments: `docs/experiments.md`.
