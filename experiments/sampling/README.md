# Points per table

**Question / setup.** A: the number of points per training table log-uniform from 1 up to the cap (default: uniform from 10), so many small tables. B: A, and half of the batches are SCM tables with shapes from the same sampler.

**Configs.** `exp_2a_loguniform_points`, `exp_2b_loguniform_points_scm` (`sampler=loguniform`, `source=mix_matched`)

**Figures and files.** `figures/comparison.png` (without B), `figures/comparison_with_50p_scm.png`.

**Result.** A improves anomaly r (0.359 against 0.327) at the same NRMSE; adding SCM tables (B) does not help.

Index of all experiments: `docs/experiments.md`.
