# Sampled-worlds test run (2026-10-21)

Copied from `/projects/prjs1788/tnp-crop/wofost_worlds_test/` (`scripts/slurm/run_worlds_test.sh`, job 28423057):
datasets A, A_w, B, C, D on 7 cached locations (2 BE, 2 DK, 3 FR) x 3 worlds x 2005-2020, D with
irrigated share 0.5. Small test of the code, not final statistics. Method: `docs/wofost_worlds.md`.

- `morris_mu_star.png`, `morris_summary.csv`: Morris sensitivity (5 locations x 2008/2012/2018, rain-fed).
- `check1_reproduction.csv`: dataset A vs `wofost_calendar` (336/336 identical).
- `check2_timing.{csv,png}`: flowering/maturity day of year vs calendar EOS.
- `check3_yield.{csv,png}`: yield per dataset and country (kept worlds).
- `check4_correlation.csv`: feature-yield correlation agreement with CY-Bench.
- `check5_runtime.csv`: runs, rejections, CPU time and the estimate for 100k/300k runs.
- `world_rejections_<ds>.csv`: rejected worlds per country and reason.
