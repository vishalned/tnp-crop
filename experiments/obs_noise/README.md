# Observational noise

**Question / setup.** fPAR (sigma ~ U(0, 0.3), background ~ U(0.05, 0.2)), ssm and rsm (sigma ~ U(0, 0.3), multiplicative) get noise per value, with sigma drawn once per table; the CY-Bench features are recomputed (`nanotabicl/obs_noise.py`, `transforms.ObservationNoise`).

**Configs.** `exp_4_obs_noise` (`+transform@transforms.noise=obs_noise`)

**Figures and files.** `figures/observational_noise_seed0.png`; data plots `figures/noise_{series,features,params}.png` (`plot_noise.py`).

**Result.** NRMSE 0.207 / anomaly r 0.339 (seed 0): a small NRMSE gain, a smaller anomaly r than the log-uniform run.

Index of all experiments: `docs/experiments.md`.
