# Permuted-yield sanity

**Question / setup.** The yields shuffled among the rows of every training table (`transforms.PermuteYields`): real WOFOST features without the feature -> yield link.

**Configs.** `sanity_permuted_yields`

**Figures and files.** `figures/overview.png`.

**Result.** NRMSE 0.241 / anomaly r 0.020, far worse than default TabICL (0.214 / 0.296): the WOFOST gain comes from the learned relationship, not from extra training.

Index of all experiments: `docs/experiments.md`.
