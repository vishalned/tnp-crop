# Results: NanoTabICL on CY-Bench maize

All results so far for NanoTabICL (our rewritten TabICL) on the AgML-CY-Bench-Modeling benchmark,
compared with the official TabICL and the published dashboard. Maize, forecast at end of season
(`eos`), walk-forward test.

- Sections 3–6: first experiments on DE, FR and BE ([`results_cybench_pooled.csv`](results_cybench_pooled.csv)).
- Section 7: learning-rate test ([`results_lr_test.csv`](results_lr_test.csv)).
- Section 8: **exp_1**, WOFOST vs the SCM control with 3 seeds each, on all 18 European maize countries
  ([`results_exp1_summary.csv`](results_exp1_summary.csv), [`results_exp1_per_run.csv`](results_exp1_per_run.csv)).
  This is the current main result.

## 1. Names used in this file

| Name | What it is | Code | Weights |
|---|---|---|---|
| **TabICL default** | The official `tabicl` package (`TabICLRegressor`, v2.1.1) as the benchmark runs it (`model=tabicl`). 8 estimators unless stated otherwise. | official `tabicl` | official TabICLv2 regression checkpoint |
| **NanoTabICL** | Our rewritten model: vendored NanoTabICLv2 (`nanotabicl/model.py`) with our checkpoint conversion (`checkpoint.py`), test-time ensemble (`ensemble.py`) and scikit-learn wrapper (`regressor.py`). In the benchmark this is `model=tabicl_wofost`. | tnp-crop | depends on the row |
| **NanoTabICL + official weights** | NanoTabICL with the converted official TabICLv2 checkpoint, no extra training. Should equal TabICL default; it does after the grouping fix (section 3). | tnp-crop | official |
| **NanoTabICL-WOFOST run 1** | Continued pretraining of the official weights on WOFOST tables (`continue_maize_structured/best.pt`, step 500). Trained **before** the grouping fix, so it uses the old (off-by-one) grouping, also when evaluated. | tnp-crop | official → 500 steps on WOFOST |
| **NanoTabICL-WOFOST run 2 (rerun)** | The same training repeated **after** the grouping fix, with identical settings (`wofost_fixedgroup_maize_structured/step500.pt`). | tnp-crop | official → 500 steps on WOFOST |
| **NanoTabICL-SCM (control)** | The same training as run 2, but on synthetic tables from the TabICLv2 SCM prior with exactly the WOFOST batches' shapes (`scm_matched_maize_structured/step500.pt`). | tnp-crop | official → 500 steps on SCM prior tables |
| **WOFOST LR x / SCM LR x** (section 7) | The learning-rate test: the new recipe (AdamW, warmup 10%, cosine to 0, weight decay 0.01, 3000 steps) at LR x, on WOFOST tables or on the SCM control. Evaluated at step 3000. | tnp-crop | official → 3000 steps |
| **WOFOST / SCM control (exp_1)** (section 8) | The chosen recipe (LR 1e-5, 2000 steps, otherwise as section 7), 3 seeds each; seed N of the SCM control draws the same table shapes as WOFOST seed N. Results are means over the seeds. | tnp-crop | official → 2000 steps |
| **Dashboard TabICL / TabPFN** | Published CY-Bench dashboard results: all features (including `vpd`, `ndvi`, `drainage_class`), mRMR feature selection and hyperparameter search. A loose reference, not like-for-like. | benchmark | official |

Other terms:

- **1 pass** = one forward pass on the table as it is. **8 estimators** = TabICL's test-time ensemble:
  8 views (4 column orders × 2 normalisations: none / Yeo-Johnson power, outliers clipped), predictions averaged.
  NanoTabICL's ensemble is a port of `tabicl`'s and is tested equal to `TabICLRegressor(n_estimators=8)`.
- All our runs (TabICL default included) use the **same input**: the benchmark's 153 tabular features for
  maize `eos` (16 variables × 9 blocks of 30 days + 9 static features), without `vpd`, `ndvi` and
  `drainage_class` (not in WOFOST), **no mRMR, no hyperparameter search**. So within our runs, only the
  model and weights differ.
- **Walk-forward, pooled**: for each test year, the model sees all earlier years as context. The metrics
  are computed once over all test region-years together, as the dashboard does (DE 576, FR 456, BE 49
  region-years over 5 test years).
- **Seeds**: our runs report 5 repetitions, but they are byte-identical. The TabICL wrappers use
  `random_state=42` every time, so every number below comes from **one** evaluation. The dashboard's
  TabICL/TabPFN also show std 0.

## 2. Metrics

Notation: region r, year t, true yield y_rt, predicted ŷ_rt.

| Metric | Definition | What it measures |
|---|---|---|
| **NRMSE** | RMSE over all region-years / mean true yield | Overall error (lower = better) |
| **R²** | 1 − Σ(y − ŷ)² / Σ(y − ȳ)² over all region-years | Fraction of the variance explained, which largely comes from knowing which regions are always high or low |
| **Spatial r** | Pearson r between each region's mean true and mean predicted yield | Ranking of regions |
| **Temporal r** | Average over regions per year (Ȳ_t, Ŷ_t), then r between the yearly series | National year-to-year ups and downs (only ~5 points) |
| **Anomaly r** | m_r = mean true yield of region r; a = y − m_r, â = ŷ − m_r; r over all region-years | Whether each region's good and bad years are right, with the permanent regional differences removed. The strictest skill measure. |

## 3. Implementation checks

- **Features**: `nanotabicl/cybench_features.py` re-implements the benchmark's feature design. It gives
  the same 153 names in the same order as the benchmark on CY-Bench DE, and `tests/test_cybench_features.py`
  checks values against the benchmark's own functions.
- **Feature-grouping bug (fixed)**: TabICL builds each column's token from a bundle of 3 columns. The
  official code takes columns j+1, j+2, j+4; the vendored nanotabicl takes j, j+1, j+3. That is the same
  bundles rotated by one position, which matters because the row attention is position-aware (RoPE).
  `NanoTabICL(official_feature_grouping=True)` rotates the input columns by one, and the converted
  official model now equals TabICL to ~1e-8. Before the fix the difference was ~0.5% in the predictions.
  The conversion test now randomises all weights and would catch this. Checkpoints trained before the fix
  (WOFOST run 1) keep the old grouping, consistently in training and evaluation.
- **NanoTabICL + official weights = TabICL default** after the fix (rows marked ✓ below).

## 4. CY-Bench results

NRMSE, R², temporal r, anomaly r (walk-forward, pooled). **Bold** = best of our runs in that column.

### Germany (DE, 576 region-years)

| Model | Estimators | NRMSE | R² | Temporal r | Anomaly r |
|---|---|---|---|---|---|
| Dashboard TabPFN (best on dashboard) | dashboard | 0.1457 | 0.646 | 0.946 | 0.541 |
| Dashboard TabICL | dashboard | 0.1625 | 0.560 | 0.943 | 0.505 |
| TabICL default | 8 | 0.1594 | 0.574 | 0.975 | 0.486 |
| NanoTabICL + official weights ✓ | 8 | 0.1595 | 0.573 | 0.976 | 0.486 |
| NanoTabICL-WOFOST run 1 (old grouping) | 8 | 0.1587 | 0.578 | 0.929 | 0.510 |
| NanoTabICL-WOFOST run 2 (rerun) | 8 | 0.1559 | 0.592 | 0.915 | 0.496 |
| NanoTabICL-SCM control | 8 | 0.1607 | 0.567 | 0.982 | 0.476 |
| TabICL default | 1 | 0.1712 | 0.508 | 0.972 | 0.429 |
| NanoTabICL + official weights, old grouping | 1 | 0.1721 | 0.503 | 0.981 | 0.413 |
| NanoTabICL + official weights ✓ | 1 | 0.1717 | 0.506 | 0.974 | 0.428 |
| NanoTabICL-WOFOST run 1 (old grouping) | 1 | 0.1604 | 0.568 | 0.880 | 0.484 |
| NanoTabICL-WOFOST run 2 (rerun) | 1 | **0.1539** | **0.602** | 0.980 | **0.527** |
| NanoTabICL-SCM control | 1 | 0.1715 | 0.506 | **0.992** | 0.430 |

### France (FR, 456 region-years)

| Model | Estimators | NRMSE | R² | Temporal r | Anomaly r |
|---|---|---|---|---|---|
| Dashboard TabPFN (best on dashboard) | dashboard | 0.1291 | 0.614 | 0.674 | 0.367 |
| Dashboard TabICL | dashboard | 0.1447 | 0.516 | 0.806 | 0.315 |
| TabICL default | 8 | 0.1403 | 0.545 | 0.729 | 0.292 |
| NanoTabICL + official weights ✓ | 8 | 0.1406 | 0.543 | 0.713 | 0.291 |
| NanoTabICL-WOFOST run 1 (old grouping) | 8 | **0.1382** | **0.558** | **0.845** | **0.332** |
| NanoTabICL-WOFOST run 2 (rerun) | 8 | 0.1470 | 0.500 | 0.516 | 0.243 |
| NanoTabICL-SCM control | 8 | 0.1464 | 0.504 | 0.674 | 0.286 |
| TabICL default | 1 | 0.1524 | 0.463 | 0.421 | 0.169 |
| NanoTabICL + official weights, old grouping | 1 | 0.1516 | 0.469 | 0.539 | 0.185 |
| NanoTabICL + official weights ✓ | 1 | 0.1520 | 0.466 | 0.460 | 0.177 |
| NanoTabICL-WOFOST run 1 (old grouping) | 1 | 0.1514 | 0.470 | 0.448 | 0.224 |
| NanoTabICL-WOFOST run 2 (rerun) | 1 | 0.1581 | 0.422 | −0.459 | 0.108 |
| NanoTabICL-SCM control | 1 | 0.1556 | 0.441 | 0.462 | 0.200 |

### Belgium (BE, 49 region-years: very small, the noisiest)

| Model | Estimators | NRMSE | R² | Temporal r | Anomaly r |
|---|---|---|---|---|---|
| Dashboard TabICL (best on dashboard) | dashboard | 0.1876 | 0.153 | 0.749 | 0.536 |
| Dashboard TabPFN | dashboard | 0.1879 | 0.150 | 0.845 | 0.584 |
| TabICL default | 8 | 0.1818 | 0.205 | 0.780 | 0.553 |
| NanoTabICL + official weights ✓ | 8 | 0.1801 | 0.219 | 0.786 | **0.558** |
| NanoTabICL-WOFOST run 1 (old grouping) | 8 | **0.1689** | **0.313** | 0.720 | 0.511 |
| NanoTabICL-WOFOST run 2 (rerun) | 8 | 0.1848 | 0.178 | 0.614 | 0.370 |
| NanoTabICL-SCM control | 8 | 0.2068 | −0.029 | 0.636 | 0.385 |
| TabICL default | 1 | 0.1843 | 0.183 | 0.703 | 0.419 |
| NanoTabICL + official weights, old grouping | 1 | 0.1866 | 0.162 | 0.567 | 0.373 |
| NanoTabICL + official weights ✓ | 1 | 0.1851 | 0.176 | 0.689 | 0.408 |
| NanoTabICL-WOFOST run 1 (old grouping) | 1 | 0.1792 | 0.227 | 0.841 | 0.535 |
| NanoTabICL-WOFOST run 2 (rerun) | 1 | 0.2005 | 0.033 | **0.903** | 0.315 |
| NanoTabICL-SCM control | 1 | 0.2142 | −0.104 | 0.838 | 0.304 |

### Training on WOFOST tables (our own validation and test)

WOFOST walk-forward RMSE in t/ha. Validation is 2017–18, test is 2019–20. The baseline predicts each
point's mean yield over its context years.

| Run (step 500) | Train loss (steps 451–500) | Val RMSE | Val R² | Test RMSE |
|---|---|---|---|---|
| NanoTabICL-WOFOST run 1 (old grouping) | 0.0369 | 1.362 | 0.813 | not evaluated |
| NanoTabICL-WOFOST run 2 (rerun) | 0.0383 | 1.452 | 0.787 | 1.383 |
| NanoTabICL-SCM control | (prior tables) | 1.833 | 0.661 | 1.374 |
| Baseline | | 3.069 | | 1.876 |

## 5. What the results say

1. **The rewrite is exact.** With the official weights and 8 estimators, NanoTabICL equals TabICL default
   in every country (NRMSE DE 0.1595 vs 0.1594, FR 0.1406 vs 0.1403, BE 0.1801 vs 0.1818).
2. **The ensemble matters.** Going from 1 pass to 8 estimators lowers NRMSE by ~7% for the official
   weights. Comparisons must use the same number of estimators.
3. **Germany is the one consistent WOFOST gain.** Both WOFOST runs beat the official weights. The SCM
   control (same training on synthetic tables) doesn't. This suggests the gain comes from WOFOST content,
   not from extra training. DE is also where WOFOST and CY-Bench agree most on which features go with
   yield (section 6).
4. **France and Belgium are not robust.** WOFOST run 1 beat the official weights there (FR 0.1382,
   BE 0.1689). The rerun, with identical data and settings except the grouping fix, is worse than the
   official weights (FR 0.1470, BE 0.1848). The gap between the two WOFOST runs is as large as any
   effect we measure, so the FR/BE gains of run 1 can't be attributed to WOFOST.
5. **Training on SCM prior tables never helps.** The control is equal to or worse than the official
   weights everywhere: 500 more steps of the official kind of data at lr 3e-5 cost a little.
6. **Against the dashboard**: TabICL default and NanoTabICL + official weights (8 estimators) already beat
   the dashboard's TabICL in all three countries, with fewer features and no mRMR or HPO. TabPFN on the
   dashboard is still better in DE and FR.

## 6. WOFOST vs CY-Bench distribution gap

Figures and numbers are in `docs/figures/distribution_gap/` (`nanotabicl/distribution_gap.py`).
WOFOST maize: 72,525 seasons (FR 57,354, BE 10,407, DK 4,764). CY-Bench maize: DE 3,357, FR 1,627,
BE 85 region-years.

- **Yield**: WOFOST mean 8.05 t/ha, std 3.61. CY-Bench DE 9.11 / 1.60, FR 8.76 / 1.52, BE 10.33 / 1.95.
  WOFOST yields are lower and more than twice as spread out.
- **fpar / cum_fpar**: the largest shape gap. WOFOST has zeros outside the season; satellite fpar doesn't.
- **ssm**: a large scale shift (~6×), but almost no shape gap. NanoTabICL standardises each column per
  table, so it hardly sees the shift.
- **gdd / temperature**: shifted (WOFOST warmer), mostly the France-vs-Germany geography.
- **Static features**: nearly constant in WOFOST (only 3 crop calendars).
- **Feature–yield correlations** (does a feature go with higher or lower yield in both datasets?):

  | Comparison | Same sign | Correlation of correlations |
  |---|---|---|
  | WOFOST all vs CY-Bench DE | 73% | 0.71 |
  | WOFOST all vs CY-Bench FR | 34% | −0.21 |
  | WOFOST FR vs CY-Bench FR | 42% | −0.20 |
  | WOFOST all vs CY-Bench BE | 58% | 0.19 |
  | WOFOST BE vs CY-Bench BE | 63% | 0.31 |

  In WOFOST, water (precipitation, water balance, soil moisture) strongly drives yield. In French
  reported yields almost nothing does, possibly because much French maize is irrigated, while WOFOST
  simulates rain-fed maize. This matches the CY-Bench results: a consistent gain in DE and none in FR.

## 7. Learning-rate test (DE, FR, BE)

The first runs (500 steps at LR 3e-5, checkpoint taken while the LR was still at its peak) varied a lot
between runs. The official fine-tuning recipe (`tabicl` `FinetunedTabICLRegressor`) and pretraining
scripts both end with a cosine decay to ~0; the last official pretraining stage used peak LR 2e-5
(Muon matched to AdamW) with 64 tables per step, the fine-tuning API AdamW with LR 1e-5, warmup 10%,
weight decay 0.01, clipping 1. We used that recipe with 3000 steps and tried LR 3e-6, 1e-5 and 3e-5,
on WOFOST tables and on the SCM control, and chose the LR on the benchmark's **screening** split
(validation years), not on the test years.

Screening NRMSE, 8 estimators:

| | DE | FR | BE | Mean |
|---|---|---|---|---|
| NanoTabICL + official weights | 0.1644 | 0.1499 | 0.2141 | 0.1761 |
| WOFOST LR 3e-6 | 0.1582 | 0.1600 | 0.2681 | 0.1954 |
| **WOFOST LR 1e-5** | 0.1550 | 0.1608 | 0.2413 | 0.1857 |
| WOFOST LR 3e-5 | 0.1613 | 0.1694 | 0.2513 | 0.1940 |
| **SCM LR 3e-6** | 0.1619 | 0.1493 | 0.2261 | 0.1791 |
| SCM LR 1e-5 | 0.1619 | 0.1484 | 0.2420 | 0.1841 |
| SCM LR 3e-5 | 0.1613 | 0.1479 | 0.2631 | 0.1908 |

WOFOST validation RMSE (t/ha) at steps 1000 / 2000 / 3000: LR 3e-6 1.79 / 1.61 / 1.61, LR 1e-5
1.50 / 1.42 / 1.42, LR 3e-5 1.40 / 1.33 / 1.37.

Findings:

- **LR 1e-5** was best on screening for WOFOST, averaged over the three countries. A higher LR fits
  WOFOST better but transfers worse to CY-Bench (except as a single pass in DE).
- **2000 steps are enough**: WOFOST validation stops improving after ~2000 steps. 5000 are not needed.
- **The ensemble works against WOFOST-trained models.** In DE, 8 estimators improve the official weights
  (single pass 0.175 → 0.164 screening NRMSE) and the SCM models (~0.172 → ~0.162) but hurt the WOFOST models more the
  more they are trained (LR 3e-6: 0.160 → 0.158, 1e-5: 0.152 → 0.155, 3e-5: 0.145 → 0.161). WOFOST
  tables always present the 153 features in the same order and scaling, so the model can learn
  column positions; the ensemble's column shuffles and power transforms break that. The SCM tables have
  random columns, so the SCM model doesn't learn positions.
- With this recipe, the best DE model on screening was WOFOST LR 3e-5 single pass (screening 0.1448;
  test 0.1474 / R² 0.635 / anomaly r 0.548, close to the dashboard's TabPFN 0.1457 / 0.646 / 0.541).

## 8. exp_1: WOFOST vs SCM control on all European maize countries

Setup:

- Training: official TabICLv2 weights → 2000 steps, AdamW LR 1e-5, warmup 10%, cosine to 0, weight decay
  0.01, clipping 1; evaluated checkpoint = step 2000. WOFOST tables or the SCM control; 3 seeds each.
- Benchmark: the 18 European maize countries of the benchmark's `europe.yaml` (NL, FR, DE, PL, ES, IT,
  AT, BE, BG, CZ, DK, EL, HR, HU, LT, RO, SE, SK; CY-Bench also has EE, FI, IE, LV and PT maize, which the
  benchmark leaves out). Screening then walk-forward, 1 repetition (deterministic), **single pass** for
  all settings: our checkpoints on the unchanged table, default TabICL with `n_estimators=1`.
- Reference: **default TabICL** (official `tabicl` package and weights), single pass.

Overall:

| Setting | Mean NRMSE | NRMSE weighted by region-years | Mean R² | Mean anomaly r |
|---|---|---|---|---|
| Default TabICL, single pass | 0.2142 | 0.2043 | −0.075 | 0.296 |
| SCM control (mean of 3 seeds) | 0.2141 | 0.2034 | −0.053 | 0.264 |
| **WOFOST (mean of 3 seeds)** | **0.2008** | **0.1946** | **0.099** | **0.305** |

Head to head ("clearly" = difference larger than 2× the seed spread):

| Comparison | Lower NRMSE in | Clearly better | Clearly worse |
|---|---|---|---|
| WOFOST vs default TabICL | 12 of 18 countries | 12 | 5 |
| WOFOST vs SCM control | 14 of 18 | 6 | 0 |
| SCM vs default TabICL | 11 of 18 | 5 | 4 |

Per country (walk-forward NRMSE, mean ± sd over the 3 seeds; **bold** = lowest), countries by size:

| Country | Region-years | Default TabICL | SCM control (3 seeds) | WOFOST (3 seeds) | Anomaly r: TabICL / SCM / WOFOST |
|---|---|---|---|---|---|
| DE | 576 | 0.1716 | 0.1719 ± 0.0035 | **0.1540** ± 0.0012 | 0.43 / 0.41 / 0.50 |
| IT | 468 | 0.2155 | 0.2121 ± 0.0019 | **0.2098** ± 0.0008 | 0.07 / 0.08 / 0.10 |
| FR | 456 | **0.1521** | 0.1562 ± 0.0031 | 0.1592 ± 0.0014 | 0.17 / 0.15 / 0.10 |
| ES | 233 | **0.1578** | 0.1586 ± 0.0010 | 0.1649 ± 0.0054 | 0.09 / 0.07 / 0.06 |
| EL | 200 | **0.2661** | 0.2711 ± 0.0021 | 0.2745 ± 0.0040 | 0.03 / 0.04 / 0.04 |
| RO | 195 | 0.3596 | 0.3535 ± 0.0012 | **0.3336** ± 0.0097 | 0.24 / 0.27 / 0.26 |
| HU | 100 | 0.2806 | 0.2525 ± 0.0032 | **0.1961** ± 0.0027 | 0.44 / 0.47 / 0.49 |
| PL | 80 | 0.1459 | 0.1431 ± 0.0058 | **0.1244** ± 0.0038 | 0.28 / 0.27 / 0.45 |
| CZ | 68 | 0.1817 | 0.1795 ± 0.0027 | **0.1680** ± 0.0053 | 0.45 / 0.33 / 0.20 |
| BE | 49 | **0.1843** | 0.2072 ± 0.0038 | 0.2200 ± 0.0062 | 0.42 / 0.34 / 0.30 |
| LT | 48 | 0.3355 | 0.3217 ± 0.0044 | **0.2834** ± 0.0047 | 0.20 / 0.26 / 0.21 |
| AT | 45 | 0.1233 | 0.1185 ± 0.0048 | **0.1177** ± 0.0008 | 0.30 / 0.35 / 0.36 |
| NL | 42 | **0.2067** | 0.2193 ± 0.0016 | 0.2183 ± 0.0036 | 0.03 / -0.04 / -0.01 |
| SK | 40 | 0.2576 | 0.2507 ± 0.0008 | **0.2097** ± 0.0045 | 0.41 / 0.36 / 0.57 |
| BG | 30 | 0.2029 | 0.2017 ± 0.0033 | **0.1831** ± 0.0039 | 0.58 / 0.55 / 0.47 |
| DK | 10 | 0.2601 | 0.2544 ± 0.0006 | **0.2372** ± 0.0017 | -0.24 / -0.25 / -0.29 |
| HR | 10 | 0.1812 | 0.1788 ± 0.0060 | **0.1687** ± 0.0059 | 0.67 / 0.72 / 0.87 |
| SE | 5 | **0.1725** | 0.2030 ± 0.0036 | 0.1912 ± 0.0060 | 0.76 / 0.40 / 0.81 |

Findings:

1. **The WOFOST gain is real and specific to WOFOST.** WOFOST lowers NRMSE by ~5–6% on average; the SCM
   control with identical training changes almost nothing (0.2141 vs 0.2142). So the gain comes from the
   crop-model data, not from extra training or from adapting to the table shape.
2. **The seeds agree.** WOFOST's seed spread is 0.001–0.01 NRMSE, much smaller than its gains (DE 0.154 vs
   0.172, HU 0.196 vs 0.281, PL 0.124 vs 0.146, SK 0.210 vs 0.258, LT 0.283 vs 0.336). The run-to-run
   variation of the first runs (sections 4–5) came from the old recipe.
3. **Geographic pattern.** WOFOST helps in central and eastern Europe (DE, PL, CZ, SK, HU, RO, BG, LT,
   AT, HR) and in IT and DK; it hurts in FR, ES, EL, BE, NL and SE. WOFOST was simulated only at FR, BE and
   DK locations, yet it helps most elsewhere. This fits section 6: WOFOST simulates rain-fed,
   water-limited maize, which matches the mostly rain-fed east better than irrigated maize (FR, ES, EL)
   or maritime BE and NL.
4. The smallest countries (SE 5, DK 10, HR 10 region-years) are noisy.
5. This compares single passes. With 8 estimators the official model gains ~7% while WOFOST-trained models
   lose (section 7), so the gap may shrink in the ensemble comparison (not run yet).

## 9. Caveats and open points

- Sections 4–7: every row is one training run and one evaluation (the 5 benchmark repetitions are
  identical). The two old-recipe WOFOST runs differed by ±0.005–0.01 NRMSE in FR/BE. exp_1 (section 8)
  uses 3 seeds with the new recipe; its seed spread is much smaller.
- BE has 49 test region-years; single region-years move its scores a lot.
- WOFOST run 1 and run 2 differ in the grouping (and in GPU non-determinism). Their batches are identical
  (same seed, same table shapes at every step).
- On WOFOST, run 1 also had a lower val RMSE than run 2 (1.36 vs 1.45). The SCM control and run 2 have
  the same WOFOST test RMSE despite a large val gap; not understood yet.

## 10. Where things are

- Checkpoints: `/projects/prjs1788/tnp-crop/nanotabicl/{continue_maize_structured,wofost_fixedgroup_maize_structured,scm_matched_maize_structured}/`
- Benchmark outputs: `/projects/prjs1788/tnp-crop/cybench_output/tabicl_wofost/<run_folder>/` (folder names in the CSV)
- LR test: training `/projects/prjs1788/tnp-crop/nanotabicl/lr_test/`, benchmark `/projects/prjs1788/tnp-crop/cybench_output/lr_test/`
  (submitted with `scripts/slurm/submit_lr_test.sh`)
- exp_1: training `/projects/prjs1788/tnp-crop/nanotabicl/exp_1/`, benchmark `/projects/prjs1788/tnp-crop/cybench_output/exp_1/`
  (`summary.txt`; submitted with `scripts/slurm/submit_exp1.sh`, which uses the benchmark's
  `cybench/runs/tabicl_wofost/run_europe.sh` and `summarize_exp.py`)
- Dashboard reference: `/projects/prjs1788/tnp-crop/dashboard/dashboard_maize_eos_DE_FR_BE.csv`
- Benchmark integration: `../AgML-CY-Bench-Modeling`, local branch `tabicl-wofost` (`cybench/runs/tabicl_wofost/`)
