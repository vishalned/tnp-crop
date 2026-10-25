# Scaling report: from 3 countries (exp_1) to the EU grid (exp_1_eu_grid)

Branch `scaling_main`. This report describes everything that changed between the first WOFOST training
data (FR/BE/DK, used in exp_1, `docs/results.md` §8) and the EU grid data, what that did to the
benchmark results, and the checks run to find out why the WOFOST gain shrank.

**Short answer.** The gain over default TabICL fell from 6.3% (exp_1) to 1.2% (EU grid). The pipeline
code is not the cause: re-running old locations with the current code reproduces them exactly. The
drop splits into three roughly equal parts:

1. **Fewer points per country, so smaller training tables** (+0.0035 NRMSE);
2. **Which FR/BE/DK locations were simulated** (old: a clustered strip of south-west France; new: one
   point per weather cell over all of France) (+0.0042);
3. **Adding the 15 other countries** (+0.0031).

Each step is about 2–4× the seed-to-seed spread. The usual suspects are not the cause:
- no failure or date anomalies in Romania or Hungary;
- feature–yield agreement with CY-Bench did not drop for RO, HU, SK or LT.

The clearest simulation problems are elsewhere:
- rain-fed Spain and Portugal (25–33% near-zero yields);
- a crop that matures 2–3 months before the calendar's end of season everywhere south of about 47°N.

---

## 1. What changed, step by step

### 1.1 Location sampling

| | Old (exp_1) | New (EU grid) |
|---|---|---|
| Source | CropFM Zarr `europe_data_1200_merged_cleaned2_rechunked.zarr`, points with WorldCereal crop mask = maize | same |
| Countries | `nw_europe` group (NL, DE, FR, BE, UK, IE, LU, DK) | EU-27 (`eu27` group) |
| Sampling | 10,000 points drawn uniformly at random (seed 42) → `locations_maize.csv`, sorted by country, then Zarr index | one random point per 0.1° cell (cells centred on multiples of 0.1°, like the ERA5-Land grid), then 2,000 cells uniformly at random (seed 42) → `locations_maize_eu27_cell01_n2000_s42.csv` (`grid_sample_locations.py`) |
| Simulated | only the **first 1,585 rows** (the first batch run stopped there): Belgium 224 (all), Denmark 159 (all), France 1,202, the first by Zarr index | all 2,000 |
| Spatial spread | France = a strip of south-west France (lat 43.1–45.6); 1,585 points in only **487** ERA5-Land cells (up to 18 points with identical weather) | 2,000 points in 2,000 cells (one per weather pixel), 21 countries, lat 36.6–57.3 |
| Locations with all seasons OK | 1,563 (22 failed entirely) | 1,966 (34 failed entirely: 16 coastal points outside ERA5-Land's land mask, 18 with no SoilGrids data) |

The old French sample was not random over France. The CSV is sorted by Zarr index, which follows space,
and only its first 1,202 French rows were simulated.

### 1.2 Downloads (soil, weather)

The sources and code are identical: ERA5-Land `DAILY_AGGR` and SoilGrids 250 m, both via Earth Engine, one
cache file per coordinate. Only the Earth Engine Cloud project changed, from `cropfm` (old account) to
`helical-kayak-399407`, which is now configurable through `$EE_PROJECT`.

### 1.3 Simulation (WOFOST)

Identical code and settings. This was verified by re-simulating 20 old locations with the current
code: 960/960 seasons have the same sowing date and the same yield (max difference 0.000000 kg/ha).
The settings:
- `Wofost81_WLP_MLWB`, rain-fed (water-limited), variety `Grain_maize_201`;
- sowing years 2005–2020;
- 3 sowing dates per year, within ±10 days of the WorldCereal SOS, pre-drawn with seed 42;
- campaign from SOS − 60 days to at least EOS.

The crop calendar is WorldCereal's per agro-ecological zone:
- **old:** all three countries share one zone, SOS day 102 and EOS day 309;
- **new:** 6 combinations. 1,074 locations have SOS 107 / EOS 304, 750 have 102 / 309, and the rest
  are in four other zones.

### 1.4 Processing into the training table

Identical code: `process_wofost_dataset.py`, `--align calendar`, with daily series over [SOS − 60, EOS]
in CY-Bench units.

One bug was found and fixed. The manifest was read with pandas' default float parser, which can be one
ulp off. Weather cache files are named by their coordinates, so 37 successful EU grid locations didn't
find their weather and were skipped. The manifest is now read with `float_precision="round_trip"`.
The old table was not affected: all 1,563 successful locations are in it.

The re-processed old locations match the old table in all 2,695 numeric columns.

The log's "rows with some missing weather" (59k rows in the new table) is not missing data. It is the
NaN padding after the end of shorter seasons; no weather value is missing inside any window.

| | Old | New |
|---|---|---|
| Processed seasons | 75,024 | 94,368 |
| Rows used for training (matured seasons) | 72,525 | 92,788 |
| Training regions (≥ 10 points) | 3: France 1,195, Belgium 217, Denmark 151 | 18: FR 421, DE 311, PL 303, RO 214, HU 129, IT 116, BG 85, ES 79, CZ 61, SK 45, HR 37, AT 36, EL 31, DK 23, BE 22, LT 20, NL 11, PT 11. Dropped: LV 4, SI 5, SE 2 |
| Features | 153 CY-Bench features (`eos`) | same |

### 1.5 How training tables are drawn

The code is identical (`CropTables`, structured episodes). Every step draws a batch of 8 same-shape
tables:
1. pick a lead country **uniformly over the training countries**;
2. draw a number of points between 10 and min(the lead country's size, 4096 / rows per point);
3. give each point its target year plus 5–11 earlier years;
4. fill the other tables of the batch from countries with at least that many points.

The data changed the table sizes a lot (simulated from the sampling rule):

| Points per training table | p10 | median | mean | p90 |
|---|---|---|---|---|
| Old (3 countries: 1,195 / 217 / 151 points) | 32 | 120 | 149 | 321 |
| New (18 countries, 15 of them with < 130 points) | 11 | 27 | 59 | 164 |

The training step count is fixed (2,000), so the new run also saw about 2.5× fewer rows in total. The
countries are also equally likely as the lead, so Portugal (11 points) leads as often as France (421).

### 1.6 Training and evaluation (unchanged)

**Training:**
- continued pretraining from the official TabICLv2 weights;
- 2,000 steps, AdamW, LR 1e-5, warmup 10%, cosine to 0, weight decay 0.01, gradient clipping 1;
- batch of 8 tables (micro-batch 2, activation checkpointing);
- train years 2005–2016; evaluated checkpoint = `last.pt`;
- 3 seeds.

**SCM control:**
- the same recipe on SCM-prior tables with the WOFOST batches' shapes;
- seed N matches WOFOST seed N;
- it is re-drawn for each WOFOST table, so it is shape-matched to that table.

**Benchmark:**
- the 18 European maize countries of the benchmark's `europe.yaml`;
- screening, then walk-forward, single pass;
- reference = default TabICL, single pass, which is deterministic and gives the same numbers in every run;
- driven by `scripts/slurm/submit_exp1.sh`, now with `EXP` and `TABLE`.

---

## 2. Results

### 2.1 exp_1 on the new table

| Setting | Mean NRMSE | Weighted NRMSE | Anomaly r |
|---|---|---|---|
| Default TabICL | 0.2142 | 0.2043 | 0.296 |
| SCM control, old shapes / new shapes | 0.2141 / 0.2141 | 0.2034 / 0.2031 | 0.264 / 0.277 |
| WOFOST, old table (exp_1) | **0.2008** | **0.1946** | 0.305 |
| WOFOST, EU grid table | 0.2116 | 0.2016 | 0.309 |

On the EU grid table, WOFOST beats default TabICL in 9 of 18 countries (clearly better in 6, clearly worse
in 6); in exp_1 it was 12 of 18 (12 / 5). Figures: `analysis/scaling/figures/results_*.png`.

### 2.2 The decisive test, plus a size-matched control

The FR/BE/DK subset of the new table has 466 locations against 1,563 in the old one, which changes the
table sizes as well. So the old table was also cut to the same 421 / 22 / 23 locations (random, seed
42). Both subsets have exactly 22,368 rows.

| Training table | WOFOST NRMSE (± sd over 3 seeds) | Gain vs TabICL | Step |
|---|---|---|---|
| old, FR/BE/DK, 1,563 locations (exp_1) | 0.2008 ± 0.0009 | 6.3% | |
| old, subsampled to 466 locations | 0.2043 ± 0.0003 | 4.6% | +0.0035 = fewer points / smaller tables |
| **new, FR/BE/DK only, 466 locations** | **0.2085 ± 0.0015** | 2.7% | +0.0042 = which FR/BE/DK locations |
| new, 18 countries, 1,966 locations | 0.2116 ± 0.0013 | 1.2% | +0.0031 = the added countries |

The SCM control is 0.2140–0.2141 in all four, so the differences come from the WOFOST data. Figure:
`analysis/scaling/figures/decisive_test.png`; per-country values in
`analysis/scaling/decisive_test_per_country.csv`.

**Reading.** The gain does not fully come back with the new FR/BE/DK cells. By the original criterion
that points to (b), "something in the new pipeline differs". But the code is identical (§1.3, §1.4). What
differs is the set of locations and its size:
- About a third of the drop is table size: the same old data with 466 instead of 1,563 locations
  loses 0.0035.
- About 40% is location choice within FR/BE/DK, at the same size. Old France was a dense strip of
  the south-west, several points per weather cell; new France is spread over the whole country, one
  point per cell.
- About a third comes from adding the other 15 countries, (a).

Per country, adding the other countries hurt Romania most (0.351 → 0.378) and Slovakia (0.220 → 0.251).
It helped Czechia (0.167 → 0.157), Bulgaria (0.202 → 0.172) and Poland (0.147 → 0.134). France is
better with the new data in both new runs (0.148–0.150, against 0.159 in exp_1).

### 2.3 Simulation sanity per country

Script: `src/data_checks/season_sanity.py`. Outputs: `analysis/scaling/sanity/`, and the figure
`sanity_per_country.png`.

| Country | Median yield (t/ha) | Seasons < 0.5 t/ha | Matured by EOS | Flowering − SOS (days) | Maturity − EOS (days, median) |
|---|---|---|---|---|---|
| France (old / new) | 7.3 / 7.8 | 0.1% / 0% | 100% / 100% | 78 / 86 | −78 / −60 |
| Belgium (old / new) | 10.2 / 10.5 | 0 / 0 | 100% / 100% | 93 / 93 | −44 / −43 |
| Denmark (old / new) | 10.4 / 9.5 | 0 / 0 | 51% / 42% | 104 / 104 | −14 / −18 |
| Germany | 11.2 | 0 | 97% | 89 | −45 |
| Poland | 11.7 | 0 | 99% | 85 | −50 |
| Romania | 4.8 | 0.1% | 99% | 69 | −80 |
| Hungary | 8.1 | 0 | 100% | 70 | −79 |
| Italy | 5.5 | 2.9% | 100% | 70 | −87 |
| Spain | **0.8** | **33%** | 100% | 69 | −86 |
| Portugal | **0.8** | **25%** | 100% | 63 | −90 |
| Greece | 2.7 | 3.2% | 100% | 64 | −92 |
| Lithuania | 11.4 | 0 | 80% | 89 | −39 |
| Slovakia | 9.1 | 0 | 90% | 77 | −69 |

- **Romania and Hungary look normal:** no failures, no near-zero yields, plausible dates. Their losses
  are not a simulation breakdown. Romania's simulated yields are higher and more variable than reported
  ones (5.4 ± 2.9 against 4.1 ± 1.7 t/ha).
- **Spain and Portugal are broken as rain-fed simulations** (median 0.8 t/ha against 7–9 reported). Real
  maize there is irrigated.
- **The season is far too short in the south.** `Grain_maize_201` is the earliest variety, and it
  matures about 80–90 days before the calendar's EOS (late October) in RO, HU, IT, ES, PT, EL, BG and HR.
  CY-Bench's feature window runs to EOS, so for the last ~3 months the simulated crop is dead (fpar ≈ 0,
  no transpiration) while the real crop is still growing. This was already true for old south-west
  France (−78 days), but it now covers most of the data.
- **The calendar is coarse but not buggy.** It has only 6 SOS/EOS combinations over the EU, the
  WorldCereal zones; sowing dates match SOS (median offset 0).
- Denmark matures by EOS in only 42–51% of seasons (old and new alike); Lithuania 80%, Slovakia 90%.

### 2.4 Feature–yield agreement with CY-Bench, old vs new

The tool is `nanotabicl.distribution_gap`, run on the old and the new table against all 18 CY-Bench
countries. The number is the correlation, over the 153 features, between each feature's correlation
with yield in WOFOST and in CY-Bench (`analysis/scaling/feature_yield_agreement.csv`).

| Country | Old table (all) | New table (all) | New table, same country |
|---|---|---|---|
| RO | 0.44 | 0.40 | 0.53 |
| HU | 0.69 | 0.70 | 0.81 |
| SK | 0.23 | 0.26 | 0.30 |
| LT | −0.50 | −0.52 | −0.40 |
| DE | 0.71 | 0.70 | 0.84 |
| PL | 0.31 | 0.27 | 0.47 |
| ES | −0.80 | −0.78 | −0.86 |
| FR | −0.22 | −0.20 | −0.18 |

Agreement did **not** drop for RO, HU, SK or LT. Each country's own simulated data agrees better with
CY-Bench than the pooled table does. So the losses in RO and SK when the other countries are added are
not explained by worse feature–yield relations in their simulated data. Spain (−0.86) and Lithuania
(−0.40) have relations of the opposite sign to the real data, in both old and new tables.

---

## 3. What this means and possible next steps

- **The pipeline is sound.** The code reproduces the old run exactly, and the per-country simulations
  have no failure or calendar bugs.
- **The data mix matters more than the volume.**
  - More, smaller, equally weighted countries give smaller training tables, and that costs accuracy.
  - The old run's dense, weather-duplicated south-west French sample did better than an even spread
    over France.
  - Some added countries help (CZ, BG, PL) and some hurt (RO, SK).
- Options, each testable with the existing tables and no new downloads:
  1. **Sample the lead country in proportion to its size** (or its number of points) instead of
     uniformly, so that table sizes come back up.
  2. **Leave out the broken rain-fed countries** (ES, PT, and possibly EL and IT) from the training
     table.
  3. **Raise the number of training steps** to match the number of rows seen in exp_1.
  4. **Longer-season varieties in the south** (Grain_maize_202–205, as in the worlds branch), or
     irrigation (PP) for the Mediterranean. Both need new simulations.

## Files

**Locations and simulations:**
- `data/raw/locations/locations_maize_eu27_cell01_n2000_s42.csv`
- `/projects/prjs1788/tnp-crop/wofost_eu_grid/` (runs, manifest, `processed/`, `processed_fr_be_dk/`,
  `checks/`)
- `/projects/prjs1788/tnp-crop/processed_old_matched_466/`

**Experiments:**
- training under `/projects/prjs1788/tnp-crop/nanotabicl/{exp_1_eu_grid, exp_1_eu_grid_frbedk, exp_1_old_matched_466}/`;
- benchmark results under `/projects/prjs1788/tnp-crop/cybench_output/` with the same names.

**Figures and tables** in `analysis/scaling/`:
- figures: `data_*.png`, `results_*.png`, `decisive_test.png`, `sanity/sanity_per_country.png`;
- CSVs: `data_per_country.csv`, `results_exp1_eu_grid_per_run.csv`, `decisive_test_per_country.csv`,
  `feature_yield_agreement.csv`.

**Scripts:**
- `src/data_pipeline/locations/grid_sample_locations.py`
- `scripts/slurm/run_wofost_eu_grid.sh`
- `scripts/slurm/submit_exp1.sh`
- `scripts/slurm/run_scaling_checks.sh`
- `scripts/slurm/run_repro_old.sh`
- `src/data_checks/season_sanity.py`
- `analysis/scaling/plot_data.py`
- `analysis/scaling/plot_results.py`
