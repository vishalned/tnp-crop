# Country share of training tables: exp_1 (3 countries) → EU grid → densified EU grid

Branch `scaling_main`. Companion to `docs/scaling_report.md`. This report explains how training tables
are drawn per country, what the "share" of a country is, how the locations were sampled in each of the
three WOFOST datasets, and what that did to the share and to the table sizes. The densified dataset
was built to make the shares (nearly) equal; its results are in §5.

## 1. How a training batch is drawn

`CropTables.sample_batch` (`nanotabicl/crop_tables.py`, structured episodes), every training step:

1. pick a **lead country** uniformly among the training countries (those with ≥ 10 points);
2. pick **rows per point** = 1 target year + 5–11 context years, so 6–12;
3. pick **`n_points`** uniformly in [10, min(lead's points, 4096 // rows per point)];
4. build **8 tables of exactly that shape** (same `n_points`, rows per point and number of context rows).
   The first attempt uses the lead country; later attempts use a random country among those with
   ≥ `n_points` points. An attempt only succeeds if the country has `n_points` points that have the
   target year and enough earlier **matured** train years.

All tables of a batch share their shape because a batch is one tensor `[8, rows, features]` with one
`n_train`; the model has no padding. So the minimum table is 10 × 6 = **60 rows**, the cap **4,096 rows**,
and the largest `n_points` that can ever be drawn is 4096 // 6 = **682**.

## 2. What the "share" is

Three numbers per country, from `analysis/scaling/table_share.py`. It replays the rule above (2,000
steps × 8 tables × 3 seeds) on the real table, reading only the location, year and maturity columns;
the replay uses its own random numbers, so the numbers are expected values, not the exact tables of a
run. Output: `analysis/scaling/table_share.csv`.

- **lead share**: share of batches the country leads. Always ≈ 1 / number of countries.
- **lead built**: share of batches whose first table could actually be built from the lead country. Lower
  than the lead share when the country lacks points with enough matured years (Denmark, Lithuania).
- **table share**: share of **all** tables that come from the country. This is what the model sees.

The table share is unequal for two reasons:

1. **Filling the batch favours big countries.** Only countries with ≥ `n_points` points can fill the
   other 7 tables. With small shapes every country qualifies; with large shapes only the big ones.
2. **Failed lead tables go to other countries.** If the lead can't build its own table, its slot goes to
   another country.

So the shares are equal only if every country can build every shape: **≥ 682 usable points each**
(usable = matured in all 12 train years).

## 3. How the locations were sampled

### 3.1 exp_1: FR/BE/DK (`locations_maize.csv`)

- 10,000 maize points drawn at random from the CropFM Zarr (`nw_europe` group), the CSV sorted by country
  and Zarr index; only the **first 1,585 rows** were simulated (Belgium 224, Denmark 159, France 1,202).
- Because the CSV is sorted by Zarr index, which follows space, France is a strip of the south-west.
- The 1,585 points fall into only **509 ERA5-Land pixels** (median 2, up to 19 points with the same weather).
  France: 1,202 points in 330 pixels.

### 3.2 EU grid (`locations_maize_eu27_cell01_n2000_s42.csv`)

- EU-27 maize points; **one random point per 0.1° cell**, then 2,000 cells at random (`grid_sample_locations.py`).
- The cells were centred on multiples of 0.1° (rounded coordinates), meant to match ERA5-Land pixels.
  **They don't**: in Earth Engine the ERA5-Land pixels have their *edges* at multiples of 0.1° (floor). This
  was checked on the weather cache: points in the same floor cell have identical ERA5-Land values (all
  1,176 pairs), points in the same rounded cell often don't. So the 2,000 points are in **1,896** distinct
  ERA5-Land pixels (98 pixels with 2–3 points), not 2,000. Minor, but the claim "one point per weather
  pixel" in `docs/scaling_report.md` is slightly off.
- 1,966 locations with all seasons simulated; 18 training countries (Latvia 4, Slovenia 5, Sweden 2 dropped).

### 3.3 Densified EU grid (`locations_maize_eu27_dense700_s42.csv`)

`src/data_pipeline/locations/densify_grid_locations.py` (one-off). The 2,000 EU grid rows are kept
unchanged as the first rows (same `location_index`, same sowing dates, their simulations reused), and
points are appended. Per country:

1. **usable fraction** u = EU grid points that matured in all 12 train years / points sampled;
2. **points wanted** N = ceil(700 / u), capped at the country's maize points (countries with u = 0,
   Latvia and Sweden, get nothing new);
3. **points per pixel** k = max(1, ceil(N / ERA5-Land pixels)), pixels = floor 0.1° cells with maize points;
4. pixels in a seeded random order (pixels of existing points first), each filled up to k points until
   the country has N; if pixels with fewer than k points leave it short, further passes allow k+1, ….

New points never repeat an existing one and are ≥ 0.001° from a pixel edge (so they can reuse the
pixel's weather, §4). Result: **12,382 locations** (2,000 + 10,382) in 6,591 pixels.

| Country | Maize pts | Pixels | Usable now | Wanted | k | Existing | New | Total | Expected usable |
|---|---|---|---|---|---|---|---|---|---|
| France | 17,169 | 2,546 | 0.99 | 709 | 1 | 426 | 283 | 709 | 701 |
| Germany | 13,697 | 2,147 | 0.86 | 815 | 1 | 319 | 496 | 815 | 700 |
| Poland | 13,109 | 1,948 | 0.96 | 727 | 1 | 304 | 423 | 727 | 701 |
| Romania | 16,654 | 1,461 | 0.97 | 724 | 1 | 219 | 505 | 724 | 701 |
| Hungary | 11,482 | 926 | 1.00 | 700 | 1 | 129 | 571 | 700 | 700 |
| Italy | 9,673 | 654 | 0.96 | 731 | 2 | 121 | 610 | 731 | 701 |
| Bulgaria | 5,701 | 609 | 0.97 | 725 | 2 | 88 | 637 | 725 | 700 |
| Spain | 3,528 | 470 | 1.00 | 700 | 2 | 79 | 621 | 700 | 700 |
| Czechia | 2,441 | 479 | 0.77 | 905 | 2 | 62 | 843 | 905 | 701 |
| Slovakia | 2,642 | 268 | 0.85 | 826 | 4 | 46 | 780 | 826 | 700 |
| Croatia | 2,824 | 235 | 0.97 | 719 | 4 | 38 | 681 | 719 | 700 |
| Austria | 2,143 | 284 | 0.97 | 720 | 3 | 36 | 684 | 720 | 700 |
| Greece | 1,141 | 209 | 0.97 | 723 | 4 | 32 | 691 | 723 | 700 |
| Belgium | 791 | 142 | 1.00 | 700 | 5 | 22 | 678 | 700 | 700 |
| Portugal | 413 | 59 | 1.00 | 413 (all) | 7 | 11 | 402 | 413 | 413 |
| Netherlands | 347 | 104 | 1.00 | 347 (all) | 4 | 11 | 336 | 347 | 347 |
| Slovenia | 353 | 44 | 1.00 | 353 (all) | 9 | 5 | 348 | 353 | 353 |
| Lithuania | 262 | 123 | 0.35 | 262 (all) | 3 | 20 | 242 | 262 | 92 |
| Denmark | 577 | 179 | 0.04 | 577 (all) | 4 | 26 | 551 | 577 | 22 |
| Latvia / Sweden | 46 / 36 | | 0 | | | 4 / 2 | 0 | 4 / 2 | 0 |

("Maize pts" excludes points within 0.001° of a pixel edge. Full table:
`data/raw/locations/locations_maize_eu27_dense700_s42_per_country.csv`.)

- 14 countries reach ~700 usable points. Portugal, Netherlands and Slovenia (~350–410) can fill every
  12-rows-per-point shape but not the largest 6-rows-per-point ones. Slovenia becomes a training country.
- **Denmark and Lithuania stay small whatever is sampled**: their seasons often don't reach maturity
  (Grain_maize_201 in a cool climate; Denmark 50% of seasons, 1 of 23 EU grid points usable), so they
  rarely build a table. This is a crop-parameter issue, not a sampling one.

## 4. Pipeline changes for the densified run

- **Data on the project space.** Downloads, simulations, processed tables and reports now default to
  `/projects/prjs1788/tnp-crop/data` (`src/data_pipeline/paths.py`, `$TNP_DATA_ROOT`); the existing caches
  were moved there (checksums verified). Locations CSVs and crop parameters stay in the repo.
- **Weather per ERA5-Land pixel.** A location without its own weather file but in a pixel that has one
  takes that file's ERA5-Land values, fetches only its own elevation, and recomputes LAT/LON/ELEV and
  E0/ES0/ET0 for itself, restricted to the years its own download would fetch (the Angstrom A/B behind
  E0/ES0 depend on the days in the file). Checked on the cache: rebuilding a location's file from a
  pixel-mate reproduces it exactly. Points within 0.001° of a pixel edge always download (four old
  pairs ~10 m from an edge came from different pixels). The batch runner sends a pixel's locations to one
  worker, one after the other, so only the first downloads (in a test, two points of a new pixel otherwise
  both downloaded). Code: `era5_land_cell`, `records_for_location` in `gee_weather.py`. About 4,600 new
  pixels are downloaded; the other ~5,800 new points only fetch their elevation.
- **Soil per point, as before.** SoilGrids is sampled on its own 250 m Mollweide grid; only ~4% of the new
  points share a soil pixel with another point, and a quick check of a pixel mapping didn't reproduce
  Earth Engine's sampling exactly, so soil is still downloaded per point (exact-coordinate cache).
- **Simulation in chunks.** `generate_wofost_dataset.py --first-location` skips the rows already simulated
  (offsets are still drawn for every row, so the EU grid locations keep their episodes).
  `scripts/slurm/submit_eu_grid_dense.sh` runs the 10,382 new locations in 6 chunks
  (`run_wofost_eu_grid_dense.sh`, one at a time), processes each chunk on its own, and merges the EU grid
  table and the chunks' tables (`merge_processed_tables.py`, checked to give the single-run table).
- **exp_1 on the merged table**, WOFOST only (the SCM control was ≈ default TabICL in every earlier
  table): `exp_1_eu_grid_dense`, same recipe, 3 seeds.

## 5. Shares and table sizes, old → EU grid → densified

### exp_1: FR/BE/DK (1,563 locations)

| Country | Locations | Usable | Lead share | Lead built | **Table share** | Rows median | Rows max |
|---|---|---|---|---|---|---|---|
| France | 1,195 | 1,194 | 32.4% | 32.4% | **54.8%** | 1,251 | 4,096 |
| Belgium | 217 | 216 | 33.0% | 33.0% | **40.0%** | 870 | 2,580 |
| Denmark | 151 | 2 | 34.6% | 6.1% | **5.2%** | 405 | 888 |
| All | | | | | | 990 (mean 1,253) | 4,096 |

### EU grid (1,955 locations in the 18 training countries)

| Country | Locations | Usable | Lead share | Lead built | **Table share** | Rows median | Rows max |
|---|---|---|---|---|---|---|---|
| France | 421 | 421 | 5.9% | 5.9% | **11.9%** | 783 | 4,080 |
| Germany | 311 | 274 | 5.1% | 4.6% | **10.0%** | 540 | 3,311 |
| Poland | 303 | 293 | 5.4% | 5.3% | **10.2%** | 591 | 3,492 |
| Romania | 214 | 212 | 5.7% | 5.6% | **9.0%** | 420 | 2,532 |
| Hungary | 129 | 129 | 5.9% | 5.9% | **7.4%** | 308 | 1,548 |
| Italy | 116 | 116 | 5.2% | 5.2% | **7.3%** | 294 | 1,344 |
| Bulgaria | 85 | 85 | 5.2% | 5.2% | **6.6%** | 240 | 1,008 |
| Spain | 79 | 79 | 5.9% | 5.9% | **6.5%** | 231 | 948 |
| Czechia | 61 | 48 | 5.5% | 4.8% | **5.4%** | 192 | 594 |
| Slovakia | 45 | 39 | 5.1% | 4.3% | **4.4%** | 168 | 468 |
| Croatia | 37 | 37 | 5.3% | 5.3% | **4.5%** | 165 | 444 |
| Austria | 36 | 35 | 6.0% | 5.9% | **4.6%** | 160 | 420 |
| Greece | 31 | 31 | 5.8% | 5.8% | **4.2%** | 150 | 372 |
| Belgium | 22 | 22 | 6.0% | 6.0% | **3.5%** | 130 | 264 |
| Netherlands | 11 | 11 | 5.5% | 5.5% | **1.5%** | 99 | 132 |
| Portugal | 11 | 11 | 5.3% | 5.3% | **1.6%** | 90 | 132 |
| Lithuania | 20 | 7 | 6.0% | 2.1% | **1.2%** | 100 | 162 |
| Denmark | 23 | 1 | 5.3% | 0.2% | **0.1%** | 70 | 80 |
| All | | | | | | 240 (mean 529) | 4,080 |

### Densified EU grid

_Filled in after the run (`analysis/scaling/table_share.csv`, run `eu_grid_dense`)._

## 6. Results

_Filled in after `exp_1_eu_grid_dense` (figures in `analysis/scaling/figures/`)._
