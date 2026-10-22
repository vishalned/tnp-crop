# analysis/data

- `zarr_points_per_country.csv`: points per country (computed 2026-10-22) in
  - `zarr_all_points`: the CropFM zarr `/projects/prjs1788/CropFM/europe_data_1200_merged_cleaned2_rechunked.zarr`
    (1,068,260 points, 40 countries);
  - `zarr_maize_mask` / `zarr_wheat_mask`: of those, WorldCereal crop mask = 1 (maize) / 2 (winter cereals);
  - `locations_maize_csv`: `data/raw/locations/locations_maize.csv` (10,000 random maize points of the
    `nw_europe` group, seed 42);
  - `simulated_maize_first_1585`: its first 1,585 rows, the ones with downloaded soil/weather (used by
    the WOFOST runs and the worlds datasets).
