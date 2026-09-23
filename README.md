<div align="center">

# A Variant of Transformer Neural Process for Crops

</div>

<br>

## Environment Setup (uv)

This project uses [uv](https://docs.astral.sh/uv/) instead of conda for dependency management.

### First-time setup
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # install uv (one-time)
uv venv                                             # creates .venv using .python-version
uv sync --extra train      # if you're training models
uv sync --extra data       # if you're only running the data pipeline
uv sync --all-extras       # everything
```

### Running code
```bash
uv run python src/train.py experiment=tnp_synthetic
uv run python -m src.data_pipeline.soil.generate_soilgrids_soil_file -lon 6.656 -lat 52.966
uv run python -m src.data_pipeline.weather.generate_weather_file -lon 6.656 -lat 52.966 --start-date 2000-01-01 --end-date 2023-12-31
uv run python -m src.data_pipeline.wofost.run_wofost_simulation -lon 6.656 -lat 52.966 --crop wheat --year 2020
uv run python -m src.data_pipeline.wofost.generate_wofost_dataset --locations-csv path/to/locations.csv --crop wheat --num-years 5 --start-year 2010 --end-year 2024
```
The batch runner simulates `--num-years` **consecutive** years per location, starting at a random year chosen so the window fits inside `[--start-year, --end-year]` (reproducible with `--seed`); `--all-years` runs every year in that range instead. Each location's weather for the whole window is downloaded from GEE in one go before its episodes run. Locations run in parallel over `--workers` processes (default `min(8, cpu count)`; `--workers 1` = sequential); rows sharing the same coordinates always go to the same worker so they never write the same soil/weather cache file at once, and the same `--seed` gives the same episodes whatever the worker count. On an HPC cluster, run it inside a job with as many CPUs as `--workers` rather than on a login node.
Soil and weather (ERA5-Land daily, `src/data_pipeline/weather/utils_weather/gee_weather.py`) for the WOFOST run are pulled per location via Google Earth Engine (see "Google Earth Engine setup" below), and crop parameters are read from a local clone of the WOFOST_crop_parameters repo (see "Crop parameters" below) — both are one-time setup steps needed before the WOFOST commands above will run.

### Google Earth Engine setup

The GEE-based soil pipeline (`src/data_pipeline/soil/utils_soil/gee_soilgrids.py`, used by `generate_gee_soil_file.py` and the WOFOST runner) and the GEE weather pipeline (`gee_weather.py`, used by the WOFOST runner and `generate_weather_file.py`) need an authenticated Earth Engine project.

Authenticate **from inside Python, in the `uv` environment** — running `earthengine authenticate` directly from the shell did not work reliably:
```bash
uv run python
```
```python
import ee
ee.Authenticate()   # opens a browser flow once; credentials are then cached
```

`ensure_ee_initialized()` calls `ee.Initialize(project="cropfm")` — replace `"cropfm"` with your own Google Cloud project id (one with the Earth Engine API enabled) if you're not using that project.

### Crop parameters

Crop parameters are loaded from a **local clone** of the WOFOST_crop_parameters repo rather than fetched from GitHub at run time. We use the `herman-berghuijs` fork (not the upstream `ajwdewit` one) because it includes a fix for C4 crops (maize) that upstream doesn't have yet:
```bash
git clone -b wofost81 https://github.com/herman-berghuijs/WOFOST_crop_parameters.git data/crop_parameters/wofost81
```
`data/` is gitignored, so this clone needs to be repeated on every machine that runs the WOFOST pipeline. See `default_wofost_variables.default_crop_parameters_dir()`.

### Data dictionary

For an explanation of every column/field the data pipeline produces (WOFOST's daily output columns like `DVS`/`LAI`, the derived `cwb`/`fpar`/`ssm` features, the summary JSON fields, and the soil YAML structure), see [`docs/data_dictionary.md`](docs/data_dictionary.md).

### Adding a package
```bash
uv add requests                     # core dependency
uv add --optional train torchvision # only for the "train" extra
uv add --optional data opencv-python # only for the "data" extra
```
This updates `pyproject.toml`, `uv.lock`, and your `.venv` in one step.

### Removing a package
```bash
uv remove some-package
```

### Manually edited `pyproject.toml`?
If you hand-edit `pyproject.toml` instead of using `uv add`/`uv remove`, resync the lockfile:
```bash
uv lock
uv sync
```

### Quick local-only install (not saved to pyproject/lock)
```bash
uv pip install some-throwaway-package
```
Use only for ad hoc local testing — won't be shared with the team.

### Notes
- `uv.lock` and `.python-version` are committed to git — don't gitignore them.
- No manual "activate" needed — `uv run` handles the venv automatically.