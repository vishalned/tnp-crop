#!/bin/bash
# Pipeline check: re-simulate + re-process the first 20 old locations (FR/BE/DK run, cached soil/weather)
# with the current code and compare with the old manifest and processed table.
#SBATCH --job-name=repro-old
#SBATCH --output=/gpfs/home2/vnedungadi/tnp-crop/slurm_logs/repro-old%j.out
#SBATCH --error=/gpfs/home2/vnedungadi/tnp-crop/slurm_logs/repro-old%j.err
#SBATCH --partition=rome
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
set -euo pipefail
cd /home/vnedungadi/tnp-crop
R=/projects/prjs1788/tnp-crop/wofost_eu_grid/checks/repro_old20
uv run python -m src.data_pipeline.wofost.generate_wofost_dataset --locations-csv data/raw/locations/locations_maize.csv \
    --max-locations 20 --workers 8 --seed 42 -o $R > $R.log 2>&1
uv run python -m src.data_pipeline.wofost.process_wofost_dataset --manifest $R/dataset_manifest.csv \
    --locations-csv data/raw/locations/locations_maize.csv --wofost-dir $R -o $R/processed >> $R.log 2>&1
uv run python - <<'PY'
import numpy as np, pandas as pd
R = "/projects/prjs1788/tnp-crop/wofost_eu_grid/checks/repro_old20"
new = pd.read_csv(f"{R}/dataset_manifest.csv", float_precision="round_trip")
old = pd.read_csv("/projects/prjs1788/tnp-crop/wofost_calendar/dataset_manifest.csv", float_precision="round_trip")
k = ["location_index", "year", "jitter_index"]
m = new.merge(old, on=k, suffixes=("", "_old"))
d = (m.yield_kg_per_ha - m.yield_kg_per_ha_old).abs()
print(f"manifest: {len(new)} seasons, matched {len(m)}, same sowing date {(m.sowing_date == m.sowing_date_old).sum()}, "
      f"max |yield diff| {d.max():.6f} kg/ha, identical {(d < 1e-6).sum()}")
a = pd.read_parquet(f"{R}/processed/wofost_maize_daily.parquet")
b = pd.read_parquet("/projects/prjs1788/tnp-crop/processed/wofost_maize_daily.parquet", filters=[("location_index", "<", 20)])
k = ["location_index", "sowing_year", "jitter_index"]
a, b = a.sort_values(k).reset_index(drop=True), b.sort_values(k).reset_index(drop=True)
print("processed rows", len(a), len(b), "same columns", list(a.columns) == list(b.columns))
num = [c for c in a.columns if c in b.columns and a[c].dtype.kind in "fi"]
diff = {c: np.nanmax(np.abs(a[c].to_numpy(float) - b[c].to_numpy(float))) for c in num}
bad = {c: v for c, v in diff.items() if v > 1e-6}
print("numeric columns compared", len(num), "differing", len(bad), dict(list(bad.items())[:10]))
PY
