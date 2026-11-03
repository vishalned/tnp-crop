"""Collect every experiment's results into the repository, so figures can be rebuilt without the cluster files.

    uv run python -m analysis.collect_results [--bench ROOT] [--runs ROOT]

* `results/runs.csv`: one row per experiment x setting x pretraining seed x country, from each benchmark folder's `per_run.csv`
  (`cybench/runs/tabicl_wofost/summarize_exp.py`): experiment, setting, seed (empty for the default TabICL: it is one deterministic run),
  country, n_region_years, nrmse, r2, spatial_r, temporal_r, anomaly_r.
* `results/training/<experiment>/<run>/`: the small files of each training run (config.json, log.csv, metrics_*.json, noise_params.csv is
  skipped: it is large); checkpoints and predictions stay in the project space.

`results/experiments.csv` (the index: question, config, status) is written by hand.
"""

import argparse
import glob
import os
import shutil

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(os.path.dirname(HERE), "results")
ROOT = "/projects/prjs1788/tnp-crop"
KEEP = ("config.json", "log.csv")  # + metrics_*.json


def collect_runs(bench: str) -> pd.DataFrame:
    frames = []
    for path in sorted(glob.glob(f"{bench}/*/per_run.csv")):
        d = pd.read_csv(path)
        frames.append(d.assign(experiment=os.path.basename(os.path.dirname(path))))
    df = pd.concat(frames, ignore_index=True)
    cols = ["experiment", "setting", "seed", "country", "n_region_years", "nrmse", "r2", "spatial_r", "temporal_r", "anomaly_r"]
    return df[cols].sort_values(["experiment", "setting", "seed", "country"], na_position="first").reset_index(drop=True)


def collect_training(runs: str) -> int:
    n = 0
    for cfg in sorted(glob.glob(f"{runs}/*/*/config.json")):
        run_dir = os.path.dirname(cfg)
        dest = os.path.join(RESULTS, "training", os.path.basename(os.path.dirname(run_dir)), os.path.basename(run_dir))
        os.makedirs(dest, exist_ok=True)
        for name in [*KEEP, *[os.path.basename(p) for p in glob.glob(f"{run_dir}/metrics_*.json")]]:
            if os.path.exists(f"{run_dir}/{name}"):
                shutil.copy2(f"{run_dir}/{name}", f"{dest}/{name}")
        n += 1
    return n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bench", default=f"{ROOT}/cybench_output")
    p.add_argument("--runs", default=f"{ROOT}/nanotabicl")
    a = p.parse_args()
    os.makedirs(RESULTS, exist_ok=True)
    runs = collect_runs(a.bench)
    runs.round(6).to_csv(f"{RESULTS}/runs.csv", index=False)
    print(f"results/runs.csv: {len(runs)} rows, {runs.experiment.nunique()} experiments")
    print(f"results/training: {collect_training(a.runs)} runs")


if __name__ == "__main__":
    main()
