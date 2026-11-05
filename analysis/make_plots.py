"""Draw the bar plots of every experiment (or the named ones) from results/runs.csv.

    uv run python -m analysis.make_plots [experiment ...]        # experiments/<id>/plots.yaml -> experiments/<id>/figures/

A plots.yaml is a list of plots: {file, title, seed (mean | 0 | ...), metrics (optional: two or more of nrmse, anomaly_r, r2, temporal_r, spatial_r), weighted (optional), errorbars (optional: +-1 std over the seeds; needs seed: mean), figsize (optional), title_size (optional),
runs: [name | {use: name, label: ...} | {label, experiment, setting, color}]}. Names are in analysis/runs.yaml. Next to each figure a CSV with the
mean over the countries (and weighted mean) of every run is written.
"""

import os
import sys

import yaml

from analysis.bars import draw, load_runs, seed_std, values

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main(only=()):
    named = yaml.safe_load(open(f"{ROOT}/analysis/runs.yaml"))
    runs = load_runs()
    for path in sorted(os.listdir(f"{ROOT}/experiments")):
        spec = f"{ROOT}/experiments/{path}/plots.yaml"
        if not os.path.exists(spec) or (only and path not in only):
            continue
        for plot in yaml.safe_load(open(spec)):
            entries = []
            for r in plot["runs"]:
                r = dict(named[r]) if isinstance(r, str) else {**named.get(r.get("use"), {}), **{k: v for k, v in r.items() if k != "use"}}
                entries.append({"label": r["label"], "color": r["color"],
                                "values": values(runs, r["experiment"], r["setting"], plot.get("seed", "mean")),
                                "std": seed_std(runs, r["experiment"], r["setting"]) if plot.get("errorbars") else None})
            out = f"{ROOT}/experiments/{path}/figures/{plot['file']}"
            summary = draw(entries, out, plot["title"], weighted=plot.get("weighted", False),
                           figsize=tuple(plot.get("figsize", (16, 7.8))), title_size=plot.get("title_size"),
                           metrics=tuple(plot.get("metrics", ("nrmse", "anomaly_r"))), errorbars=bool(plot.get("errorbars")))
            summary.round(4).to_csv(out.rsplit(".", 1)[0] + ".csv")
            print(f"{path}/figures/{plot['file']}: {len(entries)} runs")


if __name__ == "__main__":
    main(sys.argv[1:])
