"""Sanity test: does WOFOST continued pretraining help, or is the gain what any extra pretraining gives?

    uv run python -m analysis.sanity_permuted_yields.plot_overview

CY-Bench walk-forward, 18 European maize countries, single pass, pretraining seed 0 for every pretrained run (default TabICL is one run).
Same bars, names and colours as the other clean bar plots. The last bar is the sanity run: the log-uniform-points recipe on the densified OM-fix table with
the yields shuffled among the rows of every training table (real WOFOST features, no feature -> yield link). Writes `figures/overview.png` and `overview.csv`
(mean over the countries).
"""

import os

import pandas as pd

from analysis.irr_noise_exps.plot_separate import BASE, W, draw

OUT = os.path.dirname(os.path.abspath(__file__))
RUNS = {
    "TabICL default": BASE["TabICL default"],
    "SCM control": BASE["SCM control"],
    "WOFOST, densified (OM fix, new soil)": BASE["WOFOST, densified (OM fix, new soil)"],
    "+ log-uniform points per table": BASE["+ log-uniform points per table"],
    "+ irrigation map (MIRCA-OS 2015)": ("exp_3_irrigation_map", W, 0.0, "#2a78d6"),
    "Sanity: yields permuted within each table": ("sanity_permuted_yields", W, 0.0, "#b3122e"),
}


def main():
    res = draw(RUNS, f"{OUT}/figures/overview.png",
               "CY-Bench walk-forward, 18 European maize countries: permuted-yield sanity run (pretraining seed 0)")
    out = pd.DataFrame([{"run": r, **v} for r, v in res.items()]).round(4)
    out.to_csv(f"{OUT}/overview.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
