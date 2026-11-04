"""CY-Bench result of pretraining with cultivar variation (exp_7_cultivar_tsum), same style as the other clean bar plots.

    uv run python -m analysis.cultivar_tsum.plot_results

18 European maize countries, walk-forward, single pass, pretraining seed 0 for every pretrained run (default TabICL is one run). The new run
is the log-uniform-points run with TSUM1 and TSUM2 scaled by one factor ~ U(0.8, 1.2) per location in the simulation. Writes
`figures/cultivar_tsum_seed0.png` and `results_seed0.csv` (mean over the countries).
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
    "+ cultivar variation (TSUM x U(0.8, 1.2))": ("exp_7_cultivar_tsum", W, 0.0, "#a65628"),
}


def main():
    res = draw(RUNS, f"{OUT}/figures/cultivar_tsum_seed0.png",
               "CY-Bench walk-forward, 18 European maize countries: cultivar variation (pretraining seed 0)")
    out = pd.DataFrame([{"run": r, **v} for r, v in res.items()]).round(4)
    out.to_csv(f"{OUT}/results_seed0.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
