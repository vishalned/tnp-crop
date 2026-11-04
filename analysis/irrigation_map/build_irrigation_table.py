"""The irrigation-map training table: per location the irrigated (PP) or the rainfed simulation, as the MIRCA-OS map says.

    uv run python -m analysis.irrigation_map.build_irrigation_table      # reads two ~5.6 GB tables: run it in a job

Locations the map marks irrigated (`location_irrigation.csv`) take the potential-production season (yield, fPAR and the rest of
the crop and weather series); potential production has no water balance, so for those seasons `ssm`, `rsm` and the soil statics
(`awc`, `bulk_density`) come from the rainfed run of the same season (same location, sowing year and jitter; same sowing date and
weather). All other locations keep their rainfed season. Seasons of an irrigated location that the rainfed run failed on (no soil) are
dropped. Adds `irrigated` and `irrigated_share` columns. Output: one parquet (+ the columns json) in `--out`.
"""

import argparse
import json
import os
import shutil

import pandas as pd

RUN = "/projects/prjs1788/tnp-crop"
KEY = ["location_index", "sowing_year", "jitter_index"]
HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--rainfed", default=f"{RUN}/wofost_eu_grid_dense_omfix/processed/wofost_maize_daily.parquet")
    p.add_argument("--irrigated", default=f"{RUN}/wofost_eu_grid_dense_pp/processed/wofost_maize_daily.parquet")
    p.add_argument("--assignment", default=f"{HERE}/location_irrigation.csv")
    p.add_argument("--out", default=f"{RUN}/wofost_eu_grid_dense_irrmap/processed")
    args = p.parse_args()

    assign = pd.read_csv(args.assignment)
    irr = set(assign.loc[assign["irrigated"], "location_index"])
    share = assign.set_index("location_index")["irrigated_share"]
    rain = pd.read_parquet(args.rainfed)
    columns = list(rain.columns)
    soil_cols = [c for c in columns if c.startswith(("ssm_d", "rsm_d"))] + ["awc", "bulk_density"]
    print(f"rainfed table {len(rain)} seasons; {len(irr)} irrigated locations", flush=True)

    pp = pd.read_parquet(args.irrigated)
    pp = pp[pp["location_index"].isin(irr)].set_index(KEY)
    rain_irr = rain[rain["location_index"].isin(irr)].set_index(KEY)
    keep = pp.index.intersection(rain_irr.index)
    print(f"irrigated locations: {len(pp)} PP seasons, {len(rain_irr)} rainfed seasons, {len(keep)} in both "
          f"(dropped: {len(pp) - len(keep)} PP seasons without a rainfed twin)", flush=True)
    mixed = pp.loc[keep].copy()
    mixed[soil_cols] = rain_irr.loc[keep, soil_cols].to_numpy()
    mixed = mixed.reset_index()[columns]
    mixed["irrigated"] = True
    rest = rain[~rain["location_index"].isin(irr)].copy()
    rest["irrigated"] = False
    out = pd.concat([rest, mixed], ignore_index=True).sort_values(KEY).reset_index(drop=True)
    out["irrigated_share"] = out["location_index"].map(share)

    os.makedirs(args.out, exist_ok=True)
    out.to_parquet(f"{args.out}/wofost_maize_daily.parquet", index=False)
    cols_json = args.rainfed.replace(".parquet", "_columns.json")
    if os.path.exists(cols_json):
        shutil.copy(cols_json, f"{args.out}/wofost_maize_daily_columns.json")
    n_irr = int(out["irrigated"].sum())
    print(f"{len(out)} seasons ({n_irr} irrigated from {out.loc[out['irrigated'], 'location_index'].nunique()} locations, "
          f"{len(out) - n_irr} rainfed from {out.loc[~out['irrigated'], 'location_index'].nunique()} locations) -> {args.out}", flush=True)
    print(json.dumps(out.groupby("irrigated")["yield_t_per_ha"].describe().round(2).to_dict("index")), flush=True)


if __name__ == "__main__":
    main()
