"""Simulation audit of a WOFOST batch run: do the simulated values make
sense? (Completeness/errors of the dataset itself: `coverage_audit`.)

    uv run python -m src.data_checks.simulation_audit --manifest data/raw/wofost/dataset_manifest.csv \
        --locations-csv data/raw/locations/locations_wheat.csv

From the manifest (every successful episode): yield distributions per
country/year, runs that never reached maturity, yield outliers, the effect
of the sowing-date jitter, year-to-year variability, yield vs latitude and
soil water capacity.

From the daily WOFOST files (a random sample, `--max-daily-files`):
phenology (flowering/maturity dates, season length, runs stopped by the
max-duration limit), physical sanity of the trajectories (LAI range,
biomass never decreasing, storage organs <= biomass, harvest index, water
stress factor and soil moisture in range), and whether yield responds to
the season's rain and heat.

Plus the daily curves (DVS, LAI, biomass/yield, soil moisture + water
stress, weather) of `--num-example-locations` random locations, all
jitters of one year, and their yield across years.
"""

import argparse
import sys

import matplotlib.dates as mdates
import numpy as np
import pandas as pd

from src.data_checks.common import Report, default_output_dir, load_manifest, plt, resolve_path

# plausible ranges (dry matter). Outside these is worth a look, not necessarily wrong.
YIELD_RANGE_T_HA = {"wheat": (1.0, 16.0), "maize": (1.0, 18.0)}
# typical peak LAI of a healthy crop; a low peak means little leaf area (and so low fpar)
LAI_MAX_RANGE = {"wheat": (3.0, 8.0), "maize": (3.0, 7.0)}
HARVEST_INDEX_RANGE = (0.2, 0.7)


def summarize_daily(path: str) -> dict:
    """Per-episode statistics from one daily WOFOST CSV."""
    d = pd.read_csv(path)
    d["day"] = pd.to_datetime(d["day"])
    out = {"days_simulated": len(d)}
    dvs = d["DVS"].to_numpy()
    for event, threshold in (("flowering", 1.0), ("maturity", 2.0)):
        hit = np.flatnonzero(dvs >= threshold)
        out[f"{event}_date"] = d["day"].iloc[hit[0]] if len(hit) else pd.NaT
    out["lai_max"] = d["LAI"].max()
    out["tagp_final"] = d["TAGP"].iloc[-1]
    out["wso_final"] = d["WSO"].iloc[-1] if "WSO" in d else np.nan
    out["harvest_index"] = out["wso_final"] / out["tagp_final"] if out["tagp_final"] > 0 else np.nan
    # tiny end-of-season drops (senescence/translocation) are normal; flag drops > 1% of the peak
    out["tagp_decreases"] = bool((d["TAGP"].cummax() - d["TAGP"]).max() > 0.01 * d["TAGP"].max())
    out["wso_exceeds_tagp"] = bool(("WSO" in d) and (d["WSO"] > d["TAGP"] + 1e-6).any())
    if "RFTRA" in d:
        out["rftra_out_of_range"] = bool(((d["RFTRA"] < -1e-6) | (d["RFTRA"] > 1 + 1e-6)).any())
        out["water_stress_days"] = int((d["RFTRA"] < 0.9).sum())  # transpiration reduced by >10%
    sm_cols = [c for c in d.columns if c.startswith("SM_layer")]
    if sm_cols:
        sm = d[sm_cols].to_numpy()
        out["sm_min"], out["sm_max"] = np.nanmin(sm), np.nanmax(sm)
    if "RAIN" in d:
        out["season_rain_mm"] = d["RAIN"].sum() * 10.0  # cm -> mm
        out["season_tmax_mean"] = d["TMAX"].mean()
        out["hot_days"] = int((d["TMAX"] > 30).sum())
    return out


def plot_location(df_loc: pd.DataFrame, wofost_dir: str, rng: np.random.Generator):
    """One figure: all jitters of one random year of a location, plus its yield over years."""
    ok = df_loc[df_loc["success"]]
    year = int(rng.choice(ok["year"].unique()))
    runs = ok[ok["year"] == year].sort_values("jitter_index")
    first = runs.iloc[0]
    fig, axes = plt.subplots(2, 3, figsize=(15, 7.5))
    fig.suptitle(f"{first['country']} ({first['longitude']:.3f}, {first['latitude']:.3f}), {first['crop']}, "
                 f"sowing year {year}: {len(runs)} jitters", fontsize=11)
    weather = None
    stress_ax = axes[1, 0].twinx()
    for _, r in runs.iterrows():
        path = resolve_path(r["daily_path"], wofost_dir)
        if path is None:
            continue
        d = pd.read_csv(path)
        d["day"] = pd.to_datetime(d["day"])
        lbl = f"j{r['jitter_index']} sown {r['sowing_date']:%m-%d} ({r['yield_t_per_ha']:.1f} t/ha)"
        axes[0, 0].plot(d["day"], d["DVS"], label=lbl)
        axes[0, 1].plot(d["day"], d["LAI"], label=lbl)
        line, = axes[0, 2].plot(d["day"], d["TAGP"] / 1000, label=f"TAGP {lbl}")
        if "WSO" in d:
            axes[0, 2].plot(d["day"], d["WSO"] / 1000, ls="--", color=line.get_color())
        if "SM_layer0" in d:
            axes[1, 0].plot(d["day"], d["SM_layer0"], color=line.get_color())
        if "RFTRA" in d:
            stress_ax.plot(d["day"], d["RFTRA"], ls=":", color=line.get_color())
        if weather is None or len(d) > len(weather):
            weather = d
    axes[0, 0].axhline(1, color="grey", lw=0.5); axes[0, 0].axhline(2, color="grey", lw=0.5)
    axes[0, 0].set_title("development stage DVS (1 = flowering, 2 = maturity)"); axes[0, 0].legend(fontsize=7)
    axes[0, 1].set_title("leaf area index LAI (m2/m2)")
    axes[0, 2].set_title("biomass TAGP (solid) and storage organs WSO (dashed), t/ha")
    axes[1, 0].set_title("topsoil moisture SM_layer0 (solid, left), RFTRA (dotted, right)")
    axes[1, 0].set_ylabel("cm3/cm3"); stress_ax.set_ylim(-0.05, 1.05)
    stress_ax.set_ylabel("RFTRA (1 = no water stress)")
    if weather is not None and "TMAX" in weather:
        ax = axes[1, 1]
        ax.plot(weather["day"], weather["TMAX"], color="#c0392b", lw=0.8, label="TMAX")
        ax.plot(weather["day"], weather["TMIN"], color="#2c7fb8", lw=0.8, label="TMIN")
        ax.set_title("weather: temperature (degC) and rain (mm, bars)"); ax.legend(fontsize=7, loc="upper left")
        ax2 = ax.twinx(); ax2.bar(weather["day"], weather["RAIN"] * 10, color="#7fb3d5", alpha=0.6, width=1.0)
    ax = axes[1, 2]
    for j, g in ok.groupby("jitter_index"):
        ax.plot(g["year"], g["yield_t_per_ha"], "o-", ms=3, lw=0.8, label=f"jitter {j}")
    ax.set_title("yield (TWSO) per sowing year, t/ha"); ax.legend(fontsize=7)
    locator = mdates.AutoDateLocator(maxticks=7)
    for a in list(axes.flat)[:5]:
        a.xaxis.set_major_locator(locator)
        a.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    fig.tight_layout()
    return fig


def run(manifest_path: str, locations_csv: str = None, out_dir: str = None, wofost_dir: str = None,
        num_example_locations: int = 5, max_daily_files: int = 500, seed: int = 0) -> str:
    rng = np.random.default_rng(seed)
    df = load_manifest(manifest_path, locations_csv)
    out_dir = out_dir or default_output_dir(manifest_path, "simulation")
    rep = Report(out_dir, "Simulation audit")
    ok = df[df["success"]].copy()
    rep.text(f"Manifest: `{manifest_path}`; {len(ok):,} successful episodes, {ok['location_index'].nunique():,} locations.")
    countries = sorted(ok["country"].unique())

    # --- yields -----------------------------------------------------------------
    rep.section("Yield (TWSO, dry matter, t/ha)")
    ystats = ok.groupby(["crop", "country"])["yield_t_per_ha"].describe(percentiles=[0.05, 0.5, 0.95]).round(2)
    rep.table(ystats, "yield_per_country")
    for crop, (lo, hi) in YIELD_RANGE_T_HA.items():
        c = ok[ok["crop"] == crop]
        if len(c):
            out = c[(c["yield_t_per_ha"] < lo) | (c["yield_t_per_ha"] > hi)]
            rep.flag(len(out) == 0, f"{crop}: {len(out):,} of {len(c):,} yields outside {lo}-{hi} t/ha")
            if len(out):
                rep.table(out[["location_index", "country", "year", "jitter_index", "sowing_date", "yield_t_per_ha", "final_dvs"]]
                          .sort_values("yield_t_per_ha"), f"yield_outliers_{crop}", max_rows=15, index=False)
    rep.text("\nFor reference, reported (fresh-weight) national yields are roughly 6-9 t/ha for wheat and 8-11 t/ha "
             "for grain maize in NW Europe; water-limited, N-unlimited WOFOST dry-matter yields usually sit above that.")

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    axes[0].boxplot([ok.loc[ok["country"] == c, "yield_t_per_ha"] for c in countries], tick_labels=countries)
    axes[0].set_ylabel("yield t/ha"); axes[0].set_title("yield per country"); axes[0].tick_params(axis="x", rotation=30)
    for c in countries:
        s = ok[ok["country"] == c].groupby("year")["yield_t_per_ha"]
        axes[1].plot(s.median().index, s.median().values, "o-", ms=3, label=c)
    axes[1].set_xlabel("sowing year"); axes[1].set_ylabel("median yield t/ha"); axes[1].set_title("median yield per year")
    axes[1].legend(fontsize=7)
    rep.figure(fig, "yield_distribution", "Yield per country and per year")

    # --- maturity -----------------------------------------------------------------
    rep.section("Did the runs reach maturity?")
    ok["matured"] = ok["final_dvs"] >= 2.0
    mat = ok.groupby(["crop", "country"]).agg(episodes=("matured", "size"), not_matured=("matured", lambda s: int((~s).sum())),
                                              final_dvs_min=("final_dvs", "min"))
    mat["not_matured_share"] = (mat["not_matured"] / mat["episodes"]).round(3)
    rep.table(mat, "maturity_per_country")
    rep.flag(mat["not_matured"].sum() == 0, f"{int(mat['not_matured'].sum()):,} episodes stopped before maturity "
             "(forced end at max duration: their yield is not a harvest yield; the training store drops them)")

    # --- jitter effect -------------------------------------------------------------
    rep.section("Effect of the sowing-date jitter")
    cell = ok.groupby(["location_index", "year"])["yield_t_per_ha"]
    spread = pd.DataFrame({"n": cell.size(), "mean": cell.mean(), "range": cell.max() - cell.min()})
    spread = spread[spread["n"] > 1]
    if len(spread):
        rel = spread["range"] / spread["mean"]
        rep.text(f"- yield range across jitters of a location-year: median {spread['range'].median():.2f} t/ha "
                 f"({rel.median():.1%} of the mean), 95th pct {spread['range'].quantile(0.95):.2f} t/ha")
        rep.flag(rel.median() > 0.001, "jitters give different outcomes (if identical, the sowing date isn't reaching the model)")
        if "sowing_offset_days" in ok:
            dev = ok["yield_t_per_ha"] - cell.transform("mean")
            for crop, g in ok.assign(dev=dev).groupby("crop"):
                slope = np.polyfit(g["sowing_offset_days"], g["dev"], 1)[0] if g["sowing_offset_days"].nunique() > 1 else np.nan
                rep.text(f"- {crop}: {slope * 10:+.3f} t/ha per 10 days later sowing (within location-year)")
    # --- variability across years/space ---------------------------------------------
    rep.section("Variability across years and space")
    loc_year = ok.groupby(["location_index", "year"])["yield_t_per_ha"].mean()
    cv = loc_year.groupby("location_index").agg(lambda s: s.std() / s.mean())
    rep.text(f"- year-to-year CV of yield per location: median {cv.median():.1%} (5-95%: {cv.quantile(0.05):.1%}-"
             f"{cv.quantile(0.95):.1%}); real-world wheat is typically ~10-20%")
    per_loc = ok.groupby("location_index").agg(longitude=("longitude", "first"), latitude=("latitude", "first"),
                                               yield_mean=("yield_t_per_ha", "mean"), awc=("awc", "first"))
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    sc = axes[0].scatter(per_loc["longitude"], per_loc["latitude"], c=per_loc["yield_mean"], s=10, cmap="viridis")
    fig.colorbar(sc, ax=axes[0], label="mean yield t/ha"); axes[0].set_title("mean yield per location")
    axes[1].scatter(per_loc["latitude"], per_loc["yield_mean"], s=8); axes[1].set_xlabel("latitude"); axes[1].set_ylabel("mean yield t/ha")
    axes[2].scatter(per_loc["awc"], per_loc["yield_mean"], s=8); axes[2].set_xlabel("topsoil available water capacity (awc)")
    axes[2].set_ylabel("mean yield t/ha")
    rep.figure(fig, "yield_space", "Mean yield per location, vs latitude and vs soil water capacity")

    # --- daily files: phenology + physical sanity ------------------------------------
    rep.section(f"Daily trajectories (random sample of up to {max_daily_files:,} episodes)")
    sample = ok.sample(n=min(max_daily_files, len(ok)), random_state=seed)
    records = []
    for idx, r in sample.iterrows():
        path = resolve_path(r["daily_path"], wofost_dir)
        if path is None:
            continue
        s = summarize_daily(path)
        s["index"] = idx
        records.append(s)
    if not records:
        rep.text("No daily files found (pass --wofost-dir if the data moved); skipping.")
    else:
        daily = pd.DataFrame(records).set_index("index").join(sample[["country", "crop", "year", "sowing_date", "yield_t_per_ha", "final_dvs"]])
        rep.text(f"{len(daily):,} daily files read.")
        daily["season_days"] = (daily["maturity_date"] - daily["sowing_date"]).dt.days
        daily["flowering_doy"] = daily["flowering_date"].dt.dayofyear
        daily["maturity_doy"] = daily["maturity_date"].dt.dayofyear
        phen = daily.groupby(["crop", "country"]).agg(
            flowering_doy_median=("flowering_doy", "median"), maturity_doy_median=("maturity_doy", "median"),
            maturity_doy_min=("maturity_doy", "min"), maturity_doy_max=("maturity_doy", "max"),
            season_days_median=("season_days", "median"), lai_max_median=("lai_max", "median"),
            harvest_index_median=("harvest_index", "median"), water_stress_days_median=("water_stress_days", "median"),
        ).round(2)
        rep.table(phen, "phenology_per_country")

        checks = [
            ("TAGP (cumulative biomass) never drops by more than 1% of its peak", ~daily["tagp_decreases"]),
            ("storage organs WSO never exceed TAGP", ~daily["wso_exceeds_tagp"]),
            ("max LAI in the typical range for the crop (" + ", ".join(f"{c} {lo}-{hi}" for c, (lo, hi) in LAI_MAX_RANGE.items()) + ")",
             pd.Series([LAI_MAX_RANGE.get(c, (0, np.inf))[0] <= v <= LAI_MAX_RANGE.get(c, (0, np.inf))[1]
                        for c, v in zip(daily["crop"], daily["lai_max"])], index=daily.index)),
            (f"harvest index WSO/TAGP within {HARVEST_INDEX_RANGE} (matured runs)",
             daily.loc[daily["final_dvs"] >= 2, "harvest_index"].between(*HARVEST_INDEX_RANGE).reindex(daily.index, fill_value=True)),
        ]
        if "rftra_out_of_range" in daily:
            checks.append(("water stress factor RFTRA within [0, 1]", ~daily["rftra_out_of_range"].astype(bool)))
        if "sm_max" in daily:
            checks.append(("soil moisture within (0, 0.6] cm3/cm3", (daily["sm_min"] > 0) & (daily["sm_max"] <= 0.6)))
        checks.append(("flowering reached", daily["flowering_date"].notna()))
        for name, passed in checks:
            n_bad = int((~passed).sum())
            rep.flag(n_bad == 0, f"{name}: {n_bad:,} of {len(daily):,} episodes fail")
        rep.text("\nA low peak LAI together with a very high harvest index means the crop builds little leaf/stem "
                 "biomass and puts nearly everything into the grain. Worth checking against a known reference before "
                 "training on it, since fpar (a model input) is computed from LAI.")

        fig, axes = plt.subplots(1, 3, figsize=(16, 4))
        for c in countries:
            g = daily[daily["country"] == c]
            axes[0].hist(g["maturity_doy"].dropna(), bins=30, histtype="step", label=c)
            axes[1].hist(g["season_days"].dropna(), bins=30, histtype="step", label=c)
        axes[0].set_xlabel("maturity day of year"); axes[0].legend(fontsize=7)
        axes[1].set_xlabel("season length, sowing -> maturity (days)")
        axes[2].scatter(daily["harvest_index"], daily["yield_t_per_ha"], s=6)
        axes[2].set_xlabel("harvest index WSO/TAGP"); axes[2].set_ylabel("yield t/ha")
        rep.figure(fig, "phenology", "Maturity dates, season length and harvest index")

        if "season_rain_mm" in daily:
            rep.section("Does yield respond to the season's weather?")
            valid = daily.dropna(subset=["season_rain_mm", "season_tmax_mean", "yield_t_per_ha"])
            for var, label in (("season_rain_mm", "season rain (mm)"), ("season_tmax_mean", "mean TMAX (degC)"),
                               ("water_stress_days", "water-stress days")):
                if var in valid and valid[var].nunique() > 1:
                    rep.text(f"- corr(yield, {label}) = {valid['yield_t_per_ha'].corr(valid[var]):+.2f}")
            rep.text("Expect: yield up with rain and down with water-stress days in water-limited runs; "
                     "a correlation near zero everywhere suggests the weather isn't driving the model.")
            fig, axes = plt.subplots(1, 3, figsize=(16, 4))
            for ax, var, label in zip(axes, ("season_rain_mm", "season_tmax_mean", "water_stress_days"),
                                      ("season rain, sowing -> end (mm)", "mean TMAX over the season (degC)",
                                       "days with transpiration reduced > 10% (RFTRA < 0.9)")):
                if var in valid:
                    ax.scatter(valid[var], valid["yield_t_per_ha"], s=6)
                    ax.set_xlabel(label); ax.set_ylabel("yield t/ha")
            rep.figure(fig, "yield_vs_weather", "Yield vs season rain, heat and water stress")

    # --- example curves -----------------------------------------------------------------
    rep.section(f"Daily curves of {num_example_locations} random locations")
    has_files = ok["daily_path"].map(lambda p: resolve_path(p, wofost_dir) is not None)
    locations = ok.loc[has_files, "location_index"].unique()
    if len(locations) == 0:
        rep.text("No daily files found (pass --wofost-dir if the data moved); skipping.")
    for loc_id in rng.choice(locations, size=min(num_example_locations, len(locations)), replace=False):
        fig = plot_location(df[df["location_index"] == loc_id], wofost_dir, rng)
        rep.figure(fig, f"curves_location_{loc_id}", f"Location {loc_id}")
    rep.text("What to look for: DVS rising to 1 (flowering) then 2 (maturity); LAI rising and collapsing after "
             "flowering; TAGP rising and levelling off; WSO starting after flowering and ending near the yield; "
             "RFTRA dropping below 1 in dry spells as topsoil moisture falls; jitters diverging only moderately.")

    return rep.save()


def main():
    if len(sys.argv) == 1:
        print("Usage: python -m src.data_checks.simulation_audit --manifest <dataset_manifest.csv> [--locations-csv <csv>] "
              "[--wofost-dir <dir>] [-o <out dir>] [--num-example-locations 5] [--max-daily-files 500] [--seed 0]")
        sys.exit(1)
    parser = argparse.ArgumentParser(description="Simulation audit of a WOFOST batch run (do the values make sense?).")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--locations-csv", default=None, help="The locations CSV the batch ran on (gives countries).")
    parser.add_argument("--wofost-dir", default=None, help="Where the episode files are, if the manifest paths moved.")
    parser.add_argument("-o", "--output-dir", default=None, help="Default: data/reports/dataset_checks/<run>/simulation")
    parser.add_argument("--num-example-locations", type=int, default=5)
    parser.add_argument("--max-daily-files", type=int, default=500, help="Daily files sampled for phenology/sanity checks.")
    parser.add_argument("--seed", type=int, default=0)
    a = parser.parse_args()
    run(a.manifest, a.locations_csv, a.output_dir, a.wofost_dir, a.num_example_locations, a.max_daily_files, a.seed)


if __name__ == "__main__":
    main()
