"""Coverage audit of a WOFOST batch run: is the dataset complete and
well-formed? (Values/plausibility of the simulations: `simulation_audit`.)

    uv run python -m src.data_checks.coverage_audit --manifest data/raw/wofost/dataset_manifest.csv \
        --locations-csv data/raw/locations/locations_wheat.csv

Checks: locations/countries/crops, locations with and without errors (per
country), error causes, year coverage per country, completeness of the
location x year x jitter grid, jittered sowing dates per country,
duplicates, and whether every successful episode's files exist.
Writes `summary.md` + CSV tables + PNG figures to the output folder.
"""

import argparse
import sys

import numpy as np
import pandas as pd

from src.data_checks.common import Report, default_output_dir, error_category, load_manifest, plt, resolve_path


def location_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per location: episodes attempted/failed and a status label."""
    loc = df.groupby("location_index").agg(
        country=("country", "first"), crop=("crop", "first"),
        longitude=("longitude", "first"), latitude=("latitude", "first"),
        episodes=("success", "size"), failed=("success", lambda s: int((~s).sum())),
    )
    loc["status"] = np.select([loc["failed"] == 0, loc["failed"] == loc["episodes"]], ["no errors", "all failed"], "some errors")
    return loc


def run(manifest_path: str, locations_csv: str = None, out_dir: str = None, wofost_dir: str = None,
        check_files: bool = True, sowing_jitter_days: int = 10) -> str:
    df = load_manifest(manifest_path, locations_csv)
    out_dir = out_dir or default_output_dir(manifest_path, "coverage")
    rep = Report(out_dir, "Coverage audit")
    rep.text(f"Manifest: `{manifest_path}`" + (f"  \nLocations: `{locations_csv}`" if locations_csv else
             "  \n(no locations CSV given: countries unknown)"))

    # --- 1. overview + locations with/without errors ----------------------
    loc = location_table(df)
    rep.section("Overview")
    years = sorted(df["year"].unique())
    rep.text(f"- episodes: {len(df):,} ({df['success'].sum():,} success, {(~df['success']).sum():,} failed, "
             f"{(~df['success']).mean():.1%} failure rate)")
    rep.text(f"- locations: {len(loc):,}; countries: {loc['country'].nunique()}; crops: {sorted(df['crop'].dropna().unique())}")
    rep.text(f"- sowing years: {years[0]}-{years[-1]} ({len(years)} years); jitters per location-year: "
             f"{sorted(int(j) for j in df['jitter_index'].unique())}")
    counts = loc["status"].value_counts()
    rep.text(f"- locations with no errors: {counts.get('no errors', 0):,}; with some errors: {counts.get('some errors', 0):,}; "
             f"all episodes failed: {counts.get('all failed', 0):,}")

    rep.section("Per country")
    per_country = loc.groupby("country").agg(
        locations=("status", "size"),
        no_errors=("status", lambda s: int((s == "no errors").sum())),
        some_errors=("status", lambda s: int((s == "some errors").sum())),
        all_failed=("status", lambda s: int((s == "all failed").sum())),
        episodes=("episodes", "sum"), failed_episodes=("failed", "sum"),
    )
    per_country["failure_rate"] = (per_country["failed_episodes"] / per_country["episodes"]).round(4)
    per_country.loc["TOTAL"] = per_country.sum(numeric_only=True)
    per_country.loc["TOTAL", "failure_rate"] = round(per_country.loc["TOTAL", "failed_episodes"] / per_country.loc["TOTAL", "episodes"], 4)
    count_cols = [c for c in per_country.columns if c != "failure_rate"]
    per_country[count_cols] = per_country[count_cols].astype(int)
    rep.table(per_country, "per_country")
    rep.text("\nCountry imbalance matters: training samples countries uniformly, so a country with few usable "
             "locations gets revisited often (less diversity per episode).")

    fig, ax = plt.subplots(figsize=(8, 3.5))
    pc = per_country.drop(index="TOTAL")
    ax.bar(pc.index, pc["no_errors"], label="no errors", color="#4c9a6a")
    ax.bar(pc.index, pc["some_errors"], bottom=pc["no_errors"], label="some errors", color="#e0a030")
    ax.bar(pc.index, pc["all_failed"], bottom=pc["no_errors"] + pc["some_errors"], label="all failed", color="#c0392b")
    ax.set_ylabel("locations"); ax.legend(); ax.tick_params(axis="x", rotation=30)
    rep.figure(fig, "locations_per_country", "Locations per country by error status")

    fig, ax = plt.subplots(figsize=(7, 6))
    colors = {"no errors": "#4c9a6a", "some errors": "#e0a030", "all failed": "#c0392b"}
    for status, g in loc.groupby("status"):
        ax.scatter(g["longitude"], g["latitude"], s=8, c=colors[status], label=f"{status} ({len(g)})")
    ax.set_xlabel("longitude"); ax.set_ylabel("latitude"); ax.legend(); ax.set_aspect(1.4)
    rep.figure(fig, "location_map", "Locations by error status")

    # --- errors -------------------------------------------------------------
    rep.section("Error causes")
    failed = df[~df["success"]].copy()
    if failed.empty:
        rep.text("No failed episodes.")
    else:
        failed["cause"] = failed["error"].map(error_category)
        causes = failed.groupby("cause").agg(episodes=("cause", "size"), locations=("location_index", "nunique"),
                                             example=("error", "first")).sort_values("episodes", ascending=False)
        causes["example"] = causes["example"].str.slice(0, 200)
        rep.table(causes, "error_causes", index=True)
        by_country = failed.pivot_table(index="cause", columns="country", values="year", aggfunc="size", fill_value=0)
        rep.table(by_country, "error_causes_per_country")
        rep.text("\n'ERA5-Land has no data' = the point is outside ERA5-Land's land mask (coast/sea): drop or move those "
                 "locations. Errors spread over many locations/years point at a pipeline issue instead.")

    # --- 2. years per country -----------------------------------------------
    rep.section("Year coverage per country")
    ok = df[df["success"]]
    yr = df.groupby("country").agg(first_year=("year", "min"), last_year=("year", "max"), years=("year", "nunique"))
    yr["first_year_ok"] = ok.groupby("country")["year"].min()
    yr["last_year_ok"] = ok.groupby("country")["year"].max()
    rep.table(yr, "years_per_country")
    grid = ok.pivot_table(index="country", columns="year", values="location_index", aggfunc="nunique", fill_value=0)
    total = df.groupby("country")["location_index"].nunique()
    frac = grid.div(total, axis=0).reindex(index=total.index, columns=years, fill_value=0)
    fig, ax = plt.subplots(figsize=(max(6, len(years) * 0.5), 0.5 * len(frac) + 1.5))
    im = ax.imshow(frac.values, vmin=0, vmax=1, cmap="viridis", aspect="auto", interpolation="nearest")
    ax.set_xticks(range(len(years)), years, rotation=90); ax.set_yticks(range(len(frac)), frac.index)
    for i in range(frac.shape[0]):
        for j in range(frac.shape[1]):
            ax.text(j, i, f"{frac.values[i, j]:.0%}", ha="center", va="center", fontsize=7,
                    color="white" if frac.values[i, j] < 0.6 else "black")
    fig.colorbar(im, label="share of the country's locations with >= 1 successful jitter")
    rep.figure(fig, "year_coverage", "Share of each country's locations with a successful run per year")

    # --- completeness of the location x year x jitter grid ----------------------
    rep.section("Completeness of the location x year x jitter grid")
    jitters = sorted(df["jitter_index"].unique())
    expected = len(years) * len(jitters)
    attempted = df.groupby("location_index").size()
    rep.flag((attempted == expected).all(),
             f"every location attempted all {len(years)} years x {len(jitters)} jitters = {expected} episodes "
             f"({(attempted < expected).sum()} locations have fewer rows: batch interrupted, or a run that samples a subset of years)")
    dup = df.duplicated(["location_index", "year", "jitter_index"]).sum()
    rep.flag(dup == 0, f"no duplicate (location, year, jitter) rows ({dup} duplicates)")
    coords = loc.groupby(["longitude", "latitude"]).size()
    rep.flag((coords == 1).all(), f"no two location_index share coordinates ({(coords > 1).sum()} shared; fine if "
                                   "they're different crops, they share soil/weather caches)")
    usable = ok.groupby(["location_index", "year"]).size().groupby("location_index").size()
    rep.text(f"- locations with a successful run in every year: {(usable == len(years)).sum():,} / {len(loc):,}")
    cells = ok.groupby(["location_index", "year"]).size()
    rep.text("- successful jitters per location-year: " + ", ".join(f"{k}: {v:,}" for k, v in cells.value_counts().sort_index().items()))

    # --- 3. jittered sowing dates per country ------------------------------
    rep.section("Sowing dates (jitter) per country")
    if "sowing_offset_days" in df:
        rep.flag(df["sowing_offset_days"].abs().max() <= sowing_jitter_days,
                 f"all sowing offsets within +/-{sowing_jitter_days} days (range {df['sowing_offset_days'].min()}.."
                 f"{df['sowing_offset_days'].max()})")
        distinct = df.groupby(["location_index", "year"])["sowing_offset_days"].nunique()
        size = df.groupby(["location_index", "year"]).size()
        rep.flag((distinct == size).all(), f"jitters of a location-year have distinct sowing dates "
                                           f"({(distinct < size).sum()} location-years with repeated offsets)")
    sow = ok.assign(sowing_doy=ok["sowing_date"].dt.dayofyear)
    stats = sow.groupby(["country", "crop"]).agg(
        episodes=("sowing_doy", "size"), doy_min=("sowing_doy", "min"), doy_median=("sowing_doy", "median"),
        doy_max=("sowing_doy", "max"),
        **({"offset_mean": ("sowing_offset_days", "mean"), "offset_std": ("sowing_offset_days", "std")} if "sowing_offset_days" in sow else {}),
    ).round(2)
    rep.table(stats, "sowing_per_country")
    countries = sorted(sow["country"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    axes[0].boxplot([sow.loc[sow["country"] == c, "sowing_doy"] for c in countries], tick_labels=countries)
    axes[0].set_ylabel("sowing day of year"); axes[0].tick_params(axis="x", rotation=30)
    if "sowing_offset_days" in sow:
        bins = np.arange(-sowing_jitter_days - 0.5, sowing_jitter_days + 1.5)
        for c in countries:
            axes[1].hist(sow.loc[sow["country"] == c, "sowing_offset_days"], bins=bins, histtype="step", label=c, density=True)
        axes[1].set_xlabel("sowing offset from season start (days)"); axes[1].set_ylabel("density"); axes[1].legend(fontsize=7)
    rep.figure(fig, "sowing_dates", "Sowing day of year and jitter offset per country")

    # --- 4. crops ------------------------------------------------------------
    rep.section("Crops")
    crops = df.groupby(["crop", "country"]).agg(locations=("location_index", "nunique"), episodes=("success", "size"),
                                                success=("success", "sum")).unstack("crop", fill_value=0)
    rep.table(crops, "crops_per_country")
    if "crop" in df and df.groupby("location_index")["crop"].nunique().max() > 1:
        rep.flag(False, "some location_index has more than one crop")

    # --- files ---------------------------------------------------------------
    if check_files:
        rep.section("Episode files")
        paths = ok["daily_path"].map(lambda p: resolve_path(p, wofost_dir))
        missing = paths.isna().sum()
        rep.flag(missing == 0, f"daily CSV exists for every successful episode ({missing:,} missing of {len(ok):,}"
                 + ("; pass --wofost-dir if the data moved" if missing else "") + ")")

    return rep.save()


def main():
    if len(sys.argv) == 1:
        print("Usage: python -m src.data_checks.coverage_audit --manifest <dataset_manifest.csv> "
              "[--locations-csv <csv>] [--wofost-dir <dir>] [-o <out dir>] [--no-file-check]")
        sys.exit(1)
    parser = argparse.ArgumentParser(description="Coverage audit of a WOFOST batch run (is the dataset complete?).")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--locations-csv", default=None, help="The locations CSV the batch ran on (gives countries).")
    parser.add_argument("--wofost-dir", default=None, help="Where the episode files are, if the manifest paths moved.")
    parser.add_argument("-o", "--output-dir", default=None, help="Default: data/reports/dataset_checks/<run>/coverage")
    parser.add_argument("--no-file-check", action="store_true", help="Skip checking that episode files exist.")
    parser.add_argument("--sowing-jitter-days", type=int, default=10)
    a = parser.parse_args()
    run(a.manifest, a.locations_csv, a.output_dir, a.wofost_dir, not a.no_file_check, a.sowing_jitter_days)


if __name__ == "__main__":
    main()
