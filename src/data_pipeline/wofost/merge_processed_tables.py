"""Merge processed tables of one crop, built in parts (`process_wofost_dataset.py` per chunk of locations),
into the table a single processing run would give.

    uv run python -m src.data_pipeline.wofost.merge_processed_tables -o <dir> <part dir> <part dir> ...

The parts must come from one locations CSV (`location_index` = its row number) and not overlap. A part
has as many day columns as its longest window, so the part with the most day columns gives the column
order and the others get NaN for the days they lack, as in a single run; rows are sorted like
`process_wofost_dataset.py` sorts them. Works in Arrow (no pandas copy), so it needs about the final
table's size in memory. Also merges the parts' `dataset_manifest.csv` when every part dir's parent has one
(`--manifests`). `--drop-countries` leaves countries out of the merged table (e.g. ones simulated but
not wanted for training).
"""

import argparse
import json
import os
import shutil

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


def merge_tables(part_dirs: list, output_dir: str, crop: str = "maize", drop_countries=()) -> str:
    name = f"wofost_{crop}_daily"
    tables = [pq.read_table(os.path.join(d, f"{name}.parquet")) for d in part_dirs]
    template = max(range(len(tables)), key=lambda i: tables[i].num_columns)
    columns = tables[template].column_names
    for d, t in zip(part_dirs, tables):
        extra = set(t.column_names) - set(columns)
        if extra:
            raise ValueError(f"{d} has columns the template part lacks: {sorted(extra)[:5]}")
    merged = pa.concat_tables(tables, promote_options="default").select(columns)
    del tables
    if drop_countries:
        keep = pc.invert(pc.is_in(merged["country"], value_set=pa.array(list(drop_countries))))
        print(f"dropping {merged.num_rows - pc.sum(keep).as_py()} seasons of {list(drop_countries)}")
        merged = merged.filter(keep)
    indices = pc.sort_indices(merged, sort_keys=[("location_index", "ascending"), ("year", "ascending"),
                                                 ("jitter_index", "ascending")])
    merged = merged.take(indices)
    keys = pc.binary_join_element_wise(pc.cast(merged["location_index"], pa.string()),
                                       pc.cast(merged["year"], pa.string()),
                                       pc.cast(merged["jitter_index"], pa.string()), "_")
    if pc.count_distinct(keys).as_py() != merged.num_rows:
        raise ValueError("The parts overlap: some (location_index, year, jitter_index) appear twice.")
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{name}.parquet")
    pq.write_table(merged, path)
    # the template part's column list describes exactly these columns
    shutil.copy(os.path.join(part_dirs[template], f"{name}_columns.json"), os.path.join(output_dir, f"{name}_columns.json"))
    with open(os.path.join(output_dir, f"{name}_columns.json")) as f:
        num_days = json.load(f)["num_days"]
    print(f"{crop}: {merged.num_rows} seasons x {num_days} days from {len(part_dirs)} parts -> {path}")
    return path


def merge_manifests(manifests: list, output_path: str) -> None:
    frames = [pd.read_csv(m, float_precision="round_trip") for m in manifests]
    merged = pd.concat(frames, ignore_index=True).sort_values(["location_index", "year", "jitter_index"])
    if merged.duplicated(["location_index", "year", "jitter_index"]).any():
        raise ValueError("The manifests overlap.")
    merged.to_csv(output_path, index=False)
    print(f"{len(merged)} episodes ({(merged['status'] == 'success').sum()} successful) -> {output_path}")


def main():
    p = argparse.ArgumentParser(description="Merge processed tables built in parts.")
    p.add_argument("parts", nargs="+", help="Directories holding wofost_<crop>_daily.parquet (+ _columns.json).")
    p.add_argument("-o", "--output-dir", required=True)
    p.add_argument("--crop", default="maize")
    p.add_argument("--drop-countries", nargs="*", default=[], help="Countries left out of the merged table.")
    p.add_argument("--manifests", nargs="*", default=[], help="dataset_manifest.csv files to merge as well.")
    p.add_argument("--manifest-out", default=None, help="Default: <output dir>/../dataset_manifest.csv")
    args = p.parse_args()
    merge_tables(args.parts, args.output_dir, args.crop, args.drop_countries)
    if args.manifests:
        merge_manifests(args.manifests, args.manifest_out or os.path.join(os.path.dirname(os.path.abspath(args.output_dir)), "dataset_manifest.csv"))


if __name__ == "__main__":
    main()
