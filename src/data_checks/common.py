"""Shared loading/output helpers for the dataset checks (`coverage_audit`,
`simulation_audit`)."""

import os
import re
from typing import Optional

import matplotlib

matplotlib.use("Agg")  # file output only, works on headless cluster nodes
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import rootutils  # noqa: E402

root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
DEFAULT_REPORTS_DIR = os.path.join(str(root), "data", "reports", "dataset_checks")


def load_manifest(manifest_path: str, locations_csv: Optional[str] = None) -> pd.DataFrame:
    """The batch run's manifest, with `country` (and the locations CSV's
    other columns, e.g. `zarr_index`) joined on `location_index` -- the row
    number of the locations CSV the batch ran on. Without a locations CSV,
    `country` is "unknown"."""
    df = pd.read_csv(manifest_path)
    if "jitter_index" not in df:
        df["jitter_index"] = 0
    df["success"] = df["status"].eq("success")
    df["sowing_date"] = pd.to_datetime(df["sowing_date"], errors="coerce")
    df["yield_t_per_ha"] = df["yield_kg_per_ha"] / 1000.0

    if locations_csv:
        locs = pd.read_csv(locations_csv).reset_index(names="location_index")
        extra = [c for c in locs.columns if c not in df.columns or c == "location_index"]
        df = df.merge(locs[extra], on="location_index", how="left")
        if "longitude" in locs:
            check = df.merge(locs[["location_index", "longitude", "latitude"]], on="location_index", suffixes=("", "_loc"))
            if not np.allclose(check["longitude"], check["longitude_loc"]) or not np.allclose(check["latitude"], check["latitude_loc"]):
                raise ValueError("Manifest coordinates don't match the locations CSV rows; is it the CSV the batch ran on?")
    if "country" not in df:
        df["country"] = "unknown"
    df["country"] = df["country"].fillna("unknown")
    return df


def error_category(error: str) -> str:
    """Error message with the variable parts (numbers, paths, coordinates)
    stripped, so identical failure causes group together."""
    if not isinstance(error, str):
        return ""
    msg = re.sub(r"/\S+", "<path>", error)
    msg = re.sub(r"-?\d+(\.\d+)?", "<n>", msg)
    return msg[:160]


def resolve_path(path, fallback_dir: Optional[str]) -> Optional[str]:
    """Manifest paths are absolute on the machine that ran the batch; fall back
    to the same file name under `fallback_dir/<crop>` if the data moved."""
    if isinstance(path, str) and os.path.exists(path):
        return path
    if fallback_dir and isinstance(path, str):
        for candidate in (os.path.join(fallback_dir, os.path.basename(path)),
                          os.path.join(fallback_dir, os.path.basename(os.path.dirname(path)), os.path.basename(path))):
            if os.path.exists(candidate):
                return candidate
    return None


class Report:
    """Collects a markdown summary, CSV tables and PNG figures in one folder."""

    def __init__(self, out_dir: str, title: str):
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self.lines = [f"# {title}", ""]

    def section(self, title: str):
        self.lines += ["", f"## {title}", ""]
        print(f"\n== {title}")

    def text(self, text: str):
        self.lines.append(text)
        print(text)

    def flag(self, ok: bool, text: str):
        self.text(f"- {'OK  ' if ok else 'WARN'} {text}")

    def table(self, df: pd.DataFrame, name: str, max_rows: int = 40, index: bool = True):
        path = os.path.join(self.out_dir, f"{name}.csv")
        df.to_csv(path, index=index)
        shown = df.head(max_rows)
        self.lines.append(shown.to_markdown(index=index) if _has_tabulate() else "```\n" + shown.to_string(index=index) + "\n```")
        if len(df) > max_rows:
            self.lines.append(f"\n({len(df)} rows, full table: `{name}.csv`)")
        print(shown.to_string(index=index))

    def figure(self, fig, name: str, caption: str = ""):
        path = os.path.join(self.out_dir, f"{name}.png")
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        self.lines += [f"![{caption or name}]({name}.png)", ""]
        print(f"[figure] {path}")

    def save(self) -> str:
        path = os.path.join(self.out_dir, "summary.md")
        with open(path, "w") as f:
            f.write("\n".join(self.lines) + "\n")
        print(f"\nReport written to {path}")
        return path


def _has_tabulate() -> bool:
    try:
        import tabulate  # noqa: F401
        return True
    except ImportError:
        return False


def default_output_dir(manifest_path: str, kind: str) -> str:
    stem = os.path.splitext(os.path.basename(os.path.dirname(os.path.abspath(manifest_path))) or "manifest")[0]
    return os.path.join(DEFAULT_REPORTS_DIR, stem, kind)
