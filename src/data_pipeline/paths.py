"""Where the pipeline's data lives.

Downloads (weather, soil), simulations, processed tables and reports go to the project space, not the repo:
home is nearly full. `$TNP_DATA_ROOT` overrides it. Small inputs that the docs and scripts refer to by repo
path (`data/raw/locations`, `data/crop_parameters`) stay in the repo.
"""

import os

DATA_ROOT = os.environ.get("TNP_DATA_ROOT", "/projects/prjs1788/tnp-crop/data")
RAW_DIR = os.path.join(DATA_ROOT, "raw")
PROCESSED_DIR = os.path.join(DATA_ROOT, "processed")
REPORTS_DIR = os.path.join(DATA_ROOT, "reports")
