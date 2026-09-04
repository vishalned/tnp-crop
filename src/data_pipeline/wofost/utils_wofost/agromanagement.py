import datetime
import random
from typing import Union


def jitter_sowing_date(
    year: int,
    anchor_doy: int,
    jitter_days: int = 0,
    rng: random.Random = None,
) -> datetime.date:
    """Sample a sowing date around a start-of-season anchor (day-of-year),
    jittered by up to `jitter_days` in either direction.

    Per wofost_synthetic_pretraining_plan step 3: sowing date is
    "WorldCereal SOS +/- ~1-2 weeks jitter, not free-random" -- keeps things
    agronomically plausible instead of sampling a sowing date uniformly at
    random.
    """
    rng = rng if rng is not None else random
    offset = rng.randint(-jitter_days, jitter_days) if jitter_days > 0 else 0
    return datetime.date(year, 1, 1) + datetime.timedelta(days=anchor_doy - 1 + offset)


def build_agromanagement(
    crop_name: str,
    variety_name: str,
    sowing_date: Union[str, datetime.date],
    max_duration: int,
) -> list:
    """Build the PCSE agromanagement structure for a single-campaign run:
    sowing on `sowing_date`, crop end at maturity, no fertilization events
    (water-limited, no-N config -- wofost_synthetic_pretraining_plan decision
    #13 dropped SNOMIN entirely).
    """
    if isinstance(sowing_date, str):
        sowing_date = datetime.date.fromisoformat(sowing_date)

    return [
        {
            sowing_date: {
                "CropCalendar": {
                    "crop_name": crop_name,
                    "variety_name": variety_name,
                    "crop_start_date": sowing_date,
                    "crop_start_type": "sowing",
                    "crop_end_date": None,
                    "crop_end_type": "maturity",
                    "max_duration": max_duration,
                },
                "TimedEvents": None,
                "StateEvents": None,
            }
        }
    ]
