import datetime
import random
from typing import Optional, Union


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
    return sowing_date_from_offset(year, anchor_doy, offset)


def sowing_date_from_offset(year: int, anchor_doy: int, offset_days: int) -> datetime.date:
    """Sowing date `offset_days` after (negative: before) the start-of-season
    anchor day-of-year in `year`."""
    return datetime.date(year, 1, 1) + datetime.timedelta(days=anchor_doy - 1 + offset_days)


def build_agromanagement(
    crop_name: str,
    variety_name: str,
    sowing_date: Union[str, datetime.date],
    max_duration: int,
    campaign_start: Optional[datetime.date] = None,
    campaign_end: Optional[datetime.date] = None,
) -> list:
    """Build the PCSE agromanagement structure for a single-campaign run:
    sowing on `sowing_date`, crop end at maturity, no fertilization events
    (water-limited, no-N config -- wofost_synthetic_pretraining_plan decision
    #13 dropped SNOMIN entirely).

    `campaign_start` (default: the sowing date) lets the simulation start
    before sowing: the soil water balance then runs on bare soil until the
    crop is sown. `campaign_end` adds a trailing empty campaign, so the
    simulation (soil water balance included) goes on after maturity until
    that date instead of stopping at maturity.
    """
    if isinstance(sowing_date, str):
        sowing_date = datetime.date.fromisoformat(sowing_date)
    campaign_start = campaign_start if campaign_start is not None else sowing_date
    if campaign_start > sowing_date:
        raise ValueError(f"campaign_start ({campaign_start}) is after the sowing date ({sowing_date}).")

    agromanagement = [
        {
            campaign_start: {
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
    if campaign_end is not None:
        if campaign_end <= sowing_date + datetime.timedelta(days=max_duration):
            raise ValueError(f"campaign_end ({campaign_end}) must be after the latest crop end "
                             f"(sowing + max_duration = {sowing_date + datetime.timedelta(days=max_duration)}).")
        agromanagement.append({campaign_end: None})
    return agromanagement
