"""Fixed token vocabulary for the crop TNP.

`MODALITIES` defines the `modality_id` of every variable (its row in the
model's `nn.Embedding` table). The order is part of the model: never
reorder or remove entries, only append -- a checkpoint trained with one
order is meaningless with another.
"""

MODALITIES = [
    # weather (time series)
    "tmin",
    "tmax",
    "precip",
    "radiation",
    "wind",
    "humidity",
    # soil / terrain (static, once per point)
    "clay",
    "nitrogen",
    "ph",
    "soc",
    "water_holding_capacity",
    "rooting_depth",
    "elevation",
    "slope",
    # labels (per point-year)
    "yield",
    "phenology_flowering",
    "phenology_maturity",
]
MODALITY_ID = {name: i for i, name in enumerate(MODALITIES)}
NUM_MODALITIES = len(MODALITIES)

# Weather variables, in the column order of the store's weather array, with
# the function used to aggregate daily values into a bucket. Means for
# state-like variables, sums for fluxes.
WEATHER_AGGREGATION = {
    "tmin": "mean",  # degC
    "tmax": "mean",  # degC
    "precip": "sum",  # mm
    "radiation": "sum",  # MJ/m2
    "wind": "mean",  # m/s at 2 m
    "humidity": "mean",  # vapour pressure, hPa
}
WEATHER_VARIABLES = list(WEATHER_AGGREGATION)

# Static variables available in the store. Layered soil variables carry one
# column per depth (`{name}_{i}`), with the layer mid-depth in cm as the
# token's depth coordinate. `rooting_depth` is in the vocabulary but not in
# the store: the WOFOST runs use the same maximum rooting depth everywhere,
# so it carries no per-point information yet.
SOIL_LAYER_MID_DEPTHS_CM = [2.5, 10.0, 22.5]  # CropFM zarr soil: 0-5, 5-15, 15-30 cm
LAYERED_STATIC_VARIABLES = ["clay", "nitrogen", "ph", "soc"]
SCALAR_STATIC_VARIABLES = ["water_holding_capacity", "elevation", "slope"]

# Label variables and the store (seasons.csv) column holding each value.
LABEL_COLUMNS = {
    "yield": "yield_t_per_ha",
    "phenology_flowering": "flowering_days",  # days after the season start
    "phenology_maturity": "maturity_days",  # days after the season start
}

# Reference epoch of the continuous time coordinate `t` (days since).
TIME_REFERENCE_DATE = "2000-01-01"
