import requests
import pandas as pd

from src.data_pipeline.soil.utils_soil.default_soil_variables import (
    default_soilgrid_variables,
    default_zs,
)


request_url = "https://rest.isric.org/soilgrids/v2.0/properties/query"


def request_soilgrids(lat, lon) -> dict:
    p1 = {"lat": lat, "lon": lon}
    props = {"property": default_soilgrid_variables(), "depth": get_depth_soilgrids()}
    res = requests.get(request_url, params={**p1, **props})
    result = res.json()

    return result


def get_depth_soilgrids() -> list:
    zmins, zmaxs = default_zs()

    depth_name_template = '{zmin}-{zmax}cm'
    depths = []
    for zmin, zmax in zip(zmins, zmaxs):
        depth = depth_name_template.format(zmin=zmin, zmax=zmax)
        depths.append(depth)

    print(f"depths = {depths}")

    return depths


def get_df_soilgrids(lat: float, lon: float) -> pd.DataFrame:
    print(f"getting soilgrids for longitude: {lon} and latitude: {lat}")
    resultd = request_soilgrids(lat, lon)

    check_value_empty(resultd)

    depths = get_depth_soilgrids()
    zmins, zmaxs = default_zs()

    variables = default_soilgrid_variables()

    soild = {}
    soild["latitude"] = []
    soild["longitude"] = []
    soild["zmin"] = []
    soild["zmax"] = []
    for i in range(0, len(depths)):
        soild["zmin"].append(zmins[i])
        soild["zmax"].append(zmaxs[i])
        soild["latitude"].append(lat)
        soild["longitude"].append(lon)
    for i, var in enumerate(variables):
        var_name = resultd['properties']["layers"][i]['name']
        if var_name in variables:
            soild[var_name] = []
            for j in range(0, len(depths)):
                raw_value = resultd['properties']["layers"][i]["depths"][j]["values"]["mean"]
                d_factor = resultd["properties"]["layers"][i]["unit_measure"]["d_factor"]
                value = raw_value / d_factor
                soild[var_name].append(value)

    df_soilgrids = pd.DataFrame.from_dict(soild)
    return df_soilgrids


def check_value_empty(data, first=True):
    if isinstance(data, dict):
        for key, value in data.items():
            if key == "mean" and value is None:
                raise ValueError(
                    f"The key 'mean' has a value of None! Soil data in "
                    f"given lat/lon might not exist in SoilGrids! "
                    f"Please retry with a different lat/lon combination."
                )
            # Recursively check nested dictionaries or lists
            check_value_empty(value)
    elif isinstance(data, list):
        for item in data:
            check_value_empty(item)
