import argparse
import os
import sys
import rootutils

from src.data_pipeline.weather.utils_weather.openmeteo_weather import get_df_weather


root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
DEFAULT_WEATHER_SAVE_DIR = os.path.join(str(root), "data", "raw", "weather")


def generate_weather_file(
    longitude: float,
    latitude: float,
    start_date: str = None,
    end_date: str = None,
    openmeteo_model: str = None,
    output_dir: str = None,
) -> str:
    """
    Fetch daily meteo variables (min/max temperature, precipitation, solar
    radiation, wind speed, vapour pressure) for a location from PCSE's
    OpenMeteoWeatherDataProvider (ERA5-Land by default) and cache them as a CSV file.

    Args:
        longitude (float): Longitude value.
        latitude (float): Latitude value.
        start_date (str, optional): Earliest day to keep, e.g. "2000-01-01".
        end_date (str, optional): Latest day to keep, e.g. "2023-12-31".
        openmeteo_model (str, optional): Open-Meteo model keyword, e.g. "era5_land"
                                          (default) or "best_match".
        output_dir (str, optional): Target directory to save the CSV file.
                                     Defaults to DEFAULT_WEATHER_SAVE_DIR.

    Returns:
        str: Absolute path to the generated CSV file.
    """
    save_dir = output_dir if output_dir is not None else DEFAULT_WEATHER_SAVE_DIR
    os.makedirs(save_dir, exist_ok=True)

    df_weather = get_df_weather(
        latitude=latitude,
        longitude=longitude,
        start_date=start_date,
        end_date=end_date,
        openmeteo_model=openmeteo_model,
    )

    path_file = os.path.join(save_dir, f"weather_{longitude}_{latitude}.csv")
    df_weather.to_csv(path_file, index=False)

    print(f"CSV weather file has been created at {path_file}.")
    return path_file


def main():
    if len(sys.argv) == 1:
        print("No arguments provided!")
        print(
            "Usage: python generate_weather_file.py --lon <longitude> --lat <latitude> "
            "[--start-date YYYY-MM-DD] [--end-date YYYY-MM-DD] "
            "[--openmeteo-model era5_land] [--output-dir <path>]"
        )
        print(
            "Example: python generate_weather_file.py -lon 6.656 -lat 52.966 "
            "--start-date 2000-01-01 --end-date 2023-12-31"
        )
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Fetch and cache daily weather variables for a given longitude/latitude.")
    parser.add_argument("-lon", "--longitude", dest="longitude", type=float, required=True, help="Longitude for the weather data.")
    parser.add_argument("-lat", "--latitude", dest="latitude", type=float, required=True, help="Latitude for the weather data.")
    parser.add_argument("--start-date", dest="start_date", type=str, default=None, help="Earliest day to keep, e.g. 2000-01-01.")
    parser.add_argument("--end-date", dest="end_date", type=str, default=None, help="Latest day to keep, e.g. 2023-12-31.")
    parser.add_argument(
        "--openmeteo-model",
        dest="openmeteo_model",
        type=str,
        default=None,
        help="Open-Meteo model keyword, e.g. era5_land (default) or best_match.",
    )
    parser.add_argument("-o", "--output-dir", dest="output_dir", type=str, default=DEFAULT_WEATHER_SAVE_DIR, help="Directory to save generated weather CSV files.")

    args = parser.parse_args()

    generate_weather_file(
        longitude=args.longitude,
        latitude=args.latitude,
        start_date=args.start_date,
        end_date=args.end_date,
        openmeteo_model=args.openmeteo_model,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
