# %%
import numpy as np
import pandas as pd
import xarray as xr

import MCScomposites
import LoadData


def main():
    pass


# %%
def buildDCSarray(df, parameter, field=False):
    # Create time index
    time = pd.date_range(start="1998-01-30", end="2018-12-31 23:50", freq="10min")

    lats = [0.0, 4.0, 8.0, 12.0]
    lons = [322, 337]

    zeros = np.zeros((len(time), len(lats), len(lons)))

    # Create DataArray
    da = xr.DataArray(
        zeros,
        coords={"time": time, "lat": lats, "lon": lons},
        dims=["time", "lat", "lon"],
        name=parameter,
    )

    buoys = [
        (-38, 4, "38°W 4°N"),
        (-38, 8, "38°W, 8°N"),
        (-38, 12, "38°W, 12°N"),
        (-23, 0, "23°W 0°N"),
        (-23, 4, "23°W, 4°N"),
        (-23, 12, "23°W, 12°N"),
    ]

    for lon, lat, _ in buoys:
        # Filter DataFrame for specific lat/lon
        df_latlon = df[df["lat"] == lat][df["lon"] == lon]
        df_latlon = df_latlon.dropna()
        if field:
            start_times = df_latlon.dropna()["tmax"]
            end_times = df_latlon.dropna()["tmin"]
            for start, end in zip(start_times, end_times):
                da.loc[dict(lat=lat, lon=360 + lon)] = xr.where(
                    (da.time >= start) & (da.time <= end),
                    1,
                    da.loc[dict(lat=lat, lon=360 + lon)],
                )
        else:
            # da.loc[dict(lat=lat, lon=360+lon)] = da.loc[dict(lat=lat, lon=360+lon)].where(
            da.loc[dict(lat=lat, lon=360 + lon)] = xr.where(
                da.sel(lat=lat, lon=360 + lon).time.isin(df_latlon[parameter]),
                1,
                da.loc[dict(lat=lat, lon=360 + lon)],
            )

    return da


# %%
# create dataframe with LWSE start and end times for each buoy
# The dataframe corresponds to the cold pool detection dataframe


def LWSE_df(datLWSE, duration_thld=6):
    buoys = [
        (-38, 4, "38°W 4°N"),
        (-38, 8, "38°W, 8°N"),
        (-38, 12, "38°W, 12°N"),
        (-23, 0, "23°W 0°N"),
        (-23, 4, "23°W, 4°N"),
        (-23, 12, "23°W, 12°N"),
    ]

    LWSE_list = []
    for lon, lat, _ in buoys:
        # Create a DataFrame for each lat/lon pair
        starts = datLWSE.start.sel(
            lat=lat, lon=360 - abs(lon), duration_thld=duration_thld
        ).compute()
        ends = datLWSE.end.sel(
            lat=lat, lon=360 - abs(lon), duration_thld=duration_thld
        ).compute()
        LWSE_list.append(
            pd.DataFrame(
                {
                    "lat": lat,
                    "lon": lon,
                    "tmax": starts[starts == 1].time.values,
                    "tmin": ends[ends == 1].time.values,
                    "DCS": np.arange(0, len(starts[starts == 1].time.values), 1),
                }
            )
        )
    return pd.concat(LWSE_list, ignore_index=True)


# %%
if __name__ == "__main__":
    # %%
    # Load cold pool dataframes
    workdir = "data/"
    df = LoadData.loadColdPools(
        f"{workdir}/coldpools",
        date_columns=[
            "DCS_Init",
            "DCS_End",
            "tmax",
            "tmin",
            "DCS_Time_Closest",
            "Time_Surface_Max_235K",
        ],
    )
    # %%
    # load LWSE datasets
    datLWSE = xr.open_dataset(f"{workdir}/Observations/datLWSE_1998-2018_lowpass.nc")
    # %%
    df_LWSE = LWSE_df(datLWSE, duration_thld=6)
    # df_LWSE.to_csv(f"{workdir}/Observations/LWSE/df_LWSE.csv")
    mLWSEduration = MCScomposites.median_coldpool_duration(df_LWSE).round("min")
