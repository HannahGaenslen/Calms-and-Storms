#!/usr/bin/env my_env
# %%
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

import LoadData

from argparse import ArgumentParser

def main():
    parser = ArgumentParser()

    parser.add_argument(
        "-p",
        "--PIRATAdir",
        help="Path to PIRATA directory with air temperature cdf files",
    )
    parser.add_argument(
        "-c",
        "--ColdPoolDir",
        help="Path to ColdPool directory",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=".",
        help="Path to output directory",
    )

    args = vars(parser.parse_args())
 
    data = LoadData.loadPIRATAdata(args["PIRATAdir"])
    df = LoadData.loadColdPools(args["ColdPoolDir"])

    data_rescaled = composites(data, df)
    data_rescaled.to_netcdf(args["output"])

# %%
## Rescaling

def mean_coldpool_duration(df):
    """
    Calculate the mean cold pool duration
    """
    CP_duration = []
    for index, row in df.iterrows():
        # duration = row['tend'] - row['tmax']
        duration = row.tmin - row.tmax
        CP_duration.append(duration)
    CP_duration = pd.to_timedelta(CP_duration, unit="h")
    mean_CP_duration = CP_duration.mean()
    return mean_CP_duration

def median_coldpool_duration(df):
    """
    Calculate the mean cold pool duration
    """
    CP_duration = []
    for index, row in df.iterrows():
        # duration = row['tend'] - row['tmax']
        duration = row.tmin - row.tmax
        CP_duration.append(duration)
    CP_duration = pd.to_timedelta(CP_duration, unit="h")
    median_CP_duration = CP_duration.median()
    return median_CP_duration


# %%
def rescale(ds, start, end, mduration):
    """
    Rescale PIRATA time series for each DCS event.
    (Original function by Julia Windmiller)

    Parameters:
        ds (xarray.DataSet): PIRATA data with time, lat, lon dimensions.
        start (pd.Timestamp): Start time of the DCS event.
        end (pd.Timestamp): End time of the DCS event.
        mduration (pd.Timedelta): Mean duration of the DCS induced cold pool.

    Returns:
        time_shifted, rescaled_time (xarray.DataArray): Rescaled time series.
    """

    timestep = pd.Timedelta(minutes=10)

    # shift time to onset time:
    time_shifted = ds.time - ds.time.sel(time=start)
    rescaled_time = time_shifted.copy().astype("<m8[ns]")

    # rescale time during the DCS event to values between 0 and 1
    rescaled_time_duringDCS = time_shifted.sel(
        time=slice(start, end)
    ) / time_shifted.sel(time=end)
    # multiply by the mean duration of the DCS induced cold pool
    rescaled_time_duringDCS = rescaled_time_duringDCS * (mduration - timestep)
    # print(rescaled_time_duringDCS, len(rescaled_time_duringDCS))

    # overwrite values during the DCS event with rescaled values
    rescaled_time.loc[dict(time=slice(start, end))][:] = rescaled_time_duringDCS

    # in case the DCS event is at the end of the time series, we need to adjust the time
    if (time_shifted.time.isel(time=-1) >= (end + timestep)).values:
        adjusted_time_after_DCS = (
            time_shifted.sel(time=slice(end + timestep, None))
            - time_shifted.sel(time=end + timestep)
        ) + mduration

        rescaled_time.loc[dict(time=slice(end + timestep, None))][:] = (
            adjusted_time_after_DCS
        )

    return rescaled_time


def round_up_to_10min(td):
    return ((td + pd.Timedelta(9, "m")) // pd.Timedelta(10, "m")) * pd.Timedelta(
        10, "m"
    )


# %%
def plot_rescaled(ds, parameter, df, DCS):
    mduration = median_coldpool_duration(df).round("min")
    start = df["tmax"][df["DCS"] == DCS].values[0]
    end = df["tmin"][df["DCS"] == DCS].values[0]
    interval = slice(start - pd.Timedelta(12, "h"), end + pd.Timedelta(12, "h"))
    rescaled_time = rescale(ds, start, end, mduration)

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, sharey=True)

    interval = slice(start - pd.Timedelta(hours=6), end + pd.Timedelta(hours=6))

    ax1.plot(
        round_up_to_10min(rescaled_time).sel(time=interval).values
        / np.timedelta64(1, "h"),
        ds[parameter].sel(
            time=interval,
            lat=df["lat"][df["DCS"] == DCS].iloc[0],
            lon=360 - abs(df["lon"][df["DCS"] == DCS].iloc[0]),
        ),
        label="Rescaled",
    )
    ax1.set_xlabel("Rescaled Time (h)")
    ax1.axvline(x=0, color="black", linestyle="--", label="tmax")
    ax1.axvline(
        x=mduration / np.timedelta64(1, "h"),
        color="black",
        linestyle="--",
        label="tmin",
    )

    ds[parameter].sel(
        lat=df["lat"][df["DCS"] == DCS].iloc[0],
        lon=360 - abs(df["lon"][df["DCS"] == DCS].iloc[0]),
        time=slice(start - pd.Timedelta(6, "h"), end + pd.Timedelta(6, "h")),
    ).plot(ax=ax2, label="Original")
    ax2.axvline(x=start, color="black", linestyle="--", label="tmax")
    ax2.axvline(x=end, color="black", linestyle="--", label="tmin")

    plt.xticks(rotation=45)
    ax1.set_ylim(23, 29)
    # ax2.set_ylim(23, 29)

    # plt.ylim(23, 30)
    plt.title(
        f"DSC {DCS} {df['lat'][df['DCS'] == DCS].iloc[0]}°N {df['lon'][df['DCS'] == DCS].iloc[0]}°E"
    )
    ax1.grid()
    ax2.grid()
    plt.show()


# %%
## Composites
def composites(ds, df, negative_lon=False, interval=pd.Timedelta(12, "h")):  # parameter
    # dfs= []
    dataarrays = []
    for index, row in df.iterrows():
        if row["tmax"] is pd.NaT or row["tmin"] is pd.NaT:
            continue  # Skip if tmax or tend is not available
        else:
            print("DCS: ", row["DCS"], "tmax:", row["tmax"], "tmin:", row["tmin"])

            # Extract event-specific times
            start = pd.to_datetime(row["tmax"])
            end = pd.to_datetime(row["tmin"])
            mduration = median_coldpool_duration(df).round("min")

            rescaled_time = rescale(ds, start, end, round_up_to_10min(mduration))
            ds = ds.assign_coords(
                time_rescaled=("time", rescaled_time.data.astype("timedelta64[m]"))
            )

            if negative_lon:
                lon = row["lon"]
            else:
                lon = 360 - abs(row["lon"])

            # ds_slice = ds[parameter].sel(
            if negative_lon:
                lon = row["lon"]
            else:
                lon = 360 - abs(row["lon"])
                
            ds_slice = ds.sel(
                time=slice(start - interval, end + interval),
                lat=row["lat"],
                lon=lon,
            )
            # print(ds_slice)

            t_bins = pd.timedelta_range(
                start=-interval,
                end=round_up_to_10min(mduration) + interval,
                freq="10min",
            )
            gp = ds_slice.groupby_bins("time_rescaled", t_bins)
            da_CP = gp.mean(dim="time").drop_vars(["lat", "lon"])
            da_CP = da_CP.chunk({"time_rescaled_bins": len(t_bins)})
            da_CP = da_CP.assign_coords(
                time_rescaled=(
                    "time_rescaled_bins",
                    np.array(
                        [iv.left for iv in da_CP.time_rescaled_bins.values],
                        dtype="timedelta64[m]",
                    ),
                )
            )
            da_CP = da_CP.swap_dims({"time_rescaled_bins": "time_rescaled"}).drop_vars(
                "time_rescaled_bins"
            )
            da_CP = da_CP.assign_coords(DCS=row["DCS"])

            dataarrays.append(da_CP)

    # return pd.concat(dfs, axis=1)
    dat_rescaled = xr.concat(dataarrays, dim="DCS").chunk({"DCS": 1000})

    hours = dat_rescaled.time_rescaled / np.timedelta64(1, "h")
    dat_rescaled = dat_rescaled.assign_coords(hours=("time_rescaled", hours.data))
    return dat_rescaled

# %%
if __name__ == "__main__":
    main()
