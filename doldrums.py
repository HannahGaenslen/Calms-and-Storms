#!/usr/bin/env my_env
# %%
import numpy as np
import xarray as xr
from scipy.signal import butter, filtfilt

from argparse import ArgumentParser


# %%
def main():
    parser = ArgumentParser()

    parser.add_argument("FILE", help="path to PIRATA wind speed netcdf file")
    parser.add_argument(
        "-o",
        "--output_file",
        default=".",
        help="give path to output file",
    )
    parser.add_argument(
        "-r",
        "--resolution",
        default="1/6",
        help="Time resolution of the data in hours",
    )

    args = vars(parser.parse_args())

    ds_in = load_data(args["FILE"])
    print("Calculating low wind speed events")
    time_res = fraction(args["resolution"])

    ds_out = create_LWSE_dataset(
        ds_in.WS_401, time_res=time_res
    )
    print("Writing output to: ", args["output_file"])
    ds_out.to_netcdf(args["output_file"], engine="netcdf4")


# %%
def fraction(fraction):
    if "/" in fraction:
        a, b = fraction.split("/")
        value = int(a) / int(b)
    else:
        value = float(fraction)
    return value


# %%
def load_data(file):
    ds = xr.open_dataset(file, engine="netcdf4")

    # Qulity masking:
    ds["WS_401"] = ds["WS_401"].where(
        (ds["QWS_5401"] != 4) & (ds["QWS_5401"] != 5), np.nan
    )
    return ds.sel(depth=-4, drop=True)


# %%
# Function for a Butterworth-Lowpass filter
def butter_lowpass(cutoff, fs=1 / 600, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype="lowpass", analog=False)
    return b, a


def apply_lowpass_filter(data, cutoff_freq, fs, order=5):
    b, a = butter_lowpass(cutoff_freq, fs, order=order)
    y = filtfilt(b, a, data)
    return y


def create_filtered_Wind_Speed(
    da_ws, fs=(1 / 600), cutoff_freq=(1 / (2 * 3600)), order=4
):
    """
    Applies a Butterworth lowpass filter to wind speed data and returns a filtered DataArray.

    This function removes high-frequency fluctuations from the wind speed time series using a lowpass filter,
    then fills missing time steps and adds latitude and longitude as dimensions.

    Parameters:
    da_ws : xarray.DataArray
        Wind speed data with a 'time' dimension.
    fs : float, optional
        Sampling frequency in Hz (default is 1/600 for 10-minute data).
    cutoff_freq : float, optional
        Cutoff frequency for the lowpass filter (default is 1/(2*3600) for 2-hour cutoff).
    order : int, optional
        Order of the Butterworth filter (default is 4).

    Returns:
    xarray.DataArray
        Filtered wind speed data with 'time', 'lat', and 'lon' dimensions, named "WS_filtered".
    """
    lat_b = int(da_ws.lat.isel(lat=0).values)
    lon_b = int(da_ws.lon.isel(lon=0).values)
    da_ws = da_ws.isel(lat=0, lon=0)
    ws_lowpass = xr.DataArray(
        apply_lowpass_filter(
            da_ws.dropna(dim="time").values, cutoff_freq, fs, order=order
        ),
        coords={"time": da_ws.dropna(dim="time").time.values},
        dims=["time"],
        name="WS_filtered",
        attrs=dict(description=f"filtered wind speed with cut-off-window of {1/(cutoff_freq*3600)}h ", units="m s-1")
    )
    ws_lowpass_filled = ws_lowpass.reindex(time=da_ws.time.values)
    ws_lowpass_filled = ws_lowpass_filled.expand_dims({"lat": [lat_b], "lon": [lon_b]})
    return ws_lowpass_filled


# %%
def LWSE_field(da_duration, da_ws, dur_thresh):
    # filter low wind speed events by duration and create a binary map with all events > duration threshold
    field = xr.where(da_duration.isnull(), np.nan, xr.where(da_duration > dur_thresh, 1, 0))
    return xr.DataArray(
        field.where(da_ws.notnull()),
        name="field",
        attrs=dict(description="Low wind speed event field. 1 if in event, 0 otherwise.")
    )


def start_field(da_field):
    # convert the low wind speed event field to a lwse start field
    start_mask = (da_field - da_field.shift(time=1)) == 1
    return xr.DataArray(
        xr.where(da_field.isnull(), np.nan, xr.where(start_mask, 1, 0)),
        name="start",
        attrs=dict(description="Low wind speed event start field. 1 if start of an event, 0 otherwise.")
    )


def end_field(da_field):
    # convert the low wind speed event field to a lwse end field
    end_mask = (da_field - da_field.shift(time=-1)) == 1
    return xr.DataArray(
        xr.where(da_field.isnull(), np.nan, xr.where(end_mask, 1, 0)),
        name="end",
        attrs=dict(description="Low wind speed event end field. 1 if end of an event, 0 otherwise.")
    )


# %%
def LWSE_mask(da, ws_thresh=3):
    """
    Creates mask for low wind speed

    Parameters:
    da: xarray.DataArray, wind speed data with time dimension.
    ws_thresh: float, threshold for low wind speed (default is 3 m/s).

    Returns:
    xarray.DataArray, Binary mask (1 for low wind speed, 0 otherwise) with a 'time' dimension.
    """
    da_mask = xr.where(da <= ws_thresh, 1, 0)
    return da_mask.rename("LWSE_mask")


def LWSE_count(da_mask):
    """
    Calculates the event count for low wind speed events.

    For each time step, assigns a unique integer to each contiguous sequence of low wind speed (where da_mask == 1).
    This allows grouping and analysis of individual low wind speed events.

    Parameters:
    da_mask : xarray.DataArray
        Binary mask (1 for low wind speed, 0 otherwise) with a 'time' dimension.

    Returns:
    xarray.DataArray
        DataArray with the same shape as da_mask, where each low wind speed event is labeled by a unique integer.
    """
    da_count = (((da_mask != da_mask.shift(time=1)).cumsum() + 1) * da_mask) // 2
    return da_count.rename("LWSE_count")


def LWSE_duration(da_mask, da_count, time_res=1 / 6):
    """
    Calculates the event duration for low wind speed events.

    For each time step, assigns duration (h) to each contiguous sequence of low wind speed.

    Parameters:
    da_mask : xarray.DataArray
        Binary mask (1 for low wind speed, 0 otherwise) with a 'time' dimension.
    da_count: xarray.DataArray
        DataArray with the same shape as da_mask, where each low wind speed event is labeled by a unique integer.
    time_res: float
        time resolution of the data in hours (default is 10 minutes)

    Returns:
    xarray.DataArray
    """
    # this works only by buoy
    # group low wind speed events by count
    gb = da_mask.groupby(da_count)
    # compute the duration of each low wind speed event (by using xarray count function)
    # gb + gb.count(), adds the count (length of array with ones) for each group to all elements in that group.
    da_duration = ((gb + gb.count(...)) - 1) * da_count * time_res
    da_duration = da_duration.reset_coords("LWSE_count", drop=True)
    return da_duration.rename("LWSE_duration")


def create_LWSE_dataset_buoy(da_ws, ws_thresh=3, dur_thresh=6, time_res=1 / 6):
    da_mask = LWSE_mask(da_ws, ws_thresh)
    da_count = LWSE_count(da_mask)
    da_duration = LWSE_duration(da_mask, da_count, time_res)

    da_LWSE_field = LWSE_field(da_duration, da_ws, dur_thresh)
    da_LWSE_start = start_field(da_LWSE_field)
    da_LWSE_end = end_field(da_LWSE_field)

    datLWSE = xr.Dataset(
        {
            "WS": da_ws,
            "field": da_LWSE_field,
            "start": da_LWSE_start,
            "end": da_LWSE_end,
        }
    )
    return datLWSE


# %%
def get_buoys(ds):
    buoys = np.array(np.meshgrid(ds.lat.values, ds.lon.values)).T.reshape(-1, 2)
    new_buoys = []
    for i in range(len(buoys)):
        if ds.sel(lat=buoys[i][0], lon=buoys[i][1]).dropna(dim="time").time.size == 0:
            print(buoys[i])
        else:
            new_buoys.append(buoys[i])
    return new_buoys


# %%
def create_LWSE_dataset(
    da_ws,
    filter=True,
    ws_thresh=3,
    dur_thresh=6,
    time_res=1 / 6,
    fs=(1 / 600),
    cutoff_freq=(1 / (2 * 3600)),
    order=4,
):
    """
    Creates a dataset with (filtered) wind speed, low wind speed event field, start and end.

    Parameters:
    da_ws : xarray.DataArray 
        wind speed data with dimensions lat, lon, time
    """
    buoys = get_buoys(da_ws)
    ds_buoys = []
    for lat, lon in buoys:
        print(f"{lon}°E {lat}°N")
        ws = da_ws.sel(lat=[lat], lon=[lon])
        print(ws)
        if filter:
            ws = create_filtered_Wind_Speed(ws, fs, cutoff_freq, order)
        ds_buoys.append(create_LWSE_dataset_buoy(ws, ws_thresh, dur_thresh, time_res))
        print()
    return xr.merge(ds_buoys)


# %%
if __name__ == "__main__":
    main()

# %%
