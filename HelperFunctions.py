#!/usr/bin/env my_env
# %%
import numpy as np
import pandas as pd
import xarray as xr


def get_sample_size(da_rescaled):
    return da_rescaled.dropna("DCS", how="all").DCS.size


def get_StandardError(da_rescaled):
    return da_rescaled.std("DCS") / np.sqrt(get_sample_size(da_rescaled))


def CP_df(df):
    return df[np.invert(np.isnat(df.tmax))]


def DCS_by_buoy(df, lat, lon):
    # get the list of DCS indicies for a specific buoy
    df_cp = df[np.invert(np.isnat(df.tmax))]
    return df_cp[(df_cp.lat == lat) & (df_cp.lon == lon)].DCS.values


def DCS_by_basin(df, lon):
    # get the list of DCS indicies for a specific buoy
    df_cp = df[np.invert(np.isnat(df.tmax))]
    return df_cp[(df_cp.lon == lon)].DCS.values


def LWSEmean_week(datLWSE, dur_thld=6):
    return datLWSE.field.sel(duration_thld=dur_thld).groupby("time.weekofyear").mean()


def weekly_LWSEmean_buoy(df, da_mean_week, lat, lon):
    DCS_buoy = DCS_by_buoy(df, lat, lon)
    cp_week_dist = df[(df["DCS"].isin(DCS_buoy))].tmax.dt.isocalendar().week
    cp_week = cp_week_dist.value_counts().sort_index()

    LWSE_week = da_mean_week.sel(lat=lat, lon=(360 - abs(lon))).values
    cp_week = cp_week.reindex(range(1, 54), fill_value=0).to_numpy()

    return LWSE_week, cp_week


def get_seasonal_avg(da_rescaled, df, mean_week, buoys):
    # get number of cold pool events with an adjacent LWSE
    # n = get_sample_size(da_rescaled.where(da_rescaled.sum("DCS") > 0, drop=True))
    n = get_sample_size(
        da_rescaled.where(da_rescaled.sum("time_rescaled") > 0, drop=True)
    )
    # print(n)
    n_buoy = []
    seasonal_avg_buoy = []
    for lat_buoy, lon_buoy in buoys:
        dcs_buoy = DCS_by_buoy(df, lat=lat_buoy, lon=lon_buoy)
        dcs_buoy = np.intersect1d(da_rescaled.DCS.values, dcs_buoy)
        da_rescaled_buoy = da_rescaled.sel(DCS=dcs_buoy)
        # seasonal average (weighted by cold-pool seasonality)
        wLWSE_rate, wCP_rate = weekly_LWSEmean_buoy(df, mean_week, lat_buoy, lon_buoy)
        # seasonal_avg = np.average(wLWSE_rate, weights=wCP_rate)
        seasonal_avg_buoy.append(np.nansum(wLWSE_rate * wCP_rate) / np.nansum(wCP_rate))
        # get sample size of cold pool events where low wind speed events occur within the interval
        # n_buoy.append(get_sample_size(da_rescaled_buoy.where(da_rescaled.sum("DCS") > 0, drop=True)) / n)
        n_buoy.append(
            get_sample_size(
                da_rescaled_buoy.where(da_rescaled.sum("time_rescaled") > 0, drop=True)
            )
            / n
        )

    # print(sum(n_buoy))
    seasonal_avg = np.average(seasonal_avg_buoy, weights=n_buoy)
    return seasonal_avg
