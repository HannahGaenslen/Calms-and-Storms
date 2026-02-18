#!/usr/bin/env my_env

"""
Extracts DCSs within a box around buoy. 
"""

import pandas as pd
import xarray as xr
import math

from argparse import ArgumentParser


def main():
    parser = ArgumentParser()

    parser.add_argument("--cdf", help="Path to TOOCAN cdf file")
    parser.add_argument(
        "-o",
        "--output_dir",
        help="Path to csv file",
    )

    args = vars(parser.parse_args())

    data = loadDataset(args["cdf"])
    buoys = [
        (4, -38),
        (8, -38),
        (12, -38),
        (0, -23),
        (4, -23),
        (12, -23),
    ]
    df = dataframe(data, buoys)
    df.to_csv(args["output_dir"])


def loadDataset(cdfpath, latlim=(-1, 15), lonlim=(-45, -20)):
    """ Load TOOCAN dataset in selected region and apply quality control. """
    data = xr.open_dataset(cdfpath, engine="netcdf4")

    # quality control and lat-lon-masking
    filter_DCS_qualitycontrol = data.INT_DCS_qualitycontrol < 11110
    mask_lat = (data.INT_latmax <= latlim[1]) & (data.INT_latmin >= latlim[0])
    mask_lon = (data.INT_lonmax >= lonlim[0]) & (data.INT_lonmin <= lonlim[1])

    dcs_itcz = data.DCS.where(
        filter_DCS_qualitycontrol & mask_lat & mask_lon, drop=True
    )

    return data.sel(DCS=dcs_itcz)


def DCSnearBuoy(lat_b, lon_b, data):
    """ Extract index of near buoy DCS from TOOCAN data """
    # MCSs are defined by 100km horizontal extend. 1° is 110km at the equator. 
    # The margin is set to a 100km box around the buoy at lat_b, lon_b.
    margin = 0.89831924 / 2

    mask_lat = (data.INT_latmax <= (lat_b + margin)) & (
        data.INT_latmin >= (lat_b - margin)
    )
    mask_lon = (data.INT_lonmax >= (lon_b - margin)) & (
        data.INT_lonmin <= (lon_b + margin)
    )

    dcs_buoy = data.DCS.where(mask_lat.load() & mask_lon.load(), drop=True)
    return dcs_buoy.values


def TimeDCSclosestBuoy(dcs_buoy, lat_b, lon_b, data):
    dcs_buoy_dist = []
    dcs_buoy_time = []
    for dcs in dcs_buoy:
        DCS_init = data.INT_UTC_timeInit.sel(DCS=dcs).values
        DCS_end = data.INT_UTC_timeEnd.sel(DCS=dcs).values
        print(f"DCS: {dcs}, Init: {DCS_init}, End: {DCS_end}")
        DCS_lat = data.LC_lat.sel(DCS=dcs, time=slice(DCS_init, DCS_end)).values
        DCS_lon = data.LC_lon.sel(DCS=dcs, time=slice(DCS_init, DCS_end)).values

        DCS_dist = []
        for DCS_x, DCS_y, t in zip(
            DCS_lon,
            DCS_lat,
            data.sel(DCS=dcs, time=slice(DCS_init, DCS_end)).time.values,
        ):
            # print(f'DCS: {DCS_x}, {DCS_y}, {t}')
            DCS_dist.append((math.dist([DCS_x, DCS_y], [lon_b, lat_b]), t))

        # Get the first element (nearest to the buoy)
        print(sorted(DCS_dist, key=lambda x: x[0])[0])
        dcs_buoy_dist.append(sorted(DCS_dist, key=lambda x: x[0])[0][0])  # distance
        dcs_buoy_time.append(sorted(DCS_dist, key=lambda x: x[0])[0][1])  # time

    return dcs_buoy_dist, dcs_buoy_time

def timeSurfaceMax(data, dcs):
    data_dcs_i = data.sel(DCS=dcs)
    t_maxsurf = data_dcs_i.LC_UTC_time.where(data_dcs_i.LC_surfkm2_235K == data_dcs_i.INT_surfmaxkm2_235K, drop=True)
    return t_maxsurf

def dataframe(data, buoys):
    """
    Creates a dataframe holding information on DCSs close to buoys.
    
    data: TOOCAN dataset
    buoys: list of tuples (latitude, longitude)
    """
    df = pd.DataFrame()

    for lat_b, lon_b in buoys:
        dcs_buoy = DCSnearBuoy(lat_b, lon_b, data)
        distance_closest, time_closest = TimeDCSclosestBuoy(
            dcs_buoy, lat_b, lon_b, data
        )
        t_surfmax = []
        for dcs in dcs_buoy:
            t_surfmax.append(timeSurfaceMax(data, dcs).values[0])

        d = {
            "lat": lat_b,
            "lon": lon_b,
            "DCS": dcs_buoy,
            "DCS_Init": data.INT_UTC_timeInit.sel(DCS=dcs_buoy).values,
            "DCS_End": data.INT_UTC_timeEnd.sel(DCS=dcs_buoy).values,
            "DCS_Duration": data.INT_duration.sel(DCS=dcs_buoy).values,
            "DCS_Time_Closest": time_closest,
            "DCS_Distance_Closest": distance_closest,
            "Time_Surface_Max_235K": t_surfmax,
            "Classification": data.INT_classif.sel(DCS=dcs_buoy).values,
            "BT_Min": data.INT_tbmin.sel(DCS=dcs_buoy).values,
            "Surface_Max_235K": data.INT_surfmaxkm2_235K.sel(DCS=dcs_buoy).values,
            "Surface_Max_220K": data.INT_surfmaxkm2_220K.sel(DCS=dcs_buoy).values,
            "Surface_Max_210K": data.INT_surfmaxkm2_210K.sel(DCS=dcs_buoy).values,
            "Surface_Max_200K": data.INT_surfmaxkm2_200K.sel(DCS=dcs_buoy).values,
        }

        df_b = pd.DataFrame(data=d)
        df = pd.concat([df, df_b], ignore_index=True)

    return df


if __name__ == "__main__":
    main()
