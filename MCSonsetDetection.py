#!/usr/bin/env my_env

""" Detect cold pools of passing MCSs. """

import numpy as np
import pandas as pd
import xarray as xr
import os

import MCSonsetDetection_single
import DCSindexExtraction

from argparse import ArgumentParser


def main():
    parser = ArgumentParser()

    parser.add_argument(
        "-p",
        "--PIRATAdir",
        help="Path to PIRATA directory with air temperature cdf files",
    )
    parser.add_argument(
        "-t",
        "--TOOCANdir",
        help="Path to TOOCAN directory",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=".",
        help="Path to output directory",
    )

    args = vars(parser.parse_args())

    buoys = [
        (4, -38),
        (8, -38),
        (12, -38),
        (0, -23),
        (4, -23),
        (12, -23),
    ]
    df = DCSnearBuoydf(args["TOOCANdir"], buoys)
    datt = loadPIRATAdata(args["PIRATAdir"])

    df = MCSonsetDetectiondf(datt, df)
    df.to_csv(args["output"], index=False)


def loadPIRATAdata(PIRATAfdir):
    datt = xr.open_mfdataset(
        f"{PIRATAfdir}/airt*_10m.cdf", combine="by_coords", engine="netcdf4"
    )
    # Apply quality filter
    datt["AT_21"] = (
        datt["AT_21"]
        .where((datt["QAT_5021"] != 4) & (datt["QAT_5021"] != 5), np.nan)
        .sel(depth=-3)
    )
    datt = datt.drop_vars(["QAT_5021", "SAT_6021"])
    return datt

def DCSnearBuoydf(TOOCANdir, buoys):
    dataframes = []

    for filename in os.listdir(TOOCANdir):
        if filename.endswith(".nc"):
            filepath = os.path.join(TOOCANdir, filename)
            data = DCSindexExtraction.loadDataset(filepath)
            df_month = DCSindexExtraction.dataframe(data, buoys)
            dataframes.append(df_month)

    df = pd.concat(dataframes, ignore_index=True).drop_duplicates('DCS')

    return df


def MCSonsetDetectiondf(datt, df, negative_lon=False, init_thld=-0.2, deltaT_thld=1):
    """
    Apply cold pool onset detection on DCSs in dataframe

    datt: xarray Dataset of air temperature data
        with dimesnions time, lat, lon
        (if it also has depth dimension drop it using `datt.sel(depth=-3)`)
        and variable 'AT_21'
    df: pandas Dataframe
    negative_lon: boolean, default: False
        if True lon coordinates are in negative degrees east (e.g. -23)
        if False lon coordinates are in positive degrees east (e.g. 360-23 = 337)
    init_thld: Initial temperature difference threshold (default: -0.2K)
    deltaT_thld: Minimum temperature drop threshold (default: 1K)

    Returns:
    pandas Dataframe with tmax and tmin timestamps
    """
    tmax, tmin, tend = [], [], []

    for index, row in df.iterrows():
        print(f"DCS number: {row['DCS']} at {abs(row['lon'])}°W {row['lat']}°N")
        t_Init, t_End = pd.to_datetime(row["DCS_Init"]), pd.to_datetime(row["DCS_End"])
        if negative_lon:
            lon = row["lon"]
        else:
            lon = 360 - abs(row["lon"])
        t_max, t_min = MCSonsetDetection_single.coldpooltimes(
            datt.sel(lat=row["lat"], lon=lon),
            t_Init,
            t_End,
            pd.to_datetime(row["DCS_Time_Closest"]),
            init_thld=init_thld,
            deltaT_thld=deltaT_thld,
            # init_thld=-0.1, deltaT_thld = 0.5
        )
        tmax.append(t_max)
        tmin.append(t_min)
        # tend.append(t_end)

    # tmax, tmin, tend in neue Spalte vom df schreiben
    df["tmax"] = tmax
    df["tmin"] = tmin
    # df['tend'] = tend

    return df


if __name__ == "__main__":
    main()
