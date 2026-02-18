#!/usr/bin/env my_env
# %%
import numpy as np
import pandas as pd
import xarray as xr
import os

from argparse import ArgumentParser


def main():
    parser = ArgumentParser()

    parser.add_argument(
        "-p",
        "--PIRATAdir",
        help="Path to PIRATA directory with air temperature cdf files",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=".",
        help="Path to output directory",
    )

    args = vars(parser.parse_args())

    data = loadPIRATAdata(args["PIRATAdir"])
    data.to_netcdf(args["output"])


def loadPIRATAvariable(PIRATAdir, name, var_name, qvar_name):
    ds = xr.open_mfdataset(
        f"{PIRATAdir}/{name}*_10m.cdf",
        combine="by_coords",
        engine="netcdf4",
    )
    ds[var_name] = ds[var_name].where(
        (ds[qvar_name] != 4) & (ds[qvar_name] != 5), np.nan
    )
    return ds[var_name].isel(depth=0).drop_vars("depth")


def specificHumidity(RH, T, P):
    """
    Calculates specific humidity
    RH: relative humidity in %
    T: air temperature in °C
    P: pressure in hPa
    """
    # saturated water pressure in hPa
    e_sat = 6.1094 * np.exp((17.062 * T) / (T + 243.04))
    # saturated mixing ratio in g/kg
    w_s = 621.97 * e_sat / (P - e_sat)
    # mixing ratio in kg/kg
    w = w_s * RH / 100 * 0.001
    # returns specific humidity in g/kg
    return w / (1 + w) * 1000


def loadPIRATAdata(PIRATAdir):
    datt = loadPIRATAvariable(PIRATAdir, "airt", "AT_21", "QAT_5021")
    dath = loadPIRATAvariable(PIRATAdir, "rh", "RH_910", "QRH_5910")
    datr = loadPIRATAvariable(PIRATAdir, "rain", "RN_485", "QRN_5485")
    datw = loadPIRATAvariable(PIRATAdir, "w", "WS_401", "QWS_5401")

    # set pressure to constant value
    datsh = xr.DataArray(
        specificHumidity(dath, datt, 1012.1),
        name="SH",
        attrs=dict(long_name="SPECIFIC HUMIDITY", units="g/kg"),
    )
    return xr.merge([datt, dath, datsh, datr, datw])


# %%


def loadColdPools(
    path_to_csv,
    date_columns=["DCS_Init", "DCS_End", "tmax", "tmin", "DCS_Time_Closest"],
):
    dataframes = []

    for filename in os.listdir(path_to_csv):
        if filename.startswith("v3_") and filename.endswith("coldpools.csv"):
            print(filename)
            filepath = os.path.join(path_to_csv, filename)
            df_year = pd.read_csv(
                filepath,
                parse_dates=date_columns,
            )  # DCS_Time_Closest
            dataframes.append(df_year)

    df = pd.concat(dataframes, ignore_index=True)
    df.DCS = df.DCS.astype(int)
    return df


# %%
if __name__ == "__main__":
    main()
