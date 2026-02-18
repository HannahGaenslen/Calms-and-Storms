#!/usr/bin/env my_env

"""
Detect a single cold pool in buoy data.
"""

import numpy as np
import pandas as pd
import xarray as xr
from scipy.signal import find_peaks
import math

from argparse import ArgumentParser


def main():
    parser = ArgumentParser()

    parser.add_argument("cdfFile", help="Path to PIRATA cdf file")
    parser.add_argument(
        "--Init",
        help="Initialisation time of MCS",
    )
    parser.add_argument(
        "--End",
        help="End time of MCS",
    )
    parser.add_argument(
        "--lat",
        help="Latitude of the buoy",
    )
    parser.add_argument(
        "--lon",
        help="Longitude of the buoy",
    )

    args = vars(parser.parse_args())

    datt = loadDataset(args["cdfFile"])
    t_Init, t_End = pd.to_datetime(args["Init"]), pd.to_datetime(args["End"])
    lat, lon = args["lat"], args["lon"]
    tmax, tmin, tend = coldpooltimes(datt.sel(lat=lat, lon=lon), t_Init, t_End)


def loadDataset(cdfpath):
    datt = xr.open_mfdataset(cdfpath, combine="by_coords", engine="netcdf4")
    # Qulity masking:
    datt["AT_21"] = (
        datt["AT_21"]
        .where((datt["QAT_5021"] != 4) & (datt["QAT_5021"] != 5), np.nan)
        .sel(depth=-3)
    )
    datt = datt.drop_vars(["QAT_5021", "SAT_6021"])
    return datt


def runningAverage(datt):
    # Apply a running average with a window size of 5
    window_size = 7  # 1h 10min window
    datt["AT_21_rolling"] = datt["AT_21"].rolling(time=window_size, center=True).mean()
    return datt


def diffrence(datt):
    datt["AT_21_diff"] = datt["AT_21_rolling"] - datt["AT_21_rolling"].shift(time=1)
    return datt


def peaks(data, minima=False):
    if minima:
        data = -data
    peaks, _ = find_peaks(data, prominence=0.1, rel_height=1)
    return peaks


def idx_maxima(T_diff, init_threshold=-0.2):
    idx_onsets = []
    for idx_init, elm in enumerate(T_diff):
        # Initial temperature diffrence below the threshold
        if elm < init_threshold:
            # print(f'∂T<{init_threshold}K: ', idx_init)
            idx_onsets.append(
                # Find nearest occation of a positive value before the initial temperature drop
                # If there is no positive value before the initial temperature drop, return 0
                next(
                    (
                        idx_init - 1 - idx
                        for idx, elm in enumerate(reversed(T_diff[:idx_init]))
                        if elm > 0
                    ),
                    0,
                )
            )
    if idx_onsets == []:
        print("No cold pools found")

    # remove duplicates and sort the indices
    idx_onsets = list(set(idx_onsets))
    idx_onsets.sort()
    return idx_onsets


def idx_minima(idx_onsets, T_rolling):
    idx_mins = []
    minima = peaks(T_rolling, minima=True)
    # print('Minima: ',  minima)
    for idx_onset in idx_onsets:
        # nearest minimum to onset time
        idx_mins.append(
            next(
                (minimum for minimum in minima if minimum > idx_onset),
                len(T_rolling) - 1,
            )
        )
        # missing: subsequent cold pool case
        # missing: tmin is last value of time interval
    return idx_mins


def idx_ends(idx_onsets, idx_min, count, T_rolling, deltaT):
    Tmin = T_rolling[idx_min]

    # (a) first instance after t_min where the temperature exceeds ∆T/e
    idx_end_a = next(
        (
            idx_min + idx
            for idx, T in enumerate(T_rolling[idx_min:])
            if T > Tmin + deltaT / math.e
        ),
        None,  # This is the default value if no match is found
    )

    # what happens if T doesn't exceed ∆T/e during the time interval?

    # (b) onset of next cold pool
    idx_end_b = None
    if count + 1 < len(idx_onsets):
        idx_end_b = idx_onsets[count + 1]

    # (c) time when T first decreases after increasing for some time
    idx_end_c = next(
        (
            idx_min + idx
            for idx, (i, j) in enumerate(
                zip(T_rolling[idx_min:], T_rolling[idx_min + 1 :])
            )
            if i > j
        ),
        None,
    )

    print(f"t_end from method (a): {idx_end_a}")
    print(f"t_end from method (b): {idx_end_b}")
    print(f"t_end from method (c): {idx_end_c}")

    if not idx_end_a:
        # set tend to the last time in the interval
        if idx_end_b and not idx_end_c:
            print("b, no a or c")
            idx_end = idx_end_b
        elif not idx_end_b and not idx_end_c:
            print("d")
            idx_end = len(T_rolling) - 1
        else:
            print("c, no a or b")
            idx_end = idx_end_c

    # does the temperature drop below Tmin - 0.15K at any time between tmin and tend?
    elif (idx_end_a and any(T_rolling[idx_min:idx_end_a] < Tmin - 0.15)) or (
        idx_end_b and any(T_rolling[idx_min:idx_end_b] < Tmin - 0.15)
    ):
        # set tend to time when T first decreases after increasing for some time
        print("c")
        idx_end = idx_end_c

    # is the onset of the next cold pool closer then tend?
    elif idx_end_a and idx_end_b and idx_end_a > idx_end_b:
        # set tend to onset of the next cold pool
        print("b")
        idx_end = idx_end_b

    # elif not idx_end_a and idx_end_b:
    #     # set tend to onset of the next cold pool
    #     print('b')
    #     idx_end = idx_end_b

    else:
        # set tend to first instance after t_min where the temperature exceeds ∆T/e
        print("a")
        idx_end = idx_end_a

    return idx_end


def coldpooltimes(ds, t_Init, t_End, t_bouy, init_thld=-0.2, deltaT_thld=1):
    """
    Find the onset time (tmax) of a MCS induced cold pool
    and the time at minimum Temperature within the cold pool and end time
        ds: Temperature Data Set
        t_Init: Start time of the MCS
        t_End: End time of the MCS
        t_bouy: Time when the DCS is closest to the buoy
        init_thld: Initial temperature difference threshold (default: -0.2K)
        deltaT_thld: Minimum temperature drop threshold (default: 1K)
    """
    t_delta = pd.Timedelta(hours=6)
    # This might not be the best way to define the time interval
    interval = slice(t_Init - t_delta, t_End + t_delta)

    runningAverage(ds)
    diffrence(ds)

    T_rolling = ds.AT_21_rolling.sel(time=interval)
    T_diff = ds.AT_21_diff.sel(time=interval)

    time = ds.time.sel(time=interval).values

    tmax = []
    tmin = []
    tend = []

    idx_onsets = idx_maxima(T_diff, init_thld)
    # print("Onset indices: ", idx_onsets)
    idx_mins = idx_minima(idx_onsets, T_rolling)

    for count, (idx_onset, idx_min) in enumerate(zip(idx_onsets, idx_mins)):
        print("Onset index: ", idx_onset, "Min index: ", idx_min)

        Tmax = T_rolling[idx_onset]
        Tmin = T_rolling[idx_min]
        deltaT = Tmax - Tmin

        # the temperature drop is at least 'deltaT_thld'
        if deltaT > deltaT_thld:
            # idx_end = idx_ends(idx_onsets, idx_min, count, T_rolling, deltaT)
            # print(f"tmax: {idx_onset}, tmin: {idx_min}, tend: {idx_end}, length time: {len(time)}")
            # print(f"tmax: {time[idx_onset]}, tmin: {time[idx_min]}, tend: {time[idx_end]} is  a cold pool")
            print(f"tmax: {time[idx_onset]}, tmin: {time[idx_min]} is  a cold pool")

            tmax.append(time[idx_onset])
            tmin.append(time[idx_min])
            # tend.append(time[idx_end])
        else:
            print(
                f"tmax: {time[idx_onset]}, tmin: {time[idx_min]} is not a cold pool with a temperature drop of at least {deltaT_thld}K"
            )

    if len(tmax) > 1:
        print("tmax: ", tmax)
        # print("t_bouy: ", t_bouy)
        # Find the tmax closest to the time when the DCS is closest to the buoy
        t_gap = [abs(t - t_bouy) for t in tmax]
        min_dist_idx = np.argmin(t_gap)
        tmax = tmax[min_dist_idx]
        tmin = tmin[min_dist_idx]
        # tend = tend[min_dist_idx]
    elif len(tmax) == 1:
        tmax = tmax[0]
        tmin = tmin[0]
        # tend = tend[0]
    elif len(tmax) == 0:
        tmax, tmin, tend = None, None, None

    return tmax, tmin  # , tend


if __name__ == "__main__":
    main()
