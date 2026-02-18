# Calms and Storms

This repository provides the code for my master's thesis on the impact of mesoscale convective systems on the doldrums.

It is structured in two parts. The detection method for low wind speed events in the doldrums, as well as the detection of mesoscale convective systems through cold pools measured at buoys.

## Get the data

Data measured by the PIRATA buoys in the Atlantic  is accessible via this website. https://www.pmel.noaa.gov/tao/drupal/disdel/

Data on deep convective systems detected by TOOCAN is available on these websites:
https://toocan.ipsl.fr/toocandatabase/
and
https://doi.org/10.15770/EUM_SEC_CLM_1005

## Doldrums

The characteristic low wind speed events of the doldrums are defined as events where the wind speed is below 3m/s for a minimum duration of 6 hours.

```
python doldrums.py <PIRATA data directory> -o <output directory> 
```

This creates a netCDF file including a low wind speed event field, which is 1 at times during an event and 0 otherwise.

## Cold pools produced by MCSs

MCSs detected by TOOCAN are selected if they closely pass by a buoy. For the lifetime of these selected MCSs, the buoy is air temperature measured by the buoy is analysed …

```
python MCSonsetDetection.py -p <PIRATA data directory> -t <TOOCAN data directory> -o <output directory> 
````

This creates a CSV file with the start and end times of the cold pool front and information on the connected MCS.

`MCSonsetDetection.sh` executes MCSonsetDetection.py by year.

Based on the onset and minimum temperature time stamps, the buoy measurements can be rescaled to a cold pool time (`MCScomposites.py`). This rescaled data can then be composited.

```
python MCScomposites.py -p <PIRATA data directory> -c <cold pool directory> -o <output directory>
```
