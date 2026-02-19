#%%
import intake
import xarray as xr
from easygems import healpix as egh
import healpy
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cf

from datetime import datetime, timedelta
#%%
# Load a catalog (remember it's 48h-long anf includes 24h of spin up)
cat = intake.open_catalog("https://tcodata.mpimet.mpg.de/internal.yaml")

# %%
# Define the date range for aggregation
start_date = datetime(2024, 8, 11)
end_date = datetime(2024, 9, 27)
date_range = [start_date + timedelta(days=i) for i in range((end_date - start_date).days + 1)]

ds_total = []

for date in date_range:
    second_day_start = date + timedelta(days=1)
    second_day_end = second_day_start + timedelta(days=1)

    ds_date = cat.ORCESTRA.ICON_LAM(date=date.strftime("%Y-%m-%d"), dim="2d").to_dask()
    ds_date = ds_date.sel(time=slice(second_day_start, second_day_end))
    ds_date = ds_date.assign(sfcwind=lambda dx: np.hypot(dx.uas, dx.vas))
    ds_total.append(ds_date)

# %%
ds_total = xr.concat(ds_total, dim='time')

#%%
day = '2024-08-17'
time = '12:00'
projection = ccrs.Robinson(central_longitude=-36)
fig, ax = plt.subplots(
    figsize=(8, 4), subplot_kw={"projection": projection}, constrained_layout=True
)
ax.set_extent([-62, -10, -2, 20], crs=ccrs.PlateCarree())

egh.healpix_show(ds_total.sfcwind.sel(time=f"{day} {time}"), vmin=0, vmax=20, cmap=plt.cm.magma, ax=ax)

ax.add_feature(cf.COASTLINE, linewidth=0.8)
ax.add_feature(cf.BORDERS, linewidth=0.4)


#%%
buoy_lon = -23
buoy_lat = 8

vertices = healpy.ang2pix(
    egh.get_nside(ds_total), buoy_lon, buoy_lat, lonlat=True, nest=egh.get_nest(ds_total)
)
# indices = healpy.query_polygon(egh.get_nside(ds_total), vertices, nest=egh.get_nest(ds_total))
print(vertices)

# ds_total.sel(cell=indices, method="nearest").sfcwind.plot()

# theta, phi = healpy.pix2ang(egh.get_nside(ds_total), indices)
# lon = np.degrees(phi)
# lat = 90 - np.degrees(theta)

# print(f"Longitude: input {buoy_lon} output {lon}")
# print(f"Latitude: input {buoy_lat} output {lat}")

# np.degrees(cell=)
np.degrees(ds_total.clon_bnds.sel(cell=vertices).isel(time=0).values)

#%%
def get_nest(ds):
    return ds.crs.healpix_order == "nest"


def circle_vertices(c_lon, c_lat, radius, n_ds=12):
    phi_ds = (2.0 * np.pi / n_ds) * np.arange(n_ds)
    lon_ds, lat_ds = radius * np.cos(phi_ds) + c_lon, radius * np.sin(phi_ds) + c_lat
    return lon_ds, lat_ds


def rectangle_vertices(c_lon_min, c_lon_max, c_lat_min, c_lat_max, n_ds=12):
    return [c_lon_min, c_lon_max, c_lon_max, c_lon_min], [
        c_lat_min,
        c_lat_min,
        c_lat_max,
        c_lat_max,
    ]


def get_healpix_indices(ds, c_lon, c_lat, radius):

    ds_nside = ds.crs.healpix_nside
    ds_nest = get_nest(ds)

    vertices = healpy.ang2vec(*circle_vertices(c_lon, c_lat, radius), lonlat=True)

    indices = healpy.query_polygon(ds_nside, vertices, nest=ds_nest)

    return indices

get_healpix_indices(ds_total, buoy_lon, buoy_lat, 0.05)
#%%
def get_nn_lon_lat_index(nside, lons, lats):
    lons2, lats2 = np.meshgrid(lons, lats)
    return xr.DataArray(
        healpy.ang2pix(nside, lons2, lats2, nest=True, lonlat=True),
        coords=[("lat", lats), ("lon", lons)],
    )

cells = get_nn_lon_lat_index(egh.get_nside(ds_total), np.arange(-46, -26, 1), np.arange(-1, 19, 1))

# get the lat lon location from the cells 
#(to make sure I get correct values for the DCS index extraction)
round(np.mean(np.degrees(ds_total.clon_bnds.sel(cell=cells.sel(lat=0, lon=-30)).isel(time=0).values)), 2)

# %%
ds_buoy = ds_total.sel(cell=buoy_cell, method="nearest")

# Create a mask for low wind speed events
LWSE_mask = xr.where(ds_buoy['sfcwind'] <= 3, 1, 0)
# Tweek mask to count events 
LWSE_count = (
    ((LWSE_mask != LWSE_mask.shift(time=1)).cumsum() + 1) 
    * LWSE_mask)//2

#%%
# group low wind speed events by count
gb = LWSE_mask.groupby(LWSE_count.copy().compute())

# compute the duration of each low wind speed event (by using xarray count function)
time_res = 1/6
LWSE_duration = ((gb + gb.count(...)) - 1) * LWSE_mask * time_res

# filter low wind speed events by duration and create a binary map with all events > 6 hours
LWSE_field = xr.where(LWSE_duration > 6, 1, 0)

# %%
fig, ax = plt.subplots()
LWSE_field.plot(ax=ax)
ax.set_ylabel('Low Wind Speed Events')

ax.set_title(f"Low Wind Speed Events at Buoy Location {buoy_lat}°N {buoy_lon}°W \n cell {LWSE_mask.cell.values} ")

# %%
fig, ax = plt.subplots()
LWSE_duration.plot(ax=ax)
ax.set_ylabel('Duration (h)')

ax.set_title(f"Duration of Low Wind Speed Events at Buoy Location {buoy_lat}°N {buoy_lon}°W \n cell {LWSE_mask.cell.values} ")
