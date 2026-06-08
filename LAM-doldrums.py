#%%
import intake
import xarray as xr
from easygems import healpix as egh
import healpy
import numpy as np

from doldrums import create_LWSE_dataset

#%%
# Load a catalog
cat = intake.open_catalog("https://tcodata.mpimet.mpg.de/internal.yaml")

ds = cat.ORCESTRA.LAM_ORCESTRA(dim="2d").to_dask()

# # Add lat lon coordinates
# # For this analysis, we'll need lat and lon coordinates, here's a way to attach them to the dataset:
# ds = ds.pipe(egh.attach_coords)

# create surface wind variable
ds = ds.assign(sfcwind=lambda dx: np.hypot(dx['uas'], dx['vas']))
ds["sfcwind"] = ds.sfcwind.assign_attrs({"long_name": "surface wind in 10", "standard_name": "sfcwind"})

#%%
def get_nn_lon_lat_index(nside, lons, lats):
    lons2, lats2 = np.meshgrid(lons, lats)
    return xr.DataArray(
        healpy.ang2pix(nside, lons2, lats2, nest=True, lonlat=True),
        coords=[("lat", lats), ("lon", lons)],
    )

nside = egh.get_nside(ds)
cells = get_nn_lon_lat_index(nside, np.arange(-46, -26, 1), np.arange(-1, 19, 1))

# Get the actual lat/lon of the matched HEALPix cell centres
# (to make sure I get correct values for the virtual buoys)
lat_test, lon_test = 0, -30
test_cell = cells.sel(lat=lat_test, lon=lon_test)
matched_lon, matched_lat = healpy.pix2ang(nside, test_cell, nest=True, lonlat=True)
print(f"Requested : lat={lat_test}, lon={lon_test}")
print(f"Cell centre: lat={matched_lat:.4f}, lon={matched_lon:.4f}")

# %%
da_ws = ds.sfcwind.sel(cell=cells).compute()

da_lwse = create_LWSE_dataset(
    da_ws,
    filter=True,
    ws_thresh=3,
    dur_thresh=6,
    time_res=1 / 6,
    fs=(1 / 600),
    cutoff_freq=(1 / (2 * 3600)),
    order=4,
)

#%% 
ds_buoys = ds.sel(cell=cells)
ds_doldrums  = xr.merge([ds_buoys, da_lwse])
ds_doldrums.to_netcdf("/work/bm1526/data/ORCESTRA-LAM/buoys_surf_vars+lwse.nc")
# %%
# (
#     (ds_doldrums.field.sum(dim="time") / 6 / 24)
#         .plot(
#             x="lon", 
#             y="lat", 
#             cbar_kwargs={"label": "Days spent in LWSE"}
#         )
# )
# %%
