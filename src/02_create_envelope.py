import os
import xarray as xr
import pandas as pd
import numpy as np
import geopandas as gpd
import pyogrio
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.pyplot as plt

import nwm
import param_nwm3
import misc



# Inputs:
# gdf_wbdhu8_boundary_az_lcc = AZ-HUC8 outer boundary Shapefile 

# Outputs:
# envelope_1km = AZ-HUC8 1km Buffer Envelope (.nc,.tif)
# envelop_250m = AZ-HUC8 250m Buffer Envelope (.nc,.tif)

# Read AZ-HUC8 Boundary
gdf_wbdhu8_boundary_az_lcc = pyogrio.read_dataframe(param_nwm3.shp_wbdhu8_boundary_az_lcc)

# Create 10km Buffer
gdf_wbdhu8_boundary_az_lcc_10km_buffer = gdf_wbdhu8_boundary_az_lcc.buffer(10*1000)

xmin = gdf_wbdhu8_boundary_az_lcc_10km_buffer.bounds['minx'].values[0]
xmax = gdf_wbdhu8_boundary_az_lcc_10km_buffer.bounds['maxx'].values[0]

ymin = gdf_wbdhu8_boundary_az_lcc_10km_buffer.bounds['miny'].values[0]
ymax = gdf_wbdhu8_boundary_az_lcc_10km_buffer.bounds['maxy'].values[0]

# Read 1km Precipitation Data
ppt = nwm.s3_nwm3_zarr_bucket('CONUS/zarr/forcing/precip.zarr')
ppt = (ppt.sel(x=slice(xmin,xmax),y=slice(ymin,ymax))).isel(time=0).drop('time')
envelope_1km = ppt['RAINRATE'].where(ppt['RAINRATE'] == 1, other=1)
remove_attrs = list(envelope_1km.attrs.keys())
for attrs in remove_attrs:
    del envelope_1km.attrs[attrs]
envelope_1km = ppt.assign({'envelope':envelope_1km})
envelope_1km = envelope_1km.drop_vars(['RAINRATE'])
remove_attrs = list(envelope_1km.attrs.keys())
for attrs in remove_attrs:
    del envelope_1km.attrs[attrs]
envelope_1km = envelope_1km.assign_attrs({'proj4':ppt['RAINRATE'].proj4})

# Read 250 m RTOUT
rtout = nwm.s3_nwm3_zarr_bucket('CONUS/zarr/rtout.zarr')
rtout= (rtout.sel(x=slice(xmin,xmax),y=slice(ymin,ymax))).isel(time=0).drop('time')
envelope_250m = rtout['sfcheadsubrt'].where(rtout['sfcheadsubrt'] == 1, other=1)
remove_attrs = list(envelope_250m.attrs.keys())
for attrs in remove_attrs:
    del envelope_250m.attrs[attrs]
envelope_250m = rtout.assign({'envelope':envelope_250m})
envelope_250m = envelope_250m.drop_vars(['sfcheadsubrt','zwattablrt'])
remove_attrs = list(envelope_250m.attrs.keys())
for attrs in remove_attrs:
    del envelope_250m.attrs[attrs]
envelope_250m = envelope_250m.assign_attrs({'proj4':ppt['RAINRATE'].proj4})

#--------------------------------Save Files-------------------------------------
savedir = os.path.join(param_nwm3.out_dir,os.path.basename(__file__).split('.')[0])
misc.makedir(savedir)

misc.makedir(os.path.dirname(param_nwm3.nc_nwm_envelope_az_lcc_1km))
envelope_1km.to_netcdf(param_nwm3.nc_nwm_envelope_az_lcc_1km)
envelope_1km = envelope_1km['envelope'].rio.set_crs(param_nwm3.crs_nwm_proj4_lcc)
envelope_1km.rio.to_raster(param_nwm3.nc_nwm_envelope_az_lcc_1km.replace('.nc','.tif'))

misc.makedir(os.path.dirname(param_nwm3.nc_nwm_envelope_az_lcc_250m))
envelope_250m.to_netcdf(param_nwm3.nc_nwm_envelope_az_lcc_250m)
envelope_250m = envelope_250m['envelope'].rio.set_crs(param_nwm3.crs_nwm_proj4_lcc)
envelope_250m.rio.to_raster(param_nwm3.nc_nwm_envelope_az_lcc_250m.replace('.nc','.tif'))
