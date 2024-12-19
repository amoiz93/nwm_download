import os
import sys
import glob
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import misc
import nwm
import param_nwm3

# Just Testing to Read CRS

# Reading NWM

# Forcing
precip = nwm3.s3_nwm_zarr_bucket('CONUS/zarr/forcing/precip.zarr') # Precipitation
t2d = nwm3.s3_nwm_zarr_bucket('CONUS/zarr/forcing/t2d.zarr')       # Air Temperature
q2d = nwm3.s3_nwm_zarr_bucket('CONUS/zarr/forcing/q2d.zarr')       # Specific Humidity
psfc = nwm3.s3_nwm_zarr_bucket('CONUS/zarr/forcing/psfc.zarr')     # Surface Pressure
swdown = nwm3.s3_nwm_zarr_bucket('CONUS/zarr/forcing/swdown.zarr') # Downward Shortwave Radiation
lwdown = nwm3.s3_nwm_zarr_bucket('CONUS/zarr/forcing/lwdown.zarr') # Downward Longwave Radiation
u2d = nwm3.s3_nwm_zarr_bucket('CONUS/zarr/forcing/u2d.zarr')       # U-Component of Wind
v2d = nwm3.s3_nwm_zarr_bucket('CONUS/zarr/forcing/v2d.zarr')       # V-Component of Wind

# LDASOUT
ldasout = nwm3.s3_nwm_zarr_bucket('CONUS/zarr/ldasout.zarr')

# RTOUT
rtout = nwm3.s3_nwm_zarr_bucket('CONUS/zarr/rtout.zarr')

# GWOUT
gwout = nwm3.s3_nwm_zarr_bucket('CONUS/zarr/gwout.zarr')

# CHROUT
chrtout = nwm3.s3_nwm_zarr_bucket('CONUS/zarr/chrtout.zarr')

# LAKEOUT
lakeout = nwm3.s3_nwm_zarr_bucket('CONUS/zarr/lakeout.zarr')



