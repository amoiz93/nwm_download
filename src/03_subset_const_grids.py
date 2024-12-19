import os
import sys
import glob
import time
import numpy as np
import pandas as pd
import xarray as xr
import rioxarray
import geopandas as gpd
import pyogrio

# Mataplotlib
import matplotlib.pyplot as plt

# Cartopy
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy.io.img_tiles as cimgt
import cartopy.feature as cfeature

# Custom Libraries
import nwm
import param_nwm3
import misc

def clip_reproject_const_grid(da_var,clip_boundary,crs,target_crs,min_threshold,max_threshold):
    # Clip to HUC8 Boundary
    da_var = da_var.rio.write_crs(crs)

    # Buffered (AZ-HUC8-Buffered)
    da_var_buffered_crs = da_var
    da_var_buffered_crs = da_var_buffered_crs.where((da_var_buffered_crs<=max_threshold)&
                                                    (da_var_buffered_crs>=min_threshold))
    da_var_buffered_target_crs = da_var.rio.reproject(target_crs)
    da_var_buffered_target_crs = da_var_buffered_target_crs.where((da_var_buffered_target_crs<=max_threshold)&
                                                    (da_var_buffered_target_crs>=min_threshold))


    # Clipped (AZ-HUC8)
    da_var_clipped_crs = da_var.rio.clip(clip_boundary.geometry)
    da_var_clipped_crs = da_var_clipped_crs.where((da_var_clipped_crs<=max_threshold)&
                                                    (da_var_clipped_crs>=min_threshold))

    da_var_clipped_target_crs = da_var_clipped_crs.rio.reproject(target_crs)
    da_var_clipped_target_crs = da_var_clipped_target_crs.where((da_var_clipped_target_crs<=max_threshold)&
                                                    (da_var_clipped_target_crs>=min_threshold))

    return {'crs':{'buffered':da_var_buffered_crs,
                   'clipped':da_var_clipped_crs},
            'target_crs':{'buffered':da_var_buffered_target_crs,
                          'clipped':da_var_clipped_target_crs}}


# Create Savedir
savedir = os.path.join(param_nwm3.out_dir,os.path.basename(__file__).split('.')[0])
misc.makedir(savedir)

# Read Boundary Data
gdf_wbdhu8_boundary_az_lcc = gpd.read_parquet(param_nwm3.gp_wbdhu8_boundary_az_lcc)
bounding_box = gdf_wbdhu8_boundary_az_lcc.total_bounds
buffer = 200000 # m
xlim = [bounding_box[0]-buffer,bounding_box[2]+buffer]
ylim = [bounding_box[3]+buffer,bounding_box[1]-buffer]

# Read 1km Precipitation Data
ppt = nwm.s3_nwm3_zarr_bucket('CONUS/zarr/forcing/precip.zarr')


geo_em_conus = xr.open_dataset('/data/EWN/amoiz/data/nwm3/nwm_parameters/nwm.v3.0.10/geo_em_CONUS.nc')
wrfinput_conus = xr.open_dataset('/data/EWN/amoiz/data/nwm3/nwm_parameters/nwm.v3.0.10/wrfinput_CONUS.nc')
GEOGRID_LDASOUT_Spatial_Metadata_conus = xr.open_dataset('/data/EWN/amoiz/data/nwm3/nwm_parameters/nwm.v3.0.10/GEOGRID_LDASOUT_Spatial_Metadata_CONUS.nc')

wrfinput_conus = wrfinput_conus.assign_coords({'west_east':ppt['x'].values,'south_north':ppt['y'].values})
wrfinput_conus = wrfinput_conus.rename({'west_east':'x','south_north':'y'})
wrfinput_az = wrfinput_conus.sel(x=slice(xlim[0],xlim[1]),y=slice(ylim[1],ylim[0]))


hgt_az = wrfinput_az['HGT'].isel(Time=0)       # Terrain Height
ivgtyp_az = wrfinput_az['IVGTYP'].isel(Time=0) # Vegetation Type
isltyp_az = wrfinput_az['ISLTYP'].isel(Time=0) # Soil Type

# ------------HGT-----------
hgt_az = clip_reproject_const_grid(hgt_az,
                          clip_boundary=gdf_wbdhu8_boundary_az_lcc,
                          crs=param_nwm3.crs_nwm_proj4_lcc,
                          target_crs=param_nwm3.crs_utm12n_nad83,
                          min_threshold=-5000,
                          max_threshold=5000)

hgt_az['crs']['buffered'].to_netcdf(os.path.join(savedir,'hgt_az_buffered_lcc.nc'))
hgt_az['target_crs']['buffered'].to_netcdf(os.path.join(savedir,'hgt_az_buffered_utm12n_nad83.nc'))
hgt_az['crs']['buffered'].rio.to_raster(os.path.join(savedir,'hgt_az_buffered_lcc.tif'))
hgt_az['target_crs']['buffered'].rio.to_raster(os.path.join(savedir,'hgt_az_buffered_utm12n_nad83.tif'))
hgt_az['crs']['clipped'].rio.to_raster(os.path.join(savedir,'hgt_az_huc8_lcc.tif'))
hgt_az['target_crs']['clipped'].rio.to_raster(os.path.join(savedir,'hgt_az_huc8_utm12n_nad83.tif'))

# ------------IVGTYPE-----------
vgtyp_dict = {1: 'Urban and Built-Up Land',
                2: 'Dryland Cropland and Pasture',
                3: 'Irrigated Cropland and Pasture',
                4: 'Mixed Dryland/Irrigated Cropland and Pasture',
                5: 'Cropland/Grassland Mosaic',
                6: 'Cropland/Woodland Mosaic',
                7: 'Grassland',
                8: 'Shrubland',
                9: 'Mixed Shrubland/Grassland',
                10: 'Savanna',
                11: 'Deciduous Broadleaf Forest',
                12: 'Deciduous Needleleaf Forest',
                13: 'Evergreen Broadleaf Forest',
                14: 'Evergreen Needleleaf Forest',
                15: 'Mixed Forest',
                16: 'Water Bodies',
                17: 'Herbaceous Wetland',
                18: 'Wooded Wetland',
                19: 'Barren or Sparsely Vegetated',
                20: 'Herbaceous Tundra',
                21: 'Wooded Tundra',
                22: 'Mixed Tundra',
                23: 'Bare Ground Tundra',
                24: 'Snow or Ice',
                25: 'Playa',
                26: 'Lava',
                27: 'White Sand'}
ivgtyp_az = clip_reproject_const_grid(ivgtyp_az,
                            clip_boundary=gdf_wbdhu8_boundary_az_lcc,
                            crs=param_nwm3.crs_nwm_proj4_lcc,
                            target_crs=param_nwm3.crs_utm12n_nad83,
                            min_threshold=1,
                            max_threshold=27)

ivgtyp_az['crs']['buffered'].to_netcdf(os.path.join(savedir,'ivgtyp_az_buffered_lcc.nc'))
ivgtyp_az['target_crs']['buffered'].to_netcdf(os.path.join(savedir,'ivgtyp_az_buffered_utm12n_nad83.nc'))
ivgtyp_az['crs']['buffered'].rio.to_raster(os.path.join(savedir,'ivgtyp_az_buffered_lcc.tif'))
ivgtyp_az['target_crs']['buffered'].rio.to_raster(os.path.join(savedir,'ivgtyp_az_buffered_utm12n_nad83.tif'))
ivgtyp_az['crs']['clipped'].rio.to_raster(os.path.join(savedir,'ivgtyp_az_huc8_lcc.tif'))
ivgtyp_az['target_crs']['clipped'].rio.to_raster(os.path.join(savedir,'ivgtyp_az_huc8_utm12n_nad83.tif'))

ivgtyp_percent = pd.DataFrame(index=range(1,28),columns=['name','percent'])
for i in ivgtyp_percent.index:
    count_values = float(ivgtyp_az['crs']['clipped'].where(ivgtyp_az['crs']['clipped']==i).count())
    total_values = float(ivgtyp_az['crs']['clipped'].count())
    ivgtyp_percent.loc[i,'percent'] = (count_values/total_values)*100
    ivgtyp_percent.loc[i,'name'] = vgtyp_dict[i]
ivgtyp_percent.to_csv(os.path.join(savedir,'ivgtyp_percent.csv'))

# ------------ISLTYP-----------
sltyp_dict = {
1: 'Sand',
2: 'Loamy Sand',
3: 'Sandy Loam',
4: 'Silt Loam',
5: 'Silt',
6: 'Loam',
7: 'Sandy Clay Loam',
8: 'Silty Clay Loam',
9: 'Clay Loam',
10: 'Sandy Clay',
11: 'Silty Clay',
12: 'Clay',
13: 'Organic Material',
14: 'Water',
15: 'Bedrock',
16: 'Other (land-ice)',
17: 'Playa',
18: 'Lava',
19: 'White Sand'}
                

isltyp_az = clip_reproject_const_grid(isltyp_az,
                            clip_boundary=gdf_wbdhu8_boundary_az_lcc,
                            crs=param_nwm3.crs_nwm_proj4_lcc,
                            target_crs=param_nwm3.crs_utm12n_nad83,
                            min_threshold=1,
                            max_threshold=19)
isltyp_az['crs']['buffered'].to_netcdf(os.path.join(savedir,'isltyp_az_buffered_lcc.nc'))
isltyp_az['target_crs']['buffered'].to_netcdf(os.path.join(savedir,'isltyp_az_buffered_utm12n_nad83.nc'))
isltyp_az['crs']['buffered'].rio.to_raster(os.path.join(savedir,'isltyp_az_buffered_lcc.tif'))
isltyp_az['target_crs']['buffered'].rio.to_raster(os.path.join(savedir,'isltyp_az_buffered_utm12n_nad83.tif'))
isltyp_az['crs']['clipped'].rio.to_raster(os.path.join(savedir,'isltyp_az_huc8_lcc.tif'))
isltyp_az['target_crs']['clipped'].rio.to_raster(os.path.join(savedir,'isltyp_az_huc8_utm12n_nad83.tif'))

isltyp_percent = pd.DataFrame(index=range(1,19),columns=['name','percent'])
for i in isltyp_percent.index:
    count_values = float(isltyp_az['crs']['clipped'].where(isltyp_az['crs']['clipped']==i).count())
    total_values = float(isltyp_az['crs']['clipped'].count())
    isltyp_percent.loc[i,'percent'] = (count_values/total_values)*100
    isltyp_percent.loc[i,'name'] = sltyp_dict[i]
isltyp_percent.to_csv(os.path.join(savedir,'isltyp_percent.csv'))
