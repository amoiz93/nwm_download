import sys
import os
import sys
import glob
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import pyogrio
import fiona
import nwm
import param_nwm3
import misc

from shapely.geometry import box
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER


# Read WBD
gdf_wbdhu2_national_nad83 = pyogrio.read_dataframe(param_nwm3.gdb_wbd_national_nad83,layer='WBDHU2')
gdf_wbdhu4_national_nad83 = pyogrio.read_dataframe(param_nwm3.gdb_wbd_national_nad83,layer='WBDHU4')
gdf_wbdhu6_national_nad83 = pyogrio.read_dataframe(param_nwm3.gdb_wbd_national_nad83,layer='WBDHU6')
gdf_wbdhu8_national_nad83 = pyogrio.read_dataframe(param_nwm3.gdb_wbd_national_nad83,layer='WBDHU8')
gdf_wbdhu10_national_nad83 = pyogrio.read_dataframe(param_nwm3.gdb_wbd_national_nad83,layer='WBDHU10')
gdf_wbdhu12_national_nad83 = pyogrio.read_dataframe(param_nwm3.gdb_wbd_national_nad83,layer='WBDHU12')
gdf_wbdhu14_national_nad83 = pyogrio.read_dataframe(param_nwm3.gdb_wbd_national_nad83,layer='WBDHU14')
gdf_wbdhu16_national_nad83 = pyogrio.read_dataframe(param_nwm3.gdb_wbd_national_nad83,layer='WBDHU16')


# Read States
gdf_us_states_nad83  = pyogrio.read_dataframe(param_nwm3.gdb_govtunit_national_nad83,layer='GU_StateOrTerritory')
gdf_az_state_nad83 = gdf_us_states_nad83[gdf_us_states_nad83['STATE_NAME'] == 'Arizona']

#-----------------------------Get HUC8 AZ Boundaries---------------------------
print('Processing AZ-HUC8 Boundaries')
gdf_az_state_lcc = gdf_az_state_nad83.to_crs(param_nwm3.crs_nwm_proj4_lcc)
gdf_wbdhu8_national_lcc = gdf_wbdhu8_national_nad83.to_crs(param_nwm3.crs_nwm_proj4_lcc)
gdf_wbdhu8_az_intersection_lcc = gdf_wbdhu8_national_lcc[gdf_wbdhu8_national_lcc.intersects(gdf_az_state_lcc.geometry.unary_union)]
gdf_wbdhu8_az_union_lcc = misc.poly_union(gdf_wbdhu8_az_intersection_lcc)
#------------------------------------------------------------------------------

#---------------------------Subset Chrtout Feature IDs-------------------------
print('Subsetting CHRTOUT')
chrtout = nwm.s3_nwm3_zarr_bucket('CONUS/zarr/chrtout.zarr')
chrtout_coords = chrtout.drop_dims('time')
df_chrtout_coords = chrtout_coords.to_dataframe()
df_chrtout_coords = df_chrtout_coords.reset_index()
chrtout_coords_geometry = gpd.points_from_xy(x=df_chrtout_coords['longitude'], 
                                        y=df_chrtout_coords['latitude'], 
                                        z=None, 
                                        crs=param_nwm3.crs_wgs84)

gdf_chrtout_coords_wgs84 = gpd.GeoDataFrame(data=df_chrtout_coords,
                                            geometry=chrtout_coords_geometry,
                                            crs=param_nwm3.crs_wgs84)
gdf_chrtout_coords_lcc = gdf_chrtout_coords_wgs84.to_crs(param_nwm3.crs_nwm_proj4_lcc)
gdf_chrtout_coords_az_wbdhu8_lcc = gdf_chrtout_coords_lcc[gdf_chrtout_coords_lcc.intersects(gdf_wbdhu8_az_union_lcc.geometry.unary_union)]
chrtout_az_feature_id = gdf_chrtout_coords_az_wbdhu8_lcc['feature_id']
#------------------------------------------------------------------------------

#---------------------------Subset Lakeout Feature IDs-------------------------
print('Subsetting LAKEOUT')
lakeout = nwm.s3_nwm3_zarr_bucket('CONUS/zarr/lakeout.zarr')
lakeout_coords = lakeout.drop_dims('time')
df_lakeout_coords = lakeout_coords.to_dataframe()
df_lakeout_coords = df_lakeout_coords.reset_index()
lakeout_coords_geometry = gpd.points_from_xy(x=df_lakeout_coords['longitude'], 
                                        y=df_lakeout_coords['latitude'], 
                                        z=None, 
                                        crs=param_nwm3.crs_wgs84)
gdf_lakeout_coords_wgs84 = gpd.GeoDataFrame(data=df_lakeout_coords,
                                            geometry=lakeout_coords_geometry,
                                            crs=param_nwm3.crs_wgs84)
gdf_lakeout_coords_lcc = gdf_lakeout_coords_wgs84.to_crs(param_nwm3.crs_nwm_proj4_lcc)
gdf_lakeout_coords_az_wbdhu8_lcc = gdf_lakeout_coords_lcc[gdf_lakeout_coords_lcc.intersects(gdf_wbdhu8_az_union_lcc.geometry.unary_union)]
lakeout_az_feature_id = gdf_lakeout_coords_az_wbdhu8_lcc['feature_id']
#------------------------------------------------------------------------------

#------------------------Subset NWM Hydrofabric Features (AZ-HUC8)-----------------------
print('Processing Hydrofabric')
# NWM Hydrofabric Reaches
gdf_nwm_reaches_conus_nad83 = pyogrio.read_dataframe(param_nwm3.gdb_nwm_hydrofabric_nad83, layer='nwm_reaches_conus')
gdf_nwm_reaches_az_nad83 = gdf_nwm_reaches_conus_nad83[gdf_nwm_reaches_conus_nad83['ID'].isin(chrtout_az_feature_id)]
gdf_nwm_reaches_az_lcc = gdf_nwm_reaches_az_nad83.to_crs(param_nwm3.crs_nwm_proj4_lcc)

# NWM Hydrofabric Catchments
gdf_nwm_catchments_conus_nad83 = pyogrio.read_dataframe(param_nwm3.gdb_nwm_hydrofabric_nad83, layer='nwm_catchments_conus')#.to_crs(ctl_nwm.crs_nwm_proj4_lcc)
gdf_nwm_catchments_az_nad83 = gdf_nwm_catchments_conus_nad83[gdf_nwm_catchments_conus_nad83['ID'].isin(chrtout_az_feature_id)]
gdf_nwm_catchments_az_lcc = gdf_nwm_catchments_az_nad83.to_crs(param_nwm3.crs_nwm_proj4_lcc)

# NWM Hydrofabric Waterbodies
gdf_nwm_waterbodies_conus_nad83 = pyogrio.read_dataframe(param_nwm3.gdb_nwm_hydrofabric_nad83, layer='nwm_waterbodies_conus')#.to_crs(ctl_nwm.crs_nwm_proj4_lcc)
gdf_nwm_waterbodies_az_nad83 = gdf_nwm_waterbodies_conus_nad83[gdf_nwm_waterbodies_conus_nad83['ID'].isin(lakeout_az_feature_id)]
gdf_nwm_waterbodies_az_lcc = gdf_nwm_waterbodies_az_nad83.to_crs(param_nwm3.crs_nwm_proj4_lcc)
#------------------------------------------------------------------------------


#------------------------Subset NWM Hydrofabric Features (Buffered Bounding Box; For plotting)-----------------------
print('Processing Buffered Hydrofabric')
bounding_box_lcc = gdf_wbdhu8_az_union_lcc.total_bounds
bounding_box_lcc = gpd.GeoSeries(box(*gdf_wbdhu8_az_union_lcc.total_bounds))
bounding_box_buffered_lcc = bounding_box_lcc.buffer(200000)
bounding_box_buffered_lcc = bounding_box_buffered_lcc.set_crs(param_nwm3.crs_nwm_proj4_lcc)
bounding_box_buffered_nad83 = bounding_box_buffered_lcc.to_crs(param_nwm3.crs_nad83)

ds_nwm3_chrtout = nwm.s3_nwm3_zarr_bucket('CONUS/zarr/chrtout.zarr')
df_nwm3_chrtout_metadata = ds_nwm3_chrtout[['feature_id','elevation','gage_id','latitude','longitude','order']].to_dataframe()
gdf_nwm_reaches_conus_nad83 = gdf_nwm_reaches_conus_nad83.rename(columns={'ID':'feature_id'})
gdf_nwm_reaches_conus_nad83 = pd.merge(gdf_nwm_reaches_conus_nad83,df_nwm3_chrtout_metadata,on='feature_id')
gdf_nwm_reaches_az_buffered_nad83 = gpd.clip(gdf_nwm_reaches_conus_nad83,list(bounding_box_buffered_nad83.total_bounds))
gdf_nwm_reaches_az_buffered_nad83['gage_id'] = (((gdf_nwm_reaches_az_buffered_nad83['gage_id'].str.decode("utf-8")).str.strip()).replace('',np.nan))

#------------------------------------Saving Outputs----------------------------
print('Saving Files')
savedir = os.path.join(param_nwm3.out_dir,os.path.basename(__file__).split('.')[0])
misc.makedir(savedir)

# Save AZ State
gdf_az_state_nad83['LOADDATE'] = gdf_az_state_nad83['LOADDATE'].astype(str)
gdf_az_state_nad83.to_crs(param_nwm3.crs_nwm_proj4_lcc).to_file(os.path.join(savedir,'az_state_lcc.shp'))
gdf_az_state_nad83.to_crs(param_nwm3.crs_nwm_proj4_lcc).to_parquet(os.path.join(savedir,'az_state_lcc.parquet.gzip'),compression='gzip')
gdf_az_state_nad83.to_crs(param_nwm3.crs_utm12n_nad83).to_file(os.path.join(savedir,'az_state_utm12n_nad83.shp'))
gdf_az_state_nad83.to_crs(param_nwm3.crs_utm12n_nad83).to_parquet(os.path.join(savedir,'az_state_utm12n_nad83.parquet.gzip'),compression='gzip')

# Save AZ-HUC8 Basins
gdf_wbdhu8_az_intersection_lcc['loaddate'] = gdf_wbdhu8_az_intersection_lcc['loaddate'].astype(str)
gdf_wbdhu8_az_intersection_lcc.to_file(os.path.join(savedir,'huc8_az_basins_lcc.shp'))
gdf_wbdhu8_az_intersection_lcc.to_parquet(os.path.join(savedir,'huc8_az_basins_lcc.parquet.gzip'),compression='gzip')
gdf_wbdhu8_az_intersection_lcc.to_crs(param_nwm3.crs_utm12n_nad83).to_file(os.path.join(savedir,'huc8_az_basins_utm12n_nad83.shp'))
gdf_wbdhu8_az_intersection_lcc.to_crs(param_nwm3.crs_utm12n_nad83).to_parquet(os.path.join(savedir,'huc8_az_basins_utm12n_nad83.parquet.gzip'),compression='gzip')

# Save AZ-HUC8 Boundary
gdf_wbdhu8_az_union_lcc = gpd.GeoDataFrame(gdf_wbdhu8_az_union_lcc)
gdf_wbdhu8_az_union_lcc.columns = ['geometry']
gdf_wbdhu8_az_union_lcc = gdf_wbdhu8_az_union_lcc.set_geometry('geometry')
gdf_wbdhu8_az_union_lcc.to_file(os.path.join(savedir,'huc8_az_boundary_lcc.shp'))
gdf_wbdhu8_az_union_lcc.to_parquet(os.path.join(savedir,'huc8_az_boundary_lcc.parquet.gzip'),compression='gzip')
gdf_wbdhu8_az_union_lcc.to_crs(param_nwm3.crs_utm12n_nad83).to_file(os.path.join(savedir,'huc8_az_boundary_utm12n_nad83.shp'))
gdf_wbdhu8_az_union_lcc.to_crs(param_nwm3.crs_utm12n_nad83).to_parquet(os.path.join(savedir,'huc8_az_boundary_utm12n_nad83.parquet.gzip'),compression='gzip')

# Save NWM Reaches
gdf_nwm_reaches_az_lcc.to_file(os.path.join(savedir,'nwm_reaches_az_lcc.shp'))
gdf_nwm_reaches_az_lcc.to_parquet(os.path.join(savedir,'nwm_reaches_az_lcc.parquet.gzip'),compression='gzip')
gdf_nwm_reaches_az_lcc.to_crs(param_nwm3.crs_utm12n_nad83).to_file(os.path.join(savedir,'nwm_reaches_az_utm12n_nad83.shp'))
gdf_nwm_reaches_az_lcc.to_crs(param_nwm3.crs_utm12n_nad83).to_parquet(os.path.join(savedir,'nwm_reaches_az_utm12n_nad83.parquet.gzip'),compression='gzip')

# Save NWM Buffered Reaches
gdf_nwm_reaches_az_buffered_nad83.to_crs(param_nwm3.crs_nwm_proj4_lcc).to_file(os.path.join(savedir,'nwm_reaches_az_buffered_lcc.shp'))
gdf_nwm_reaches_az_buffered_nad83.to_crs(param_nwm3.crs_nwm_proj4_lcc).to_parquet(os.path.join(savedir,'nwm_reaches_az_buffered_lcc.parquet.gzip'),compression='gzip')
gdf_nwm_reaches_az_buffered_nad83.to_crs(param_nwm3.crs_utm12n_nad83).to_file(os.path.join(savedir,'nwm_reaches_az_buffered_utm12n_nad83.shp'))
gdf_nwm_reaches_az_buffered_nad83.to_crs(param_nwm3.crs_utm12n_nad83).to_parquet(os.path.join(savedir,'nwm_reaches_az_buffered_utm12n_nad83.parquet.gzip'),compression='gzip')


# Save NWM Catchments
gdf_nwm_catchments_az_lcc.to_file(os.path.join(savedir,'nwm_catchments_az_lcc.shp'))
gdf_nwm_catchments_az_lcc.to_parquet(os.path.join(savedir,'nwm_catchments_az_lcc.parquet.gzip'),compression='gzip')
gdf_nwm_catchments_az_lcc.to_crs(param_nwm3.crs_utm12n_nad83).to_file(os.path.join(savedir,'nwm_catchments_az_utm12n_nad83.shp'))
gdf_nwm_catchments_az_lcc.to_crs(param_nwm3.crs_utm12n_nad83).to_parquet(os.path.join(savedir,'nwm_catchments_az_utm12n_nad83.parquet.gzip'),compression='gzip')

# Save NWM Waterbodies
gdf_nwm_waterbodies_az_lcc.to_file(os.path.join(savedir,'nwm_waterbodies_az_lcc.shp'))
gdf_nwm_waterbodies_az_lcc.to_parquet(os.path.join(savedir,'nwm_waterbodies_az_lcc.parquet.gzip'),compression='gzip')
gdf_nwm_waterbodies_az_lcc.to_crs(param_nwm3.crs_utm12n_nad83).to_file(os.path.join(savedir,'nwm_waterbodies_az_utm12n_nad83.shp'))
gdf_nwm_waterbodies_az_lcc.to_crs(param_nwm3.crs_utm12n_nad83).to_parquet(os.path.join(savedir,'nwm_waterbodies_az_utm12n_nad83.parquet.gzip'),compression='gzip')


# Save Global Maps
# SHP
gdf_wbdhu2_national_nad83.drop(columns=['loaddate']).to_file(os.path.join(savedir,'WBDHU2_nad83.shp'))
gdf_wbdhu4_national_nad83.drop(columns=['loaddate']).to_file(os.path.join(savedir,'WBDHU4_nad83.shp'))
gdf_wbdhu6_national_nad83.drop(columns=['loaddate']).to_file(os.path.join(savedir,'WBDHU6_nad83.shp'))
gdf_wbdhu8_national_nad83.drop(columns=['loaddate']).to_file(os.path.join(savedir,'WBDHU8_nad83.shp'))
gdf_wbdhu10_national_nad83.drop(columns=['loaddate']).to_file(os.path.join(savedir,'WBDHU10_nad83.shp'))
gdf_wbdhu12_national_nad83.drop(columns=['loaddate']).to_file(os.path.join(savedir,'WBDHU12_nad83.shp'))
# gdf_wbdhu14_national_nad83.drop(columns=['loaddate']).to_file(os.path.join(savedir,'WBDHU14_nad83.shp'))
# gdf_wbdhu16_national_nad83.drop(columns=['loaddate']).to_file(os.path.join(savedir,'WBDHU16_nad83.shp'))
gdf_us_states_nad83.drop(columns=['LOADDATE']).to_file(os.path.join(savedir,'us_states_nad83.shp'))
gdf_us_states_nad83.to_crs(param_nwm3.crs_utm12n_nad83).drop(columns=['LOADDATE']).to_file(os.path.join(savedir,'us_states_utm12n_nad83.shp'))


# GeoParquet
gdf_wbdhu2_national_nad83.to_parquet(os.path.join(savedir,'WBDHU2_nad83.parquet.gzip'),compression='gzip')
gdf_wbdhu4_national_nad83.to_parquet(os.path.join(savedir,'WBDHU4_nad83.parquet.gzip'),compression='gzip')
gdf_wbdhu6_national_nad83.to_parquet(os.path.join(savedir,'WBDHU6_nad83.parquet.gzip'),compression='gzip')
gdf_wbdhu8_national_nad83.to_parquet(os.path.join(savedir,'WBDHU8_nad83.parquet.gzip'),compression='gzip')
gdf_wbdhu10_national_nad83.to_parquet(os.path.join(savedir,'WBDHU10_nad83.parquet.gzip'),compression='gzip')
gdf_wbdhu12_national_nad83.to_parquet(os.path.join(savedir,'WBDHU12_nad83.parquet.gzip'),compression='gzip')
# gdf_wbdhu14_national_nad83.to_parquet(os.path.join(savedir,'WBDHU14_nad83.parquet.gzip'),compression='gzip')
# gdf_wbdhu16_national_nad83.to_parquet(os.path.join(savedir,'WBDHU16_nad83.parquet.gzip'),compression='gzip')
gdf_us_states_nad83.to_parquet(os.path.join(savedir,'us_states_nad83.parquet.gzip'),compression='gzip')
gdf_us_states_nad83.to_crs(param_nwm3.crs_utm12n_nad83).to_parquet(os.path.join(savedir,'us_states_utmn12n_nad83.parquet.gzip'),compression='gzip')