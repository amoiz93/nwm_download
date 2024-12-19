import os
import platform
import socket
import glob
import geopandas as gpd
import xarray as xr
import cartopy.crs as ccrs
import nwm

# Project Directory
inp_dir = '../inp'
log_dir = '../log'
out_dir = '../out'
plt_dir = '../plt'

# Data Directory
ewn_data_dir = '/data/EWN/amoiz/data'
nwm_data_dir = os.path.join(ewn_data_dir,'nwm3')
scratch_dir = '/scratch/amoiz2' # Can only run on ASU Cluster at this moment
#============================NWM Directories===================================
# NWM Data Directories
nwm_retrospective_dir = os.path.join(nwm_data_dir,'retrospective')

# National
nwm_national_dir = os.path.join(nwm_retrospective_dir,'national')
nwm_national_const_dir = os.path.join(nwm_national_dir,'const')

# Arizona
nwm_az_dir = os.path.join(nwm_retrospective_dir,'az')
nwm_az_const_dir = os.path.join(nwm_az_dir,'const')
nwm_az_forcing_dir = os.path.join(nwm_az_dir,'forcing')
nwm_az_ldasout_dir = os.path.join(nwm_az_dir,'ldasout')
nwm_az_chrtout_dir = os.path.join(nwm_az_dir,'chrtout')

# Arizona Pilot
nwm_az_pilot_dir = os.path.join(nwm_retrospective_dir,'az_pilot')
nwm_az_pilot_forcing_dir = os.path.join(nwm_az_pilot_dir,'forcing')
nwm_az_pilot_ldasout_dir = os.path.join(nwm_az_pilot_dir,'ldasout')
nwm_az_pilot_rtout_dir = os.path.join(nwm_az_pilot_dir,'rtout')
nwm_az_pilot_chrtout_dir = os.path.join(nwm_az_pilot_dir,'chrtout')

#-------------------------------NWM Parameters--------------------------------

nc_Fulldom_CONUS_FullRouting = os.path.join(nwm_data_dir,'nwm_parameters/nwm.v3.0.9','Fulldom_CONUS_FullRouting.nc')
nc_LAKEPARM_CONUS = os.path.join(nwm_data_dir,'nwm_parameters/v3.0_par','LAKEPARM_CONUS.nc')
nc_RouteLink_CONUS = os.path.join(nwm_data_dir,'nwm_parameters/v3.0_par','RouteLink_CONUS.nc')
nc_spatialweights_CONUS_FullRouting = os.path.join(nwm_data_dir,'nwm_parameters/ncep_nwm.v3.0.7','spatialweights_CONUS_FullRouting.nc')
nc_spatialweights_CONUS_LongRange = os.path.join(nwm_data_dir,'nwm_parameters/ncep_nwm.v3.0.7','spatialweights_CONUS_LongRange.nc')
nc_soilproperties_CONUS_FullRouting = os.path.join(nwm_data_dir,'nwm_parameters/nwm.v3.0.9','soilproperties_CONUS_FullRouting.nc')

#-------------------------------NWM Hydrofabric--------------------------------
# National
gdb_nwm_hydrofabric_nad83 = os.path.join(nwm_data_dir,'nwm_hydrofabric/NWM_v3_hydrofabric.gdb')

# AZ
gp_nwm_reaches_az_lcc = os.path.join(nwm_az_const_dir,'nwm_params/nwm_reaches_az_lcc.parquet.gzip')
gp_nwm_reaches_buffered_az_utm12n = os.path.join(nwm_az_const_dir,'nwm_params/nwm_reaches_buffered_az_utm12n.parquet.gzip')
gp_nwm_catchments_az_lcc = os.path.join(nwm_az_const_dir,'nwm_params/nwm_catchments_az_lcc.parquet.gzip')
gp_nwm_waterbodies_az_lcc = os.path.join(nwm_az_const_dir,'nwm_params/nwm_waterbodies_az_lcc.parquet.gzip')
gp_nwm_reaches_az_utm12n_nad83 = os.path.join(nwm_az_const_dir,'nwm_params/nwm_reaches_az_utm12n_nad83.parquet.gzip')
gp_nwm_reaches_buffered_az_utm12n_nad83 = os.path.join(nwm_az_const_dir,'nwm_params/nwm_reaches_az_buffered_utm12n_nad83.parquet.gzip')
gp_nwm_reaches_buffered_az_huc8_utm12n_nad83 = os.path.join(nwm_az_const_dir,'nwm_params/nwm_reaches_az_huc8_buffered_utm12n_nad83.parquet.gzip')
gp_nwm_catchments_az_utm12n_nad83 = os.path.join(nwm_az_const_dir,'nwm_params/nwm_catchments_az_utm12n_nad83.parquet.gzip')
gp_nwm_waterbodies_az_utm12n_nad83 = os.path.join(nwm_az_const_dir,'nwm_params/nwm_waterbodies_az_utm12n_nad83.parquet.gzip')
#------------------------------------------------------------------------------

#-------------------------------HUC Basins-------------------------------------
gp_wbdhu8_boundary_az_lcc = os.path.join(nwm_az_const_dir,'basins/huc8_az_boundary_lcc.parquet.gzip')
gp_wbdhu8_basins_az_lcc = os.path.join(nwm_az_const_dir,'basins/huc8_az_basins_lcc.parquet.gzip')

gp_wbdhu8_boundary_az_utm12n_nad83 = os.path.join(nwm_az_const_dir,'basins/huc8_az_boundary_utm12n_nad83.parquet.gzip')
gp_wbdhu8_basins_az_utm12n_nad83 = os.path.join(nwm_az_const_dir,'basins/huc8_az_basins_utm12n_nad83.parquet.gzip')
shp_az_simplified_watersheds_utm12n_nad83 = os.path.join(nwm_az_const_dir,'basins/az_watersheds_simplified_utm12n_nad83.shp')
#------------------------------------------------------------------------------

#-------------------------------HUC Basins Mask--------------------------------
nc_huc_basins_az_mask_lcc = os.path.join(nwm_az_const_dir,'masks/az_huc_mask.nc')
#------------------------------------------------------------------------------

#--------------------------------NWM Envelope----------------------------------
nc_nwm_envelope_az_lcc_1km = os.path.join(nwm_az_const_dir,'envelope/az_huc8_envelope_1km.nc')
nc_nwm_envelope_az_lcc_250m = os.path.join(nwm_az_const_dir,'envelope/az_huc8_envelope_250m.nc')
#------------------------------------------------------------------------------

#-------------------------------- AZ HUC NWM Gridded Parameters-----------------------------------
nc_nwm_hgt_az_buffered_utm12n_nad83 = os.path.join(nwm_az_const_dir,'nwm_grid/hgt_az_buffered_utm12n_nad83.nc')
nc_nwm_isltyp_az_buffered_utm12n_nad83 = os.path.join(nwm_az_const_dir,'nwm_grid/isltyp_az_buffered_utm12n_nad83.nc')
nc_nwm_ivgtyp_az_buffered_utm12n_nad83 = os.path.join(nwm_az_const_dir,'nwm_grid/ivgtyp_az_buffered_utm12n_nad83.nc')


#--------------------------------AZ Local Data----------------------------------
shp_az_watersheds = os.path.join(ewn_data_dir,'az_gis/Watersheds/Watersheds.shp')

#==============================================================================


#=============================USGS Directories==================================
usgs_govtunit_dir = os.path.join(ewn_data_dir,'usgs','GovtUnit')
usgs_wbd_dir = os.path.join(ewn_data_dir,'usgs','WBD')
usgs_hcdn_dir = os.path.join(ewn_data_dir,'usgs','HCDN')
#----------------------------------GovtUnit------------------------------------
# US State Boundaries
gdb_govtunit_national_nad83 = os.path.join(usgs_govtunit_dir,'National/GDB/GovernmentUnits_National_GDB.gdb') # Checked
gp_us_states_lcc = os.path.join(usgs_govtunit_dir,'Processed/national/us_states_lcc.parquet.gzip')
gp_us_states_utm12n_nad83 = os.path.join(usgs_govtunit_dir,'Processed/national/us_states_nad83.parquet.gzip')

# AZ State Boundaries
gp_az_states_lcc = os.path.join(usgs_govtunit_dir,'Processed/az/az_state_lcc.parquet.gzip')
gp_az_states_utm12n_nad83 = os.path.join(usgs_govtunit_dir,'Processed/az/az_state_utm12n_nad83.parquet.gzip')
#------------------------------------------------------------------------------

#---------------------------------------WBD------------------------------------
# US WBD National
gdb_wbd_national_nad83 = os.path.join(usgs_wbd_dir, 'original/WBD_National_GDB.gdb')

# US HUC Boundaries (NAD83)
gp_wbdhu2_nad83 = os.path.join(usgs_wbd_dir,'Processed/national/WBDHU2_nad83.parquet.gzip')
gp_wbdhu4_nad83 = os.path.join(usgs_wbd_dir,'Processed/national/WBDHU4_nad83.parquet.gzip')
gp_wbdhu6_nad83 = os.path.join(usgs_wbd_dir,'Processed/national/WBDHU6_nad83.parquet.gzip')
gp_wbdhu8_nad83 = os.path.join(usgs_wbd_dir,'Processed/national/WBDHU8_nad83.parquet.gzip')
gp_wbdhu10_nad83 = os.path.join(usgs_wbd_dir,'Processed/national/WBDHU10_nad83.parquet.gzip')
gp_wbdhu12_nad83 = os.path.join(usgs_wbd_dir,'Processed/national/WBDHU12_nad83.parquet.gzip')

#------------------------------------------------------------------------------

#--------------------------------------HCDN------------------------------------
shp_hcdn2009_wgs84 = os.path.join(usgs_hcdn_dir,'Processed/national/hcdn2009_wgs84.shp')
shp_hcdn2009_lcc = os.path.join(usgs_hcdn_dir,'Processed/national/hcdn2009_lcc.shp')
shp_hcdn2009_az_lcc = os.path.join(usgs_hcdn_dir,'Processed/az/hcdn2009_az_lcc.shp')
#------------------------------------------------------------------------------


#============================EPA Directories===================================
#epa_nhdplusv2_dir = os.path.join(ewn_data_dir,'epa','nhdplusv2')

# #----------------------------------NHDPlusv2-----------------------------------
# gdb_nhdplus_seamless = os.path.join(epa_nhdplusv2_dir,'data/NationalData/NHDPlusNationalData/NHDPlusV21_National_Seamless_Flattened_Lower48.gdb')
# shp_nhdgage_lcc = os.path.join(epa_nhdplusv2_dir,'Processed/national/Gage_lcc.shp')
# shp_nhdflowline_lcc = os.path.join(epa_nhdplusv2_dir,'Processed/national/NHDFlowline_Network_lcc.shp')

# shp_nhdgage_az_lcc = os.path.join(epa_nhdplusv2_dir,'Processed/az/Gage_AZ_lcc.shp') # This is the same as Source_FEA
# shp_nhdgage_az_full_lcc = os.path.join(epa_nhdplusv2_dir,'Processed/az/Gage_AZ_full_lcc.shp') 
# shp_nhdgage_az_lcc_flcomid = os.path.join(epa_nhdplusv2_dir,'Processed/az/Gage_AZ_lcc_FLComID.shp')
# shp_nhdgage_az_lcc_source_fea = os.path.join(epa_nhdplusv2_dir,'Processed/az/Gage_AZ_lcc_SOURCE_FEA.shp')
# shp_nhdflowline_az_lcc = os.path.join(epa_nhdplusv2_dir,'Processed/az/NHDFlowline_Network_AZ_lcc.shp')

# #------------------------------------------------------------------------------


#==============================================================================




#===================================CRS========================================
#------------------------------------NWM CRS-----------------------------------
crs_nwm_proj4_lcc = '+proj=lcc \
                   +units=m \
                   +a=6370000.0 \
                   +b=6370000.0 \
                   +lat_1=30.0 \
                   +lat_2=60.0 \
                   +lat_0=40.0 \
                   +lon_0=-97.0 \
                   +x_0=0 \
                   +y_0=0 \
                   +k_0=1.0 \
                   +nadgrids=@null \
                   +wktext  \
                   +no_defs '
crs_wgs84 = 'EPSG:4326'
crs_nad83 = 'EPSG:4269'
crs_albers = '+proj=aea \
              +lat_0=23 \
              +lon_0=-96 \
              +lat_1=29.5 \
              +lat_2=45.5 \
              +x_0=0 \
              +y_0=0 \
              +datum=NAD83 \
              +units=m \
              +no_defs \
              +type=crs'

#crs_utm12n = '+proj=utm +zone=12 +datum=WGS84 +units=m +no_defs +type=crs'
crs_utm12n_wgs84 = 'EPSG:32612'
crs_utm12n_nad83 = 'EPSG:26912'


#------------------------------------------------------------------------------

#---------------------------------Plot Properties------------------------------
# Define CRS for Plotting
cartopy_crs_nwm_lcc = ccrs.LambertConformal(central_longitude=-97,
                            central_latitude=40,
                            standard_parallels=(30,60))
cartopy_crs_nwm_albers = ccrs.AlbersEqualArea(central_longitude=-96.0,
                            central_latitude=23,
                            standard_parallels=(29.5,45.5))
cartopy_crs_atur_utm12n = ccrs.UTM(zone=12)
#==============================================================================







#===============================White River Catchment==========================
shp_white_river_outlet_az_lcc = os.path.join(nwm_az_const_dir,'White_River/white_river_outlet.shp')
tif_white_river_catchment_az_lcc =  os.path.join(nwm_az_const_dir,'White_River/catchment.tif')
tif_white_river_channel_az = os.path.join(nwm_az_const_dir,'White_River/channel_az.tif')
tif_white_river_dem_az = os.path.join(nwm_az_const_dir,'White_River/dem_az.tif')
tif_white_river_fdir_az = os.path.join(nwm_az_const_dir,'White_River/fdir_az.tif')


# Remove Later
# crs_proj4 = crs.proj4_init






#----------------------------------Notes---------------------------------------
# # PRECIP
# {'GeoTransform': '-2303999.17655 1000.0 0 1919999.66329 0 -1000.0 ',
#  '_CoordinateAxes': 'y x',
#  '_CoordinateTransformType': 'Projection',
#  'earth_radius': 6370000.0,
#  'esri_pe_string': 'PROJCS["Lambert_Conformal_Conic",GEOGCS["GCS_Sphere",DATUM["D_Sphere",SPHEROID["Sphere",6370000.0,0.0]],PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433]],PROJECTION["Lambert_Conformal_Conic_2SP"],PARAMETER["false_easting",0.0],PARAMETER["false_northing",0.0],PARAMETER["central_meridian",-97.0],PARAMETER["standard_parallel_1",30.0],PARAMETER["standard_parallel_2",60.0],PARAMETER["latitude_of_origin",40.0],UNIT["Meter",1.0]];-35691800 -29075200 10000;-100000 10000;-100000 10000;0.001;0.001;0.001;IsHighPrecision',
#  'false_easting': 0.0,
#  'false_northing': 0.0,
#  'grid_mapping_name': 'lambert_conformal_conic',
#  'inverse_flattening': 0.0,
#  'latitude_of_projection_origin': 40.0,
#  'long_name': 'CRS definition',
#  'longitude_of_central_meridian': -97.0,
#  'longitude_of_prime_meridian': 0.0,
#  'semi_major_axis': 6370000.0,
#  'spatial_ref': 'PROJCS["Lambert_Conformal_Conic",GEOGCS["GCS_Sphere",DATUM["D_Sphere",SPHEROID["Sphere",6370000.0,0.0]],PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433]],PROJECTION["Lambert_Conformal_Conic_2SP"],PARAMETER["false_easting",0.0],PARAMETER["false_northing",0.0],PARAMETER["central_meridian",-97.0],PARAMETER["standard_parallel_1",30.0],PARAMETER["standard_parallel_2",60.0],PARAMETER["latitude_of_origin",40.0],UNIT["Meter",1.0]];-35691800 -29075200 10000;-100000 10000;-100000 10000;0.001;0.001;0.001;IsHighPrecision',
#  'standard_parallel': [30.0, 60.0],
#  'transform_name': 'lambert_conformal_conic'}

# '+proj=lcc +units=m +a=6370000.0 +b=6370000.0 +lat_1=30.0 +lat_2=60.0 +lat_0=40.0 +lon_0=-97.0 +x_0=0 +y_0=0 +k_0=1.0 +nadgrids=@null +wktext  +no_defs '


# #LDASOUT
# '+proj=lcc +units=m +a=6370000.0 +b=6370000.0 +lat_1=30.0 +lat_2=60.0 +lat_0=40.0 +lon_0=-97.0 +x_0=0 +y_0=0 +k_0=1.0 +nadgrids=@null +wktext  +no_defs'
# 'PROJCS["Lambert_Conformal_Conic",GEOGCS["GCS_Sphere",DATUM["D_Sphere",SPHEROID["Sphere",6370000.0,0.0]],PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433]],PROJECTION["Lambert_Conformal_Conic_2SP"],PARAMETER["false_easting",0.0],PARAMETER["false_northing",0.0],PARAMETER["central_meridian",-97.0],PARAMETER["standard_parallel_1",30.0],PARAMETER["standard_parallel_2",60.0],PARAMETER["latitude_of_origin",40.0],UNIT["Meter",1.0]];-35691800 -29075200 10000;-100000 10000;-100000 10000;0.001;0.001;0.001;IsHighPrecision'

# #RTOUT
# {'GeoTransform': '-2303999.17655 250.0 0 1919999.66329 0 -250.0',
#  '_CoordinateAxes': 'y x',
#  '_CoordinateTransformType': 'Projection',
#  'earth_radius': 6370000.0,
#  'esri_pe_string': 'PROJCS["Lambert_Conformal_Conic",GEOGCS["GCS_Sphere",DATUM["D_Sphere",SPHEROID["Sphere",6370000.0,0.0]],PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433]],PROJECTION["Lambert_Conformal_Conic_2SP"],PARAMETER["false_easting",0.0],PARAMETER["false_northing",0.0],PARAMETER["central_meridian",-97.0],PARAMETER["standard_parallel_1",30.0],PARAMETER["standard_parallel_2",60.0],PARAMETER["latitude_of_origin",40.0],UNIT["Meter",1.0]];-35691800 -29075200 10000;-100000 10000;-100000 10000;0.001;0.001;0.001;IsHighPrecision',
#  'false_easting': 0.0,
#  'false_northing': 0.0,
#  'grid_mapping_name': 'lambert_conformal_conic',
#  'inverse_flattening': 0.0,
#  'latitude_of_projection_origin': 40.0,
#  'long_name': 'CRS definition',
#  'longitude_of_central_meridian': -97.0,
#  'longitude_of_prime_meridian': 0.0,
#  'semi_major_axis': 6370000.0,
#  'spatial_ref': 'PROJCS["Lambert_Conformal_Conic",GEOGCS["GCS_Sphere",DATUM["D_Sphere",SPHEROID["Sphere",6370000.0,0.0]],PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433]],PROJECTION["Lambert_Conformal_Conic_2SP"],PARAMETER["false_easting",0.0],PARAMETER["false_northing",0.0],PARAMETER["central_meridian",-97.0],PARAMETER["standard_parallel_1",30.0],PARAMETER["standard_parallel_2",60.0],PARAMETER["latitude_of_origin",40.0],UNIT["Meter",1.0]];-35691800 -29075200 10000;-100000 10000;-100000 10000;0.001;0.001;0.001;IsHighPrecision',
#  'standard_parallel': [30.0, 60.0],
#  'transform_name': 'lambert_conformal_conic'}

# '+proj=lcc +units=m +a=6370000.0 +b=6370000.0 +lat_1=30.0 +lat_2=60.0 +lat_0=40.0 +lon_0=-97.0 +x_0=0 +y_0=0 +k_0=1.0 +nadgrids=@null +wktext  +no_defs'
# 'PROJCS["Lambert_Conformal_Conic",GEOGCS["GCS_Sphere",DATUM["D_Sphere",SPHEROID["Sphere",6370000.0,0.0]],PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433]],PROJECTION["Lambert_Conformal_Conic_2SP"],PARAMETER["false_easting",0.0],PARAMETER["false_northing",0.0],PARAMETER["central_meridian",-97.0],PARAMETER["standard_parallel_1",30.0],PARAMETER["standard_parallel_2",60.0],PARAMETER["latitude_of_origin",40.0],UNIT["Meter",1.0]];-35691800 -29075200 10000;-100000 10000;-100000 10000;0.001;0.001;0.001;IsHighPrecision'

# # LAKEOUT
# '+proj=lcc +units=m +a=6370000.0 +b=6370000.0 +lat_1=30.0 +lat_2=60.0 +lat_0=40.0 +lon_0=-97.0 +x_0=0 +y_0=0 +k_0=1.0 +nadgrids=@'
# 'GEOGCS["GCS_WGS_1984",DATUM["D_WGS_1984",SPHEROID["WGS_1984",6378137.0,298.257223563]],PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433]];-400 -400 1000000000;-100000 10000;-100000 10000;8.98315284119521E-09;0.001;0.001;IsHighPrecision'

# # CHRTOUT
# '+proj=lcc +units=m +a=6370000.0 +b=6370000.0 +lat_1=30.0 +lat_2=60.0 +lat_0=40.0 +lon_0=-97.0 +x_0=0 +y_0=0 +k_0=1.0 +nadgrids=@'


if __name__ =='__main__':
    print(socket.gethostname())
    
