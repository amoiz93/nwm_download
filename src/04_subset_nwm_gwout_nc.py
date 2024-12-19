import os
import sys
import glob
import time
import numpy as np
import pandas as pd
import xarray as xr
import zarr
import pyogrio
import geopandas as gpd
import nwm
import param_nwm3
import misc


start_date = '2000-09-01' #
end_date = '2001-01-31' #

download_entire_range = True

ds_envelope_az_lcc_1km = xr.open_dataset(param_nwm3.nc_nwm_envelope_az_lcc_1km)

# Read NWM Parameters
gdf_nwm3_reaches_az_lcc = pyogrio.read_dataframe(param_nwm3.shp_nwm_reaches_az_lcc)
gdf_nwm3_catchments_az_lcc = pyogrio.read_dataframe(param_nwm3.shp_nwm_catchments_az_lcc)
gdf_nwm3_waterbodies_az_lcc = pyogrio.read_dataframe(param_nwm3.shp_nwm_waterbodies_az_lcc)



# var = str(sys.argv[1])
var = 'inflow' # Change variable here

# Reading variable from AWS
ds = nwm.s3_nwm3_zarr_bucket('CONUS/zarr/gwout.zarr')
ds = ds[var]

var_dict = {
            'depth':{'varname':'depth',
                      'unit_conversion':1, # mm --> mm
                      'new_unit':'mm'}, 
            'inflow':{'varname':'inflow',
                      'unit_conversion':1*3600, # m3 s-1 --> m3
                      'new_unit':'m3'}, 
            'outflow':{'varname':'outflow',
                      'unit_conversion':1*3600, # m3 s-1 --> m3
                      'new_unit':'m3'}, 
                    }

# Reading variable attributes
attrs = ds.attrs

# Convert units
ds = ds*var_dict[var]['unit_conversion']

# Change unit label in attributes
attrs['units'] = var_dict[var]['new_unit']

# Assign new attributes
ds = ds.assign_attrs(attrs)

# Spatial Subset
ds = ds.sel(feature_id=ds['feature_id'].isin(gdf_nwm3_reaches_az_lcc['ID']))

# Temporal Subset
if download_entire_range == True:
    start_date = pd.to_datetime(ds.isel(time=0).time.values)
    end_date = pd.to_datetime(ds.isel(time=-1).time.values)
date_range  = pd.date_range(start_date,end_date,freq='M')


# ds = ds.sel(time=slice(start_date,end_date))
# ds = ds[var]
# #ds_f = ds_f.sel(time=pd.date_range('1979-03-01 00:00:00','2023-02-01 00:00:00',freq='MS'))

for dt in date_range:
    elapsed_time_start = time.time()

    year_dt = dt.year
    month_dt = dt.month
    
    ds_t = ds.sel(time=str(year_dt).zfill(4)+'-'+str(month_dt).zfill(2))

    # Specify save location and save name
    savedir = os.path.join(param_nwm3.out_dir,os.path.basename(__file__).split('.')[0],'gwout','nc','hourly',var)
    misc.makedir(savedir)
    filename = os.path.join(savedir,str(year_dt).zfill(4)+str(month_dt).zfill(2)+'.nc')

    ds_t.to_netcdf(filename)
    filesize = round(os.stat(filename).st_size/(1024 * 1024),2)

    cyverse_savedir = os.path.join(cyverse_nwm3_dir,'gwout','nc','hourly',var)
    cyverse_filename = os.path.join(cyverse_savedir,str(year_dt).zfill(4)+str(month_dt).zfill(2)+'.nc')
    #upload_to_cyverse(irodsfs,filename,cyverse_filename,remove_local=True)
 
    elapsed_time_end = time.time()
    elapsed_time = round(elapsed_time_end - elapsed_time_start,2)

    print(var,year_dt,month_dt,'Elapsed Time: {}s'.format(str(elapsed_time)),'Filesize: {}MB'.format(str(filesize)))

