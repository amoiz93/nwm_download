import os
import sys
import glob
import time
import numpy as np
import pandas as pd
import xarray as xr
import zarr
import nwm
import param_nwm3
import misc


start_date = '2000-09-01' #
end_date = '2001-01-31' #

download_entire_range = True

ds_envelope_az_lcc_1km = xr.open_dataset(param_nwm.nc_nwm_envelope_az_lcc_1km)



var = str(sys.argv[1])

# Reading variable from AWS
ds = nwm.s3_nwm3_zarr_bucket('CONUS/zarr/ldasout.zarr')
ds = ds[var]

var_dict = {
            'ACCET':{'varname':'ACCET',
                      'unit_conversion':1, # mm --> mm
                      'new_unit':'mm'},
            'SNEQV':{'varname':'SNEQV',
                      'unit_conversion':1, # kg/m2 --> mm
                      'new_unit':'mm'},
            'SNOWH':{'varname':'SNOWH',
                      'unit_conversion':1, # m --> m
                      'new_unit':'m'},
            'UGDRNOFF':{'varname':'UGDRNOFF',
                      'unit_conversion':1, # mm --> mm
                      'new_unit':'mm'},                      
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
ds = ds.sel(x=ds_envelope_az_lcc_1km['x'],
            y=ds_envelope_az_lcc_1km['y'])


# Temporal Subset
if download_entire_range == True:
    start_date = pd.to_datetime(ds.isel(time=0).time.values)
    end_date = pd.to_datetime(ds.isel(time=-1).time.values)
date_range  = pd.date_range(start_date,end_date,freq='M')

for dt in date_range:
    elapsed_time_start = time.time()

    year_dt = dt.year
    month_dt = dt.month
    
    ds_t = ds.sel(time=str(year_dt).zfill(4)+'-'+str(month_dt).zfill(2))
    
    # Specify save location and save name
    savedir = os.path.join(param_nwm.out_dir,os.path.basename(__file__).split('.')[0],'ldasout','nc','3-hourly',var)
    misc.makedir(savedir)
    filename = os.path.join(savedir,str(year_dt).zfill(4)+str(month_dt).zfill(2)+'.nc')


    # Write to NetCDF
    comp = dict(zlib=True, complevel=5)
    encoding = {var: comp}
        
    ds_t.to_netcdf(filename,encoding=encoding)
    filesize = round(os.stat(filename).st_size/(1024 * 1024),2)
 
    elapsed_time_end = time.time()
    elapsed_time = round(elapsed_time_end - elapsed_time_start,2)

    print(var,year_dt,month_dt,'Elapsed Time: {}s'.format(str(elapsed_time)),'Filesize: {}MB'.format(str(filesize)))

