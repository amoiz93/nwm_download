import os
import sys
import glob
import time
import numpy as np
import pandas as pd
import xarray as xr
import nwm
import param_nwm3
import misc
import fsspec

start_date = '1979-02-01' #
end_date = '1979-03-31' #

download_entire_range = False

ds_envelope_az_lcc_1km = xr.open_dataset(param_nwm3.nc_nwm_envelope_az_lcc_1km)

var = str(sys.argv[1])

var_dict = {
            'precip':{'aws_path':'CONUS/zarr/forcing/precip.zarr',
                      'varname':'RAINRATE',
                      'unit_conversion':3600, # mm/s --> mm/h
                      'new_unit':'mm h-1'},
            't2d':{'aws_path':'CONUS/zarr/forcing/t2d.zarr',
                   'varname':'T2D',
                   'unit_conversion':1, # K --> K
                   'new_unit':'K'},
            'q2d':{'aws_path':'CONUS/zarr/forcing/q2d.zarr',
                   'varname':'Q2D',
                   'unit_conversion':1, # kg kg-1 --> kg kg-1
                   'new_unit':'kg kg-1'},
            'psfc':{'aws_path':'CONUS/zarr/forcing/psfc.zarr',
                    'varname':'PSFC',
                    'unit_conversion':0.01, # Pa --> hPa
                    'new_unit':'hPa'},
            'swdown':{'aws_path':'CONUS/zarr/forcing/swdown.zarr',
                    'varname':'SWDOWN',
                    'unit_conversion':1, # W m-2 --> W m-2
                    'new_unit':'W m-2'},
            'lwdown':{'aws_path':'CONUS/zarr/forcing/lwdown.zarr',
                    'varname':'LWDOWN',
                    'unit_conversion':1, # W m-2 --> W m-2
                    'new_unit':'W m-2'},
            'u2d':{'aws_path':'CONUS/zarr/forcing/u2d.zarr',
                    'varname':'U2D',
                    'unit_conversion':1, # m s-1 --> m s-1
                    'new_unit':'m s-1'},
            'v2d':{'aws_path':'CONUS/zarr/forcing/v2d.zarr',
                    'varname':'V2D',
                    'unit_conversion':1, # m s-1 --> m s-1
                    'new_unit':'m s-1'}
                    }


# Reading variable from AWS
ds = nwm.s3_nwm3_zarr_bucket(var_dict[var]['aws_path'])

# Reading variable attributes
attrs = ds[var_dict[var]['varname']].attrs

# Convert units
ds[var_dict[var]['varname']] = ds[var_dict[var]['varname']]*var_dict[var]['unit_conversion']

# Change unit label in attributes
attrs['units'] = var_dict[var]['new_unit']

# Assign new attributes
ds[var_dict[var]['varname']] = ds[var_dict[var]['varname']].assign_attrs(attrs)

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
    #savedir = os.path.join(param_nwm3.out_dir,os.path.basename(__file__).split('.')[0],'forcing','nc','hourly',var)
    #savedir = os.path.join('/Users/amoiz2/Desktop','forcing','nc','hourly',var)
    savedir = os.path.join(param_nwm3.scratch_dir,os.path.basename(__file__).split('.')[0],'forcing','nc','hourly',var)
    misc.makedir(savedir)
    filename = os.path.join(savedir,str(year_dt).zfill(4)+str(month_dt).zfill(2)+'.nc')

    # Write to NetCDF
    comp = dict(zlib=True, complevel=5)
    encoding = {var: comp for var in ds_t.data_vars}
        
    ds_t.load().to_netcdf(filename,encoding=encoding)
    filesize = round(os.stat(filename).st_size/(1024 * 1024),2)
    elapsed_time_end = time.time()
    elapsed_time = round(elapsed_time_end - elapsed_time_start,2)

    print(var,year_dt,month_dt,'Elapsed Time: {}s'.format(str(elapsed_time)),'Filesize: {}MB'.format(str(filesize)))
