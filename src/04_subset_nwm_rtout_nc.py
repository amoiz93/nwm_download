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

ds_envelope_az_lcc_1km = xr.open_dataset(param_nwm3.nc_nwm_envelope_az_lcc_1km)



var = 'sfcheadsubrt' #str(sys.argv[1])
# var = 'zwattablrt' #str(sys.argv[1])

# Reading variable from AWS
ds = nwm.s3_nwm3_zarr_bucket('CONUS/zarr/rtout.zarr')
ds = ds[var]

var_dict = {
            'sfcheadsubrt':{'varname':'sfcheadsubrt',
                      'unit_conversion':1, # mm --> mm
                      'new_unit':'mm'},
            'zwattablrt':{'varname':'zwattablrt',
                      'unit_conversion':1, # m --> m
                      'new_unit':'m'},
                    }

# Reading variable attributes
attrs = ds.attrs

# Convert units
ds = ds*var_dict[var]['unit_conversion']

# Change unit label in attributes
attrs['units'] = var_dict[var]['new_unit']

# Assign new attributes
ds = ds.assign_attrs(attrs)

# # Spatial Subset
# ds = ds.sel(x=slice(ds_envelope_az_lcc_1km['x'][0],ds_envelope_az_lcc_1km['x'][-1]),
#             y=slice(ds_envelope_az_lcc_1km['y'][0],ds_envelope_az_lcc_1km['y'][-1]))
# ds = ds.sel(time=pd.date_range('1979-03-01 00:00:00','2022-12-01 00:00:00',freq='MS'))

# for year in range(1979,2023):
#     print(year)
#     ds_t = ds.sel(time=str(year))
#     ds_t.to_netcdf('/data/EWN/amoiz/projects/20231219_NWMv3.0_AZ_HUC8_subset/out/04_subset_nwm_rtout_nc/yearly_zwattablrt/{}.nc'.format(str(year)))

#ds1 = ds.sel(time='2023-01-01',method='nearest')

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
    # savedir = os.path.join(param_nwm3.out_dir,os.path.basename(__file__).split('.')[0],'rtout','nc','3-hourly',var)
    savedir = os.path.join(param_nwm3.scratch_dir,os.path.basename(__file__).split('.')[0],'rtout','nc','3-hourly',var)
    misc.makedir(savedir)
    filename = os.path.join(savedir,str(year_dt).zfill(4)+str(month_dt).zfill(2)+'.nc')


    # Write to NetCDF
    comp = dict(zlib=True, complevel=5)
    encoding = {var: comp}
        
    ds_t.to_netcdf(filename)#,encoding=encoding)
    filesize = round(os.stat(filename).st_size/(1024 * 1024),2)
 
    elapsed_time_end = time.time()
    elapsed_time = round(elapsed_time_end - elapsed_time_start,2)

    print(var,year_dt,month_dt,'Elapsed Time: {}s'.format(str(elapsed_time)),'Filesize: {}MB'.format(str(filesize)))

