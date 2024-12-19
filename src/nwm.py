import os
import sys
import s3fs
import glob
import xarray as xr
import pandas as pd
import numpy as np
import geopandas as gpd
import pyogrio
import rasterio
import rioxarray
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import misc

from pynhd import NLDI, WaterData, NHDPlusHR
import pynhd as nhd

def s3_nwm3_zarr_bucket(zarr_bucket):
    # Define url path for S3 bucket which contains NOAA-NWM-Retrospective-v-3.0
    s3_path = os.path.join('s3://noaa-nwm-retrospective-3-0-pds/',zarr_bucket)
    s3 = s3fs.S3FileSystem(anon=True)
    store = s3fs.S3Map(root=s3_path, s3=s3, check=False)
    ds = xr.open_zarr(store=store,consolidated=True)
    return ds



def s3_nwm21_zarr_bucket(zarr_bucket):

    # 'gwout': '/glade/p/datashare/ishitas/nwm_retro_v2.1/gwout.zarr',
    # 'lakeout': '/glade/p/datashare/jamesmcc/nwm_retro_v2.1/lakeout.zarr',
    # 'chrtout': '/glade/p/datashare/ishitas/nwm_retro_v2.1/chrtout.zarr',
    # 'precip': '/glade/p/datashare/jamesmcc/nwm_retro_v2.1/precip.zarr',
    # 'ldasout': '/glade/p/datashare/ishitas/nwm_retro_v2.1/ldasout.zarr',
    # 'rtout': '/glade/p/datashare/jamesmcc/nwm_retro_v2.1/rtout.zarr', 

    # Define url path for S3 bucket which contains NOAA-NWM-Retrospective-v-2.1
    s3_path = os.path.join('s3://noaa-nwm-retrospective-2-1-zarr-pds/',zarr_bucket)
    s3 = s3fs.S3FileSystem(anon=True)
    store = s3fs.S3Map(root=s3_path, s3=s3, check=False)
    ds = xr.open_zarr(store=store,consolidated=True)
    return ds



def s3_nwm_nc_bucket(nc_bucket):
    # Define url path for S3 bucket which contains NOAA-NWM-Retrospective-v-2.1
    s3_path = os.path.join('s3://noaa-nwm-retrospective-2-1-pds/',nc_bucket)
    s3 = s3fs.S3FileSystem(anon=True)
    remote_files = s3.glob(s3_path)
    # Iterate through remote_files to create a fileset
    fileset = [s3.open(file) for file in remote_files]
    ds = xr.open_mfdataset(fileset,combine='by_coords',parallel=True)
    return ds

def get_chrtout_gages(ds_chrtout):
    gages = (ds_chrtout['gage_id'].to_dataframe()).iloc[:,-1]
    gages = (((gages.str.decode("utf-8")).str.strip()).replace('',np.nan)).dropna()
    return gages

def subset_chrtout_zarr(chrtout,sel_feature_id,savedir):
    #chrtout = nwm.s3_nwm_zarr_bucket('chrtout.zarr')
    df_chrtout = chrtout.sel(feature_id=sel_feature_id)
    df_chrtout = pd.DataFrame(df_chrtout['streamflow'].to_dataframe()['streamflow'].rename('Q_cms'))
    misc.makedir(os.path.join(savedir,'txt'))
    misc.makedir(os.path.join(savedir,'pkl'))
    df_chrtout.to_csv(os.path.join(savedir,'txt','{}.txt'.format(str(sel_feature_id))))
    df_chrtout.to_pickle(os.path.join(savedir,'pkl','{}.txt'.format(str(sel_feature_id))))
    return df_chrtout

def subset_chrtout_nc_mp(tup):
    return subset_chrtout_nc(*tup)

def subset_chrtout_nc(date,sel_feature_id,savedir):
    print(date)
    chrtout_date = []
    for hour in range(0,24):
        filename='model_output/{}/{}{}{}{}00.CHRTOUT_DOMAIN1.comp'.format(str(date.year).zfill(4),
                                                                            str(date.year).zfill(4),
                                                                            str(date.month).zfill(2),
                                                                            str(date.day).zfill(2),
                                                                            str(hour).zfill(2))
        #print(filename)
        chrtout_datetime = s3_nwm_nc_bucket(filename)
        chrtout_datetime = chrtout_datetime.sel(feature_id=sel_feature_id)
        chrtout_datetime = chrtout_date.drop('reference_time') # Testing
        chrtout_date.append(chrtout_datetime)
    chrtout_date = xr.concat(chrtout_date,dim='time')
    comp = dict(zlib=True, complevel=5)
    encoding = {var: comp for var in chrtout_date.data_vars}
    chrtout_date.to_netcdf(os.path.join(savedir,str(date.year).zfill(4)+\
                                               str(date.month).zfill(2)+\
                                               str(date.day).zfill(2)+'.nc'),
                           encoding=encoding)
    return chrtout_date

# def accet_to_et(accet):
#     accet_diff = accet.diff('time',label='upper') # Difference along time dimension (keep upper label)
#     accet_dt = xr.concat([accet.isel(time=0),accet_diff],dim='time')
#     et = accet_dt.where(~((accet_dt['time.month']%3 == 1) & (accet_dt['time.day'] == 1) & (accet_dt['time.hour'] == 3)), accet)
#     et = et.where(et>=0,0)
#     return et

# def accugdrnoff_to_ugdrnoff(accugdrnoff):
#     accugdrnoff_diff = accugdrnoff.diff('time',label='upper') # Difference along time dimension (keep upper label)
#     accugdrnoff_dt = xr.concat([accugdrnoff.isel(time=0),accugdrnoff_diff],dim='time')
#     ugdrnoff = accugdrnoff_dt.where(~((accugdrnoff_dt['time.month']%3 == 1) & (accugdrnoff_dt['time.day'] == 1) & (accugdrnoff_dt['time.hour'] == 3)), accugdrnoff)
#     return ugdrnoff

# def accet_to_et(acc_ds,set_negative_et_to_zero=False):
#     acc_ds_diff = acc_ds.diff('time',label='upper') # Difference along time dimension (keep upper label)
#     ds_dt = xr.concat([acc_ds.isel(time=0),acc_ds_diff],dim='time')
#     ds = ds_dt.where(~((ds_dt['time.month']%3 == 1) & (ds_dt['time.day'] == 1) & (ds_dt['time.hour'] == 3)), acc_ds)
#     if set_negative_et_to_zero==True:
#         ds = ds.where(ds>=0,0) # Sometimes ET is negative
#     ds = ds.drop_isel(time=0)
#     return ds

def accet_to_et(ds,set_negative_et_to_zero=False):
    ds_A = ds.sel(time=slice('1979','1989')).copy()
    ds_B = ds.sel(time=slice('1990','2023')).copy()
    ds_diff = ds.diff('time',label='upper') # Difference along time dimension (keep upper label)
    ds_dt = xr.concat([ds.isel(time=0),ds_diff],dim='time')
    ds_dt = ds_dt.copy()

    # 1979 - 1989
    ds_dt_A = ds_dt.sel(time=slice('1979','1989')).copy()
    ds_dt_A = ds_dt_A.where(~((ds_dt_A['time.month'] == 1) & (ds_dt_A['time.day'] == 1) & (ds_dt_A['time.hour'] == 3)), ds_A)
    for month in [4,7,10]:
        ds_dt_A = ds_dt_A.where(~((ds_dt_A['time.month'] == month) & (ds_dt_A['time.day'] == 1) & (ds_dt_A['time.hour'] == 0)), ds_A)

    # 1990 - 2023
    ds_dt_B = ds_dt.sel(time=slice('1990','2023')).copy()
    ds_dt_B = ds_dt_B.where(~((ds_dt_B['time.month']%3 == 1) & (ds_dt_B['time.day'] == 1) & (ds_dt_B['time.hour'] == 3)), ds_B)

    ds = xr.concat([ds_dt_A,ds_dt_B],dim='time')
    
    if set_negative_et_to_zero==True:
        ds = ds.where(ds>=0,0) # Sometimes ET is negative
    ds = ds.drop_isel(time=0)
    return ds

def accet_to_et_df(dfs):
    df_diff = dfs.diff()
    df_diff = df_diff.iloc[1:,:]
    df_diff = pd.concat([dfs.iloc[0:1,:],df_diff],axis=0)
    df = df_diff.copy()
    df_A = df.loc[:'1989',:].copy()
    df_B = df.loc['1990':,:].copy()

    df_A.loc[(df_A.index.month==1) & (df_A.index.day==1) & (df_A.index.hour==3),:] = dfs
    for month in [4,7,10]:
        df_A.loc[(df_A.index.month==month) & (df_A.index.day==1) & (df_A.index.hour==0),:] = dfs
    df_B.loc[(df_B.index.month%3==1) & (df_B.index.day==1) & (df_B.index.hour==3),:] = dfs
    dfs = pd.concat([df_A,df_B],axis=0)
    return dfs


def accugdrnoff_to_ugdrnoff(acc_ds):
    acc_ds_diff = acc_ds.diff('time',label='upper') # Difference along time dimension (keep upper label)
    ds_dt = xr.concat([acc_ds.isel(time=0),acc_ds_diff],dim='time')
    ds = ds_dt.where(~((ds_dt['time.month']%3 == 1) & (ds_dt['time.day'] == 1) & (ds_dt['time.hour'] == 3)), acc_ds)
    ds = ds.drop_isel(time=0)
    return ds

def acsnom_to_snom(acc_ds):
    acc_ds_diff = acc_ds.diff('time',label='upper') # Difference along time dimension (keep upper label)
    ds_dt = xr.concat([acc_ds.isel(time=0),acc_ds_diff],dim='time')
    ds = ds_dt.where(~((ds_dt['time.month']%3 == 1) & (ds_dt['time.day'] == 1) & (ds_dt['time.hour'] == 3)), acc_ds)
    ds = ds.drop_isel(time=0)
    return ds

def soil_m_to_sm(soil_m):
    sm = (soil_m.isel(soil_layers_stag=0)*100) + \
         (soil_m.isel(soil_layers_stag=1)*300) + \
         (soil_m.isel(soil_layers_stag=2)*600) + \
         (soil_m.isel(soil_layers_stag=3)*1000)
    return sm

def comid_to_spatialweights(comids,
                            ds_spatialweight,
                            x_coords,y_coords,
                            resolution=1000,
                            nx=4608,ny=3840):

    if resolution == 1000:
        nx = 4608
        ny = 3840
    elif resolution == 250:
        nx = 18432
        ny = 15360
    else:
        nx = nx
        ny = ny


    #ds_spatialweight = ds_spatialweight.sel(polyid=comids)

    spatial_weight = ds_spatialweight['weight'].to_dataframe()
    spatial_regridweight = ds_spatialweight['regridweight'].to_dataframe()
    spatial_IDmask = ds_spatialweight['IDmask'].to_dataframe()
    spatial_i_index = ds_spatialweight['i_index'].to_dataframe()
    spatial_j_index =  ds_spatialweight['j_index'].to_dataframe()

    df_spatialweight = pd.concat([spatial_i_index,
                                    spatial_j_index,
                                    spatial_IDmask,
                                    spatial_regridweight,
                                    spatial_weight], axis=1)
    df_spatialweight = df_spatialweight[df_spatialweight['IDmask'].isin(comids)]
    
    arr_regridweight = np.ones((ny,nx))
    arr_regridweight[arr_regridweight==1] = -9999

    arr_weight = np.ones((ny,nx))
    arr_weight[arr_weight==1] = -9999

    for row_index in df_spatialweight.index:
        #print(row_index)
        i_index = df_spatialweight.loc[row_index,'i_index']
        j_index = df_spatialweight.loc[row_index,'j_index']
        
        # Regridweight
        regridweight = df_spatialweight.loc[row_index,'regridweight']

        if arr_regridweight[j_index-1,i_index-1] == -9999:
            arr_regridweight[j_index-1,i_index-1] = regridweight
        else:
            arr_regridweight[j_index-1,i_index-1] = arr_regridweight[j_index-1,i_index-1] + regridweight


        # Weight
        weight = df_spatialweight.loc[row_index,'weight']

        if arr_weight[j_index-1,i_index-1] == -9999:
            arr_weight[j_index-1,i_index-1] = weight
        else:
            arr_weight[j_index-1,i_index-1] = arr_weight[j_index-1,i_index-1] + weight


    ds_spatialweight = xr.Dataset(
                                    data_vars={
                                            'regridweight':(['y','x'],arr_regridweight),
                                            'weight':(['y','x'],arr_weight),
                                            },
                                    coords = {
                                            'x':x_coords,
                                            'y':y_coords
                                            }
                                    )
    return ds_spatialweight



def calc_3_hourly_water_budget(ppt,accet,edir,soil_m,sneqv,ugdrnoff,sfcheadsubrt,gw_depth,streamflow,mask_lcc,mask_value,feature_ids,outlet_feature_id):
    # Resampling and Spatial Subsetting
    
    ppt = nwm.resample_and_mask(ppt,mask_lcc,mask_value=mask_value)
    # accet = nwm.resample_and_mask(accet,mask_lcc,mask_value=mask_value)
    # edir = nwm.resample_and_mask(edir,mask_lcc,mask_value=mask_value)
    # soil_m = nwm.resample_and_mask(soil_m,mask_lcc,mask_value=mask_value)
    # sneqv = nwm.resample_and_mask(sneqv,mask_lcc,mask_value=mask_value)
    # ugdrnoff = nwm.resample_and_mask(ugdrnoff,mask_lcc,mask_value=mask_value)
    # sfcheadsubrt = sfcheadsubrt.where(mask_lcc==mask_value)
    # gw_depth = gw_depth.sel(feature_id=feature_ids)
    # streamflow = streamflow.sel(feature_id=outlet_feature_id)

    # # Temporal Preprocessing
    # print('Temporal Preprocessing')
    # ppt = ppt.resample(time='3H').sum()
    # et = nwm.accet_to_et(accet)
    # sm = nwm.soil_m_to_sm(soil_m)
    #gw_depth = gw_depth.sel(time=ppt.time)
    #streamflow = streamflow.sel(time=ppt.time)
    #print(ppt)
    #print(accet)
    #print(gw_depth)
    #print(streamflow)
    # print('Done')
    return ppt#,et,edir,soil_m,sneqv,ugdrnoff,sfcheadsubrt,gw_depth,streamflow

def select_shp_within(input_layer,selection_layer):
    input_layer['centroid'] = input_layer.centroid
    input_layer = input_layer.set_geometry('centroid')
    selected_input_layer = input_layer[input_layer.within(selection_layer.geometry.unary_union)]
    selected_input_layer = selected_input_layer.set_geometry('geometry')
    selected_input_layer = gpd.GeoDataFrame(selected_input_layer.iloc[:,:-2],geometry=selected_input_layer['geometry'],crs=selected_input_layer.crs)
    selected_input_layer = selected_input_layer.set_geometry('geometry')
    return selected_input_layer

def resample_and_mask(ds,mask,mask_value=0):
    ds = ds.interp_like(mask,method='nearest')
    ds = ds.where(mask==mask_value)
    return ds

# NLDI by COMID
def get_basin_by_comid(comid):
    print('Reading NLDI Basin Boundary...')
    os.environ["HYRIVER_CACHE_DISABLE"] = "true" # If set to False (Default) PyNHD is very slow on ASU Cluster
    nldi = NLDI()
    basin = nldi.get_basins(comid,fsource='comid',simplified=False)
    return basin

def get_flw_main_by_comid(comid):
    print('Reading NLDI Flow Main...')
    os.environ["HYRIVER_CACHE_DISABLE"] = "true" # If set to False (Default) PyNHD is very slow on ASU Cluster
    nldi = NLDI()
    flw_main = nldi.navigate_byid(
        fsource="comid",
        fid=comid,
        navigation="upstreamMain",
        source="flowlines",
        distance=9999,
    )
    return flw_main

def get_flw_trib_by_comid(comid):
    print('Reading NLDI Flow Tributaries...')
    os.environ["HYRIVER_CACHE_DISABLE"] = "true" # If set to False (Default) PyNHD is very slow on ASU Cluster
    nldi = NLDI()
    flw_trib = nldi.navigate_byid(
        fsource="comid",
        fid=comid,
        navigation="upstreamTributaries",
        source="flowlines",
        distance=9999,
    )
    return flw_trib


# NLDI by USGS Gage ID
def get_basin_by_usgsgageid(station_id):
    print('Reading NLDI Basin Boundary...')
    os.environ["HYRIVER_CACHE_DISABLE"] = "true" # If set to False (Default) PyNHD is very slow on ASU Cluster
    nldi = NLDI()
    basin = nldi.get_basins(station_id,simplified=False)
    return basin

def get_flw_main_by_usgsgageid(station_id):
    print('Reading NLDI Flow Main...')
    os.environ["HYRIVER_CACHE_DISABLE"] = "true" # If set to False (Default) PyNHD is very slow on ASU Cluster
    nldi = NLDI()
    flw_main = nldi.navigate_byid(
        fsource="nwissite",
        fid=f"USGS-{station_id}",
        navigation="upstreamMain",
        source="flowlines",
        distance=9999,
    )
    return flw_main

def get_flw_trib_by_usgsgageid(station_id):
    print('Reading NLDI Flow Tributaries...')
    os.environ["HYRIVER_CACHE_DISABLE"] = "true" # If set to False (Default) PyNHD is very slow on ASU Cluster
    nldi = NLDI()
    flw_trib = nldi.navigate_byid(
        fsource="nwissite",
        fid=f"USGS-{station_id}",
        navigation="upstreamTributaries",
        source="flowlines",
        distance=9999,
    )
    return flw_trib

def get_basin_by_huc12pp(huc12pp):
    print('Reading NLDI Basin Boundary...')
    os.environ["HYRIVER_CACHE_DISABLE"] = "true" # If set to False (Default) PyNHD is very slow on ASU Cluster
    nldi = NLDI()
    basin = nldi.get_basins(fsource='huc12pp',feature_ids=huc12pp)
    return basin

def get_flw_main_by_huc12pp(huc12pp):
    print('Reading NLDI Flow Main...')
    os.environ["HYRIVER_CACHE_DISABLE"] = "true" # If set to False (Default) PyNHD is very slow on ASU Cluster
    nldi = NLDI()
    flw_main = nldi.navigate_byid(
        fsource="huc12pp",
        fid=huc12pp,
        navigation="upstreamMain",
        source="flowlines",
        distance=9999,
    )
    return flw_main

def get_flw_trib_by_huc12pp(huc12pp):
    print('Reading NLDI Flow Tributaries...')
    os.environ["HYRIVER_CACHE_DISABLE"] = "true" # If set to False (Default) PyNHD is very slow on ASU Cluster
    nldi = NLDI()
    flw_trib = nldi.navigate_byid(
        fsource="huc12pp",
        fid=huc12pp,
        navigation="upstreamTributaries",
        source="flowlines",
        distance=9999,
    )
    return flw_trib


def downscale_mask(mask_coarse,target_resolution,basin_lcc,crs):
    coarse_resolution = mask_coarse.x.isel(x=1) - mask_coarse.x.isel(x=0)
    
    x_coords_mask_start = mask_coarse.x.isel(x=0)-coarse_resolution*5
    x_coords_mask_end = mask_coarse.x.isel(x=-1)+coarse_resolution*5
    
    y_coords_mask_start = mask_coarse.y.isel(y=0)-coarse_resolution*5
    y_coords_mask_end = mask_coarse.y.isel(y=-1)+coarse_resolution*5
    
    x_coords_target_resolution = np.arange(x_coords_mask_start,
                                           x_coords_mask_end+target_resolution,
                                           target_resolution)
    y_coords_target_resolution = np.arange(y_coords_mask_start,
                                           y_coords_mask_end+target_resolution,
                                           target_resolution)
    mask_target_resolution = xr.DataArray(np.ones((len(y_coords_target_resolution),
                                                   len(x_coords_target_resolution))),
                                              coords={'y':y_coords_target_resolution,
                                                      'x':x_coords_target_resolution},
                                              dims=['y','x'])
    mask_target_resolution = mask_target_resolution.rio.write_crs(crs)
    mask_target_resolution = mask_target_resolution.rio.clip(basin_lcc.geometry)
    return mask_target_resolution

def huc8_to_outlet_comid(huc8_id,
                         gdf_nhdplusv21_nhdflowline_network,
                         gdf_nhdplusv21_huc12):
    ''' 
    Inputs:
            huc8_id: HUC8 ID (integer)
            gdf_nhdplusv21_nhdflowline_network: NHDFlowline_Network layer from Seamless_Flattened_Lower48 dataset on EPA's website (GeoDataFrame)
            gdf_nhdplusv21_huc12: HUC12 layer from Seamless_Flattened_Lower48 dataset on EPA's website (GeoDataFrame)

    Output:
            gdf_huc12_outlet: Most downstream HUC12 basins along with their outlet comid

    Notes:
            This version currently also returns outlets for closed basins our multiple outlets
    '''

    # Select all HUC12s in a HUC8
    gdf_huc12 = gdf_nhdplusv21_huc12[gdf_nhdplusv21_huc12['HUC_8']==huc8_id]
    
    # Select all HUC12s with no Downstream HUC12 in the HUC8
    gdf_huc12_outlet = gdf_huc12[~gdf_huc12['HU_12_DS'].isin(gdf_huc12['HUC_12'])].copy()
    gdf_huc12_nhdflowline = []
    #gdf_huc12_outlet.loc[:,'OUT_COMID'] = np.nan

    for huc12_outlet_i in gdf_huc12_outlet.index:
        print(huc12_outlet_i)
        huc12_id = gdf_huc12_outlet.loc[huc12_outlet_i,'HUC_12']
        
        try:
            gdf_comid_flw_trib = nwm.get_flw_trib_by_huc12pp(huc12_id)
            gdf_comid_flw_trib['nhdplus_comid'] = gdf_comid_flw_trib['nhdplus_comid'].astype('int')
            gdf_nhdplusv21_nhdflowline_network_huc12 = gdf_nhdplusv21_nhdflowline_network[gdf_nhdplusv21_nhdflowline_network['COMID'].isin(gdf_comid_flw_trib['nhdplus_comid'])]
            gdf_nhdplusv21_nhdflowline_network_huc12.loc[:,'HUC_8'] = huc8_id
            gdf_nhdplusv21_nhdflowline_network_huc12.loc[:,'HUC_12'] = huc12_id
            gdf_huc12_nhdflowline.append(gdf_nhdplusv21_nhdflowline_network_huc12)

            gdf_comid_outlet = gdf_nhdplusv21_nhdflowline_network_huc12[gdf_nhdplusv21_nhdflowline_network_huc12['TotDASqKM']==gdf_nhdplusv21_nhdflowline_network_huc12['TotDASqKM'].max()]
            gdf_huc12_outlet.loc[huc12_outlet_i,'OUT_COMID'] = int(gdf_comid_outlet['COMID'].values[0])
        except:
            pass

    if len(gdf_huc12_nhdflowline!=0):
        gdf_huc12_nhdflowline = pd.concat(gdf_huc12_nhdflowline)
    else:
        gdf_huc12_nhdflowline = None
    return {'outlet':gdf_huc12_outlet,
            'nhdflowline':gdf_huc12_nhdflowline}

def huc8_to_outlet_nhdflowline(huc8_id,
                        gdf_nhdplusv21_nhdflowline_network,
                        gdf_nhdplusv21_huc12):
    ''' 
    Inputs:
            huc8_id: HUC8 ID (integer)
            gdf_nhdplusv21_nhdflowline_network: NHDFlowline_Network layer from Seamless_Flattened_Lower48 dataset on EPA's website (GeoDataFrame)
            gdf_nhdplusv21_huc12: HUC12 layer from Seamless_Flattened_Lower48 dataset on EPA's website (GeoDataFrame)

    Output:
            gdf_nhdflowline: NHDFlowline for all HUC12 basins

    Notes:
            This version currently also returns outlets for closed basins our multiple outlets
    '''

    # Select all HUC12s in a HUC8
    gdf_huc12 = gdf_nhdplusv21_huc12[gdf_nhdplusv21_huc12['HUC_8']==huc8_id]
    
    gdf_comid_flw_trib_huc8 = []

    for huc12_i in gdf_huc12.index:
        huc12_id = gdf_huc12.loc[huc12_i,'HUC_12']
        try:
            gdf_comid_flw_trib = nwm.get_flw_trib_by_huc12pp(huc12_id)
            gdf_comid_flw_trib['nhdplus_comid'] = gdf_comid_flw_trib['nhdplus_comid'].astype('int')
            gdf_comid_flw_trib.loc[:,'HUC_12'] = huc12_id
            gdf_comid_flw_trib.loc[:,'HUC_8'] = huc8_id
            gdf_comid_flw_trib_huc8.append(gdf_comid_flw_trib)
        except:
            pass
    gdf_comid_flw_trib_huc8 = pd.concat(gdf_comid_flw_trib_huc8)
    return gdf_comid_flw_trib_huc8



def get_basin_properties_by_comid(comid,
                         gdf_nwm_reaches_lcc,
                         gdf_nwm_catchments_lcc,
                         gdf_nwm_waterbodies_lcc,
                         envelope_250m_lcc,
                         envelope_1km_lcc,
                         physiographic_lcc,
                         routelink_lcc,
                         lcc_proj4_crs):
    print('Reading Basin Properties...')
    basin = get_basin_by_comid(comid)
    bounds = basin.total_bounds
    flw_main = get_flw_main_by_comid(comid)
    flw_trib = get_flw_trib_by_comid(comid)

    basin_lcc = basin.to_crs(lcc_proj4_crs)
    bounds_lcc = basin_lcc.total_bounds
    flw_main_lcc = flw_main.to_crs(lcc_proj4_crs)
    flw_trib_lcc = flw_trib.to_crs(lcc_proj4_crs)


    # Subsetting NWM Reaches, Catchments and Waterbodies
    print('Subsetting NWM Reaches...')
    gdf_nwm_reaches_selbasin_lcc = gdf_nwm_reaches_lcc[gdf_nwm_reaches_lcc['ID'].isin(flw_trib['nhdplus_comid'].astype('int64'))]
    
    print('Subsetting NWM Catchments...')
    gdf_nwm_catchments_selbasin_lcc = gdf_nwm_catchments_lcc[gdf_nwm_catchments_lcc['ID'].isin(flw_trib['nhdplus_comid'].astype('int64'))]
    
    print('Subsetting NWM Waterbodies...')
    gdf_nwm_waterbodies_selbasin_lcc = gdf_nwm_waterbodies_lcc # Note: Does not subset waterbodies in this version

    # Specify basin outlet
    basin_outlet_feature_id = comid


    # Generating Masks from Envelope
    print('Generating 250m Mask...')
    envelope_250m_lcc = envelope_250m_lcc['envelope'].rio.write_crs(lcc_proj4_crs)
    mask_250m_lcc = envelope_250m_lcc.rio.clip(basin_lcc.geometry,from_disk=True)
    
    print('Generating 1km Mask...')
    envelope_1km_lcc = envelope_1km_lcc['envelope'].rio.write_crs(lcc_proj4_crs)
    mask_1km_lcc = envelope_1km_lcc.rio.clip(basin_lcc.geometry,from_disk=True)
    
    print('Generating 10m Mask...')
    mask_10m_lcc = downscale_mask(mask_250m_lcc,10,basin_lcc,lcc_proj4_crs)
    
    # Reading NMW parameters
    print('Subsetting NWM Physiographic Variables...')
    physiographic_lcc = physiographic_lcc.sel(x=mask_250m_lcc['x'],
                                      y=mask_250m_lcc['y'])
    
    print('Subsetting NWM RouteLink Variables...')
    routelink_lcc = routelink_lcc.where(routelink_lcc['link'].isin(flw_trib['nhdplus_comid'].astype('int64').values),drop=True)

    results = {'basin':basin,
               'basin_lcc':basin_lcc,
               'bounds':bounds,
               'bounds_lcc':bounds_lcc,
               'flw_main':flw_main,
               'flw_main_lcc':flw_main_lcc,
               'flw_trib':flw_trib,
               'flw_trib_lcc':flw_trib_lcc,
               'nwm_reaches_lcc':gdf_nwm_reaches_selbasin_lcc,
               'nwm_catchments_lcc':gdf_nwm_catchments_selbasin_lcc,
               'nwm_waterbodies_lcc':gdf_nwm_waterbodies_selbasin_lcc,
               'outlet_feature_id':basin_outlet_feature_id,
               'mask_250m_lcc':mask_250m_lcc,
               'mask_1km_lcc':mask_1km_lcc,
               'mask_10m_lcc':mask_10m_lcc,
               'ngrids_250m_lcc':int(mask_250m_lcc.count().values),
               'ngrids_1km_lcc':int(mask_1km_lcc.count().values),
               'ngrids_10m_lcc':int(mask_10m_lcc.count().values),
               'darea_250m_lcc_km2':int(mask_250m_lcc.count().values)*(0.25*0.25),
               'darea_1km_lcc_km2':int(mask_1km_lcc.count().values)*(1*1),
               'darea_10m_lcc_km2':int(mask_10m_lcc.count().values)*(0.01*0.01),
               'darea_shape_km2':basin_lcc.area.values[0]/(1000*1000),
               'physiographic_lcc':physiographic_lcc,
               'routelink_lcc':routelink_lcc}
    return results


def get_basin_properties_by_usgsgageid(station_id,
                         gdf_nwm_reaches_lcc,
                         gdf_nwm_catchments_lcc,
                         gdf_nwm_waterbodies_lcc,
                         gdf_nhdgage_lcc,
                         envelope_250m_lcc,
                         envelope_1km_lcc,
                         physiographic_lcc,
                         routelink_lcc,
                         lcc_proj4_crs):
    print('Reading Basin Properties...')
    basin = get_basin_by_usgsgageid(station_id)
    bounds = basin.total_bounds
    flw_main = get_flw_main_by_usgsgageid(station_id)
    flw_trib = get_flw_trib_by_usgsgageid(station_id)

    basin_lcc = basin.to_crs(lcc_proj4_crs)
    bounds_lcc = basin_lcc.total_bounds
    flw_main_lcc = flw_main.to_crs(lcc_proj4_crs)
    flw_trib_lcc = flw_trib.to_crs(lcc_proj4_crs)
    
    # Subsetting NWM Reaches, Catchments and Waterbodies
    print('Subsetting NWM Reaches...')
    gdf_nwm_reaches_selbasin_lcc = gdf_nwm_reaches_lcc[gdf_nwm_reaches_lcc['ID'].isin(flw_trib['nhdplus_comid'].astype('int64'))]
    
    print('Subsetting NWM Catchments...')
    gdf_nwm_catchments_selbasin_lcc = gdf_nwm_catchments_lcc[gdf_nwm_catchments_lcc['ID'].isin(flw_trib['nhdplus_comid'].astype('int64'))]
    
    print('Subsetting NWM Waterbodies...')
    gdf_nwm_waterbodies_selbasin_lcc = gdf_nwm_waterbodies_lcc # Note: Does not subset waterbodies in this version

    #gdf_basin_outlet = (gdf_nhdgage_lcc[gdf_nhdgage_lcc['SOURCE_FEA'].isin([station_id])])
    #basin_outlet_feature_id = gdf_basin_outlet['FLComID'].values[0]
    
    envelope_250m_lcc = envelope_250m_lcc['envelope'].rio.write_crs(lcc_proj4_crs)
    envelope_1km_lcc = envelope_1km_lcc['envelope'].rio.write_crs(lcc_proj4_crs)

    mask_250m_lcc = envelope_250m_lcc.rio.clip(basin_lcc.geometry)
    mask_1km_lcc = envelope_1km_lcc.rio.clip(basin_lcc.geometry)

    physiographic_lcc = physiographic_lcc.sel(x=mask_250m_lcc['x'],
                                      y=mask_250m_lcc['y'])
    physiographic_lcc['TOPOGRAPHY'] = physiographic_lcc['TOPOGRAPHY'].where(physiographic_lcc['TOPOGRAPHY'] != -9999.0)

    routelink_lcc = routelink_lcc.where(routelink_lcc['link'].isin(flw_trib['nhdplus_comid'].astype('int64').values),drop=True)

    results = {'basin':basin,
               'basin_lcc':basin_lcc,
               'bounds':bounds,
               'bounds_lcc':bounds_lcc,
               'flw_main':flw_main,
               'flw_main_lcc':flw_main_lcc,
               'flw_trib':flw_trib,
               'flw_trib_lcc':flw_trib_lcc,
               'nwm_reaches_lcc':gdf_nwm_reaches_selbasin_lcc,
               'nwm_catchments_lcc':gdf_nwm_catchments_selbasin_lcc,
               'nwm_waterbodies_lcc':gdf_nwm_waterbodies_selbasin_lcc,
               #'outlet_lcc':gdf_basin_outlet,
               #'outlet_feature_id':basin_outlet_feature_id,
               'mask_250m_lcc':mask_250m_lcc,
               'mask_1km_lcc':mask_1km_lcc,
               'ngrids_250m_lcc':int(mask_250m_lcc.count().values),
               'ngrids_1km_lcc':int(mask_1km_lcc.count().values),
               'darea_250m_lcc_km2':int(mask_250m_lcc.count().values)*(0.25*0.25),
               'darea_1km_lcc_km2':int(mask_1km_lcc.count().values)*(1*1),
               'physiographic_lcc':physiographic_lcc,
               'routelink_lcc':routelink_lcc}
    return results

def get_basin_properties_by_huc8id(huc8id,
                         shp_wbdhu12_lcc,
                         shp_wbdhu12_fpp_nad83,
                         shp_nwm_reaches_lcc,
                         shp_nwm_catchments_lcc,
                         shp_nwm_waterbodies_lcc,
                         envelope_250m_lcc,
                         envelope_1km_lcc,
                         physiographic_lcc,
                         routelink_lcc,
                         lcc_proj4_crs):
    # Note: This code is not yet finalized
    gdf_wbdhu12_lcc = pyogrio.read_dataframe(shp_wbdhu12_lcc)
    gdf_wbdhu12_wbdhu8_lcc = gdf_wbdhu12_lcc
    gdf_wbdhu12_wbdhu8_lcc['HUC8'] = gdf_wbdhu12_wbdhu8_lcc['HUC12'].str[:8]
    gdf_wbdhu12_selwbdhu8_lcc = gdf_wbdhu12_wbdhu8_lcc[gdf_wbdhu12_wbdhu8_lcc['HUC8']==huc8id]
    gdf_wbdhu12_selwbdhu8_outlet_lcc = gdf_wbdhu12_selwbdhu8_lcc[~gdf_wbdhu12_selwbdhu8_lcc['TOHUC'].isin(gdf_wbdhu12_selwbdhu8_lcc['HUC12'])]
    print(gdf_wbdhu12_selwbdhu8_outlet_lcc)
    if len(gdf_wbdhu12_selwbdhu8_outlet_lcc!=1):
        # TOHUC CODE Exceptions
        # OCEAN Hydrologic unit flows to an ocean, sea, or gulf (for example, Gulf of Mexico or Gulf of Alaska).
        # CANADA Hydrologic unit drains into Canada.
        # MEXICO Hydrologic unit drains into Mexico.
        # CLOSED BASIN Hydrologic unit is a closed basin with no outlet.
        TOHUC_CODE_Expections = ['CLOSED BASIN']
        gdf_wbdhu12_selwbdhu8_outlet_lcc = gdf_wbdhu12_selwbdhu8_outlet_lcc[~gdf_wbdhu12_selwbdhu8_outlet_lcc['TOHUC'].isin(TOHUC_CODE_Expections)]
        if len(gdf_wbdhu12_selwbdhu8_outlet_lcc) != 1:
            print('Error: More than 1 outlet found for HUC8')
            print(gdf_wbdhu12_selwbdhu8_outlet_lcc)
            sys.exit()
    huc12pp = gdf_wbdhu12_selwbdhu8_outlet_lcc['HUC12'].values[0]
    gdf_wbdhu12_fpp_lcc = (pyogrio.read_dataframe(shp_wbdhu12_fpp_nad83)).to_crs(lcc_proj4_crs)
    gdf_huc12pp_fpp_lcc = gdf_wbdhu12_fpp_lcc[gdf_wbdhu12_fpp_lcc['HUC_12']==huc12pp]

    print('Reading Basin Properties...')
    basin = get_basin_by_huc12pp(huc12pp)
    bounds = basin.total_bounds
    flw_main = get_flw_main_by_huc12pp(huc12pp)
    flw_trib = get_flw_trib_by_huc12pp(huc12pp)

    basin_lcc = basin.to_crs(lcc_proj4_crs)
    bounds_lcc = basin_lcc.total_bounds
    flw_main_lcc = flw_main.to_crs(lcc_proj4_crs)
    flw_trib_lcc = flw_trib.to_crs(lcc_proj4_crs)

    gdf_nwm_reaches_lcc = pyogrio.read_dataframe(shp_nwm_reaches_lcc)
    gdf_nwm_catchments_lcc = pyogrio.read_dataframe(shp_nwm_catchments_lcc)
    gdf_nwm_waterbodies_lcc = pyogrio.read_dataframe(shp_nwm_waterbodies_lcc)
    
    gdf_nwm_reaches_selbasin_lcc = gdf_nwm_reaches_lcc[gdf_nwm_reaches_lcc['feature_id'].isin(flw_trib['nhdplus_comid'].astype('int64'))]
    gdf_nwm_catchments_selbasin_lcc = gdf_nwm_catchments_lcc[gdf_nwm_catchments_lcc['feature_id'].isin(flw_trib['nhdplus_comid'].astype('int64'))]
    gdf_nwm_waterbodies_selbasin_lcc = gdf_nwm_waterbodies_lcc # Note: Does not subset waterbodies in this version

    gdf_basin_outlet = gdf_huc12pp_fpp_lcc
    basin_outlet_feature_id = huc12pp
    
    envelope_250m_lcc = xr.open_dataset(envelope_250m_lcc)['envelope'].rio.write_crs(lcc_proj4_crs)
    envelope_1km_lcc = xr.open_dataset(envelope_1km_lcc)['envelope'].rio.write_crs(lcc_proj4_crs)

    mask_250m_lcc = envelope_250m_lcc.rio.clip(basin_lcc.geometry)
    mask_1km_lcc = envelope_1km_lcc.rio.clip(basin_lcc.geometry)

    physiographic_lcc = xr.open_dataset(physiographic_lcc).sel(x=mask_250m_lcc['x'],
                                      y=mask_250m_lcc['y'])
    routelink_lcc = xr.open_dataset(routelink_lcc)
    routelink_lcc = routelink_lcc.where(routelink_lcc['link'].isin(flw_trib['nhdplus_comid'].astype('int64').values),drop=True)


    results = {'basin':basin,
               'basin_lcc':basin_lcc,
               'bounds':bounds,
               'bounds_lcc':bounds_lcc,
               'flw_main':flw_main,
               'flw_main_lcc':flw_main_lcc,
               'flw_trib':flw_trib,
               'flw_trib_lcc':flw_trib_lcc,
               'nwm_reaches_lcc':gdf_nwm_reaches_selbasin_lcc,
               'nwm_catchments_lcc':gdf_nwm_catchments_selbasin_lcc,
               'nwm_waterbodies_lcc':gdf_nwm_waterbodies_selbasin_lcc,
               'outlet_lcc':gdf_basin_outlet,
               'outlet_feature_id':basin_outlet_feature_id,
               'mask_250m_lcc':mask_250m_lcc,
               'mask_1km_lcc':mask_1km_lcc,
               'ngrids_250m_lcc':int(mask_250m_lcc.count().values),
               'ngrids_1km_lcc':int(mask_1km_lcc.count().values),
               'darea_250m_lcc_km2':int(mask_250m_lcc.count().values)*(0.25*0.25),
               'darea_1km_lcc_km2':int(mask_1km_lcc.count().values)*(1*1),
               'physiographic_lcc':physiographic_lcc,
               'routelink_lcc':routelink_lcc}
    return results

def get_basin_properties_by_huc12id(huc12pp,
                         shp_nwm_reaches_lcc,
                         shp_nwm_catchments_lcc,
                         shp_nwm_waterbodies_lcc,
                         envelope_250m_lcc,
                         envelope_1km_lcc,
                         physiographic_lcc,
                         routelink_lcc,
                         lcc_proj4_crs):
    # Note: This code is not yet finalized
    print('Reading Basin Properties...')
    basin = get_basin_by_huc12pp(huc12pp)
    bounds = basin.total_bounds
    flw_main = get_flw_main_by_huc12pp(huc12pp)
    flw_trib = get_flw_trib_by_huc12pp(huc12pp)

    basin_lcc = basin.to_crs(lcc_proj4_crs)
    bounds_lcc = basin_lcc.total_bounds
    flw_main_lcc = flw_main.to_crs(lcc_proj4_crs)
    flw_trib_lcc = flw_trib.to_crs(lcc_proj4_crs)

    gdf_nwm_reaches_lcc = pyogrio.read_dataframe(shp_nwm_reaches_lcc)
    gdf_nwm_catchments_lcc = pyogrio.read_dataframe(shp_nwm_catchments_lcc)
    gdf_nwm_waterbodies_lcc = pyogrio.read_dataframe(shp_nwm_waterbodies_lcc)
    
    gdf_nwm_reaches_selbasin_lcc = gdf_nwm_reaches_lcc[gdf_nwm_reaches_lcc['feature_id'].isin(flw_trib['nhdplus_comid'].astype('int64'))]
    gdf_nwm_catchments_selbasin_lcc = gdf_nwm_catchments_lcc[gdf_nwm_catchments_lcc['feature_id'].isin(flw_trib['nhdplus_comid'].astype('int64'))]
    gdf_nwm_waterbodies_selbasin_lcc = gdf_nwm_waterbodies_lcc # Note: Does not subset waterbodies in this version

    gdf_basin_outlet = huc12pp
    basin_outlet_feature_id = huc12pp
    
    envelope_250m_lcc = xr.open_dataset(envelope_250m_lcc)['envelope'].rio.write_crs(lcc_proj4_crs)
    envelope_1km_lcc = xr.open_dataset(envelope_1km_lcc)['envelope'].rio.write_crs(lcc_proj4_crs)

    mask_250m_lcc = envelope_250m_lcc.rio.clip(basin_lcc.geometry)
    mask_1km_lcc = envelope_1km_lcc.rio.clip(basin_lcc.geometry)

    physiographic_lcc = xr.open_dataset(physiographic_lcc).sel(x=mask_250m_lcc['x'],
                                      y=mask_250m_lcc['y'])
    routelink_lcc = xr.open_dataset(routelink_lcc)
    routelink_lcc = routelink_lcc.where(routelink_lcc['link'].isin(flw_trib['nhdplus_comid'].astype('int64').values),drop=True)


    results = {'basin':basin,
               'basin_lcc':basin_lcc,
               'bounds':bounds,
               'bounds_lcc':bounds_lcc,
               'flw_main':flw_main,
               'flw_main_lcc':flw_main_lcc,
               'flw_trib':flw_trib,
               'flw_trib_lcc':flw_trib_lcc,
               'nwm_reaches_lcc':gdf_nwm_reaches_selbasin_lcc,
               'nwm_catchments_lcc':gdf_nwm_catchments_selbasin_lcc,
               'nwm_waterbodies_lcc':gdf_nwm_waterbodies_selbasin_lcc,
               'outlet_lcc':gdf_basin_outlet,
               'outlet_feature_id':basin_outlet_feature_id,
               'mask_250m_lcc':mask_250m_lcc,
               'mask_1km_lcc':mask_1km_lcc,
               'ngrids_250m_lcc':int(mask_250m_lcc.count().values),
               'ngrids_1km_lcc':int(mask_1km_lcc.count().values),
               'darea_250m_lcc_km2':int(mask_250m_lcc.count().values)*(0.25*0.25),
               'darea_1km_lcc_km2':int(mask_1km_lcc.count().values)*(1*1),
               'physiographic_lcc':physiographic_lcc,
               'routelink_lcc':routelink_lcc}
    return results




if __name__ == "__main__":
    print('test')
    