import os
import sys
import xarray as xr
import pandas as pd
import numpy as np
import geopandas as gpd
import pyogrio
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import nwm
import param_nwm3
import misc

def select_shp_within(input_layer,selection_layer):
    input_layer['centroid'] = input_layer.centroid
    input_layer = input_layer.set_geometry('centroid')
    selected_input_layer = input_layer[input_layer.within(selection_layer.geometry.unary_union)]
    selected_input_layer = selected_input_layer.set_geometry('geometry')
    selected_input_layer = gpd.GeoDataFrame(selected_input_layer.iloc[:,:-2],geometry=selected_input_layer['geometry'],crs=selected_input_layer.crs)
    selected_input_layer = selected_input_layer.set_geometry('geometry')
    return selected_input_layer

gdf_nwm_reaches_az_utm12n = pyogrio.read_dataframe(param_nwm3.shp_nwm_reaches_az_utm12n)
gdf_nwm_catchments_az_utm12n = pyogrio.read_dataframe(param_nwm3.shp_nwm_catchments_az_utm12n)
gdf_wbdhu8_basins_az_utm12n = pyogrio.read_dataframe(param_nwm3.shp_wbdhu8_basins_az_utm12n)


selected_reaches = select_shp_within(gdf_nwm_catchments_az_utm12n,gdf_wbdhu8_basins_az_utm12n.iloc[0:1,:])

fig, ax = plt.subplots()
gdf_wbdhu8_basins_az_utm12n.iloc[0:1,:].plot(ax=ax,facecolor=None,edgecolor='k')
selected_reaches.plot(ax=ax,edgecolor='k')

