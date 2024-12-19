import os
import geopandas as gpd
import shapely
from shapely.geometry import Polygon
from functools import reduce

#-------------------------------Custom Functions-------------------------------
def poly_union(gdf):
    
    # Take Union
    gdf = gpd.GeoSeries(gdf.unary_union,crs=gdf.crs)
    
    # Remove Holes
    list_interiors = []
    for interior in gdf[0].interiors:
        p = shapely.geometry.Polygon(interior)
        if p.area > 0.0001: #Threshold area to remove holes (1m2)
            list_interiors.append(interior)
    new_polygon = shapely.geometry.Polygon(gdf[0].exterior.coords, holes=list_interiors)
    gdf = gpd.GeoSeries(new_polygon,crs=gdf.crs)
    return gdf
#------------------------------------------------------------------------------

def fillit(row):
    """A function to fill holes below an area threshold in a polygon"""
    newgeom=None
    rings = [i for i in row["geometry"].interiors] #List all interior rings
    if len(rings)>0: #If there are any rings
        to_fill = [Polygon(ring) for ring in rings if Polygon(ring).area<1000] #List the ones to fill
        if len(to_fill)>0: #If there are any to fill
            newgeom = reduce(lambda geom1, geom2: geom1.union(geom2),[row["geometry"]]+to_fill) #Union the original geometry with all holes
    if newgeom:
        return newgeom
    else:
        return row["geometry"]


#------------------------------------------------------------------------------
def makedir(path):
    os.makedirs(path,exist_ok=True)
    return path
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def unordered_lists_are_equal(list_a,list_b):
    return set(list_a) == set(list_b)
#------------------------------------------------------------------------------


