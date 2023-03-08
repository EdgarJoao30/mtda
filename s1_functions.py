import planetary_computer
import pystac_client
import geopandas as gpd
import pandas as pd
import stackstac
import numpy as np

aoi = gpd.read_file('../../2_data/Koumbia_db/Koumbia_JECAM_2018-20-21.shp')
aoi = aoi.to_crs('EPSG: 4326')
bbox = aoi.total_bounds # add buffer
year = '2018'
time = year + '-01-01/' + year + '-12-31'

catalog = pystac_client.Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1",
    modifier=planetary_computer.sign_inplace,
)

def getStack(time): # Compute AOI level cloud cover and returns 
    print('Processing year:', time)
    # complete item
    search = getSearch(bbox= bbox, time = time)
    items = search.item_collection()
    stack = stackstac.stack(items, resolution = 10, bounds_latlon = bbox, epsg=32630) # Convert item collection into Stackstac xarray object with 10m spatial resolution and clipped to AOI
    
    return stack

def getSearch(bbox = bbox, time = time): # Searches All s2 collection only based on bbox, 
        # nodata and mgrs_tile. Cloud cover is not used so we can calculate AOI based 
        # cloud cover of all the s2 collection.
        s1_search = catalog.search(
        collections=["sentinel-1-rtc"],
        bbox=bbox,
        datetime=time,
        query={"sat:orbit_state": {"eq": 'ascending'},
                'platform': {'eq': 'SENTINEL-1A'}}
        )
        return s1_search

def getDOY(stack):
    df = pd.DataFrame({'datetime': list(stack.indexes['time'])})
    df["datetime"] = pd.to_datetime(df["datetime"])
    df['doy'] = pd.DatetimeIndex(df['datetime']).dayofyear
    return np.array(df['doy'])