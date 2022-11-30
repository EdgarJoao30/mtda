import planetary_computer
import pystac_client
import geopandas as gpd
import pandas as pd
import stackstac
import numpy as np
import warnings

aoi = gpd.read_file('../../2_data/Koumbia_db/Koumbia_JECAM_2018-20-21.shp')
aoi = aoi.to_crs('EPSG: 4326')
bbox = aoi.total_bounds
bands = ['B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B11', 'B12']
year = '2018'
time = year + '-01-01/' + year + '-12-31'

# Open PC catalog
catalog = pystac_client.Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1",
    modifier=planetary_computer.sign_inplace,
    )

def getReducedStack(cloud_cover, time): # Compute AOI level cloud cover and returns 
    print('Processing year:', time)
    # xarray object (Stackstac) based on cloud cover threshold
    aoi = gpd.read_file('../../2_data/Koumbia_db/Koumbia_JECAM_2018-20-21.shp') # reload data with correct CRS for stackstac clipping
    bbox = aoi.total_bounds
    # complete item
    search = getSearch(cloud_cover=100, time = time)
    complete_item = search.item_collection()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stack = stackstac.stack(complete_item, resolution = 10, bounds = bbox) # Convert item collection into Stackstac xarray object with 10m spatial resolution and clipped to AOI
    SCL = stack.sel(band = 'SCL') # Select SCL band
    # Mask based on SCL band 
    highclouds = SCL.where(SCL == 9, 0) / 9 # High probability clouds, binary mask 0, 1
    medclouds = SCL.where(SCL == 8, 0) / 8 # Medium probability clouds, binary mask 0, 1
    shaclouds = SCL.where(SCL == 3, 0) / 3 # Cloud shadows, binary mask 0, 1
    saturated = SCL.where(SCL == 1, 0) / 1 # saturated pixels, binary mask 0, 1
    mask = highclouds + medclouds + shaclouds + saturated # Mask, binary mask 0, 1
    # Compute cloud cover
    # Computes the sum of pixels of the cloud mask: count number of pixels
    # Divides by total number of pixels for one image
    # Multiplies by 100 to get values ranging from 0-100
    # Returns xarray with (time, 1) dimentions
    bbox_cloud_cover = ((mask.sum(axis = (1, 2)) / (highclouds.shape[1] * highclouds.shape[2]))*100).compute()
    # Time indexes
    # Filters the count xarray with cloud cover threshold and extract the indexes
    times = bbox_cloud_cover.where(bbox_cloud_cover < cloud_cover).dropna('time').indexes
    print('Processing complete')
    # Filters the FULL s2 collection with the time indexes
    return stack.loc[times].sel(band = bands), np.array(bbox_cloud_cover.loc[times]), SCL.loc[times], mask.loc[times]

def getSearch(bbox = bbox, time = time, cloud_cover = 15): # Searches All s2 collection only based on bbox, 
        # nodata and mgrs_tile. Cloud cover is not used so we can calculate AOI based 
        # cloud cover of all the s2 collection.
        s2_search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox= bbox,
        datetime= time,
        query={'eo:cloud_cover': {"lt": cloud_cover}, 
                's2:nodata_pixel_percentage': {'lt': 50},
                's2:mgrs_tile': {'eq': '30PVT'}},
        )
        return s2_search

def getDOY(stack):
    df = pd.DataFrame({'datetime': list(stack.indexes['time'])})
    df["datetime"] = pd.to_datetime(df["datetime"])
    df['doy'] = pd.DatetimeIndex(df['datetime']).dayofyear
    return np.array(df['doy'])