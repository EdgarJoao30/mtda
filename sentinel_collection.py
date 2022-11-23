from pystac.extensions.eo import EOExtension as eo
import pystac_client
import planetary_computer
import geopandas as gpd
import pandas as pd
import numpy as np
import stackstac

aoi = gpd.read_file('../../2_data/Koumbia_db/Koumbia_JECAM_2018-20-21.shp')
aoi = aoi.to_crs('EPSG: 4326')
time = "2018-01-01/2022-01-01"
bbox = aoi.total_bounds

# Open PC catalog
catalog = pystac_client.Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1",
    modifier=planetary_computer.sign_inplace,
    )

# Sentinel 2 collection
class s2:
    bbox = bbox # Change aoi as needed, but also disable s2:mgrs_tile query
    time = time # Change time interval as needed

    def __init__(self, cloud_cover): # Allows to create many s2 instances based on 
        # cloud cover thresholds, by using map
        self.cloud_cover = cloud_cover 

    def getSearchAll(self): # Searches All s2 collection only based on bbox, 
        # nodata and mgrs_tile. Cloud cover is not used so we can calculate AOI based 
        # cloud cover of all the s2 collection.
        s2_search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox= self.bbox,
        datetime= self.time,
        query={'eo:cloud_cover': {"lt": 100}, 
                's2:nodata_pixel_percentage': {'lt': 50},
                's2:mgrs_tile': {'eq': '30PVT'}},
        )
        return s2_search

    def getSearch(self): # Searches s2 collection based on Cloud cover thresholds, 
        # this cuts are done with the s2 metadata at tile level, this is only used 
        # for comparison reasons, not used for AOI cloud cover.
        s2_search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox= self.bbox,
        datetime= self.time,
        query={'eo:cloud_cover': {"lt": self.cloud_cover}, 
                's2:nodata_pixel_percentage': {'lt': 50},
                's2:mgrs_tile': {'eq': '30PVT'}},
        )
        return s2_search

    def getItemAll(self): # Converts the search to an item collection
        search = self.getSearchAll()
        return search.item_collection()

    def getItem(self): # Converts the search to an item collection
        search = self.getSearch()
        return search.item_collection()

    def getDFAll(self): # Converts the item collection metadata to a pandas DF
        item = self.getItemAll()
        df = gpd.GeoDataFrame.from_features(item.to_dict(), crs="epsg:4326")
        df["datetime"] = pd.to_datetime(df["datetime"])
        df['year'] = pd.DatetimeIndex(df['datetime']).year
        df['month'] = pd.DatetimeIndex(df['datetime']).month
        df['day'] = pd.DatetimeIndex(df['datetime']).dayofyear
        df['s2:mgrs_tile'] = df['s2:mgrs_tile'].astype('category')
        return df

    def getDF(self): # Converts the item collection metadata to a pandas DF
        item = self.getItem()
        df = gpd.GeoDataFrame.from_features(item.to_dict(), crs="epsg:4326")
        df["datetime"] = pd.to_datetime(df["datetime"])
        df['year'] = pd.DatetimeIndex(df['datetime']).year
        df['month'] = pd.DatetimeIndex(df['datetime']).month
        df['day'] = pd.DatetimeIndex(df['datetime']).dayofyear
        df['s2:mgrs_tile'] = df['s2:mgrs_tile'].astype('category')
        return df

    def getReducedStack(self): # Compute AOI level cloud cover and returns 
        # xarray object (Stackstac) based on cloud cover threshold
        aoi = gpd.read_file('../../2_data/Koumbia_db/Koumbia_JECAM_2018-20-21.shp') # reload data with correct CRS for stackstac clipping
        bbox = aoi.total_bounds
        stack = stackstac.stack(self.getItemAll(), resolution = 10, bounds = bbox) # Convert item collection into Stackstac xarray object with 10m spatial resolution and clipped to AOI
        SCL = stack.sel(band = 'SCL') # Select SCL band
        # Cloud mask based on SCL band 
        highclouds = SCL.where(SCL == 9, 0) / 9 # High probability clouds, binary mask 0, 1
        medclouds = SCL.where(SCL == 8, 0) / 8 # Medium probability clouds, binary mask 0, 1
        shaclouds = SCL.where(SCL == 3, 0) / 3 # Cloud shadows, binary mask 0, 1
        cloudmask = highclouds + medclouds + shaclouds # Cloud mask, binary mask 0, 1
        # Compute cloud cover
        # Computes the sum of pixels of the cloud mask: count number of pixels
        # Divides by total number of pixels for one image
        # Multiplies by 100 to get values ranging from 0-100
        # Returns xarray with (time, 1) dimentions
        count = ((cloudmask.sum(axis = (1, 2)) / (highclouds.shape[1] * highclouds.shape[2]))*100).compute()
        # Time indexes
        # Filters the count xarray with cloud cover threshold and extract the indexes
        times = count.where(count < self.cloud_cover).dropna('time').indexes
        # Filters the FULL s2 collection with the time indexes
        return stack.loc[times], count
