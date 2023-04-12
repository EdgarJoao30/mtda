import geopandas as gpd
from geocube.api.core import make_geocube
import pystac_client
import stackstac
import planetary_computer
import numpy as np

aoi = gpd.read_file('/home/edgar/DATA/Koumbia_db/Koumbia_JECAM_2018-20-21.shp')
aoi['polygon_ID'] = np.arange(0,len(aoi),1)

aoi.rename(columns = {'CodeL2':'c2018', '20_CodeL2':'c2020','21_CodeL2':'c2021'}, inplace = True)
aoi = aoi[['c2018', 'c2020', 'c2021', 'polygon_ID', 'geometry']]

columns = list(aoi.columns[:-1])

aoi_4326 = aoi.to_crs('EPSG: 4326')
bbox = aoi_4326.total_bounds

catalog = pystac_client.Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1",
    modifier=planetary_computer.sign_inplace,
    )

search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox= bbox,
        datetime= '2018-01-01/2018-01-06',
        query={
                #'eo:cloud_cover': {"lt": 100}, 
                's2:nodata_pixel_percentage': {'lt': 50},
                's2:mgrs_tile': {'eq': '30PVT'}},
        )
item = search.item_collection()

stack = stackstac.stack(item, resolution = 10, bounds = aoi.total_bounds, assets=['B02'], xy_coords= 'center')

gt_grid = make_geocube(
    vector_data=aoi,
    like = stack,
)

#gt_grid = gt_grid.astype(np.uint16)

for column in columns:
    gt_grid[column].rio.to_raster("/home/edgar/DATA/GTimages/{}.tif".format(column))