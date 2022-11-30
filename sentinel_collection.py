import geopandas as gpd
import s2_functions as s2f

aoi = gpd.read_file('../../2_data/Koumbia_db/Koumbia_JECAM_2018-20-21.shp')
aoi = aoi.to_crs('EPSG: 4326')
year = '2018'
bbox = aoi.total_bounds

# Sentinel 2 collection
class s2:
    bbox = bbox # Change aoi as needed, but also disable s2:mgrs_tile query

    def __init__(self, cloud_cover, year = year, bbox = bbox): # Allows to create many s2 instances based on 
        # cloud cover thresholds, by using map
        self.cloud_cover = cloud_cover 
        self.bbox = bbox
        self.time = year + '-01-01/' + year + '-12-31'
        self.stack, self.bbox_cloud_cover, self.SCL, self.mask = s2f.getReducedStack(self.cloud_cover, self.time)
        self.doy = s2f.getDOY(self.stack)
        



