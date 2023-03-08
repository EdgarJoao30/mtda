import geopandas as gpd
import pystac_client
import stackstac
import planetary_computer
import xarray as xr
import numpy as np
import pandas as pd
import rioxarray
from tqdm import tqdm
import warnings

# geometry_path = './Koumbia_db/Koumbia_JECAM_2018-20-21.shp'
# bands = ['B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B11', 'B12']
# year = '2018'

def extract_ts(geometry_path, bands, year, subset, output_path):
    aoi = gpd.read_file(geometry_path)
    aoi_4326 = aoi.to_crs('EPSG: 4326')
    bbox_4326 = aoi_4326.total_bounds
    bands = bands
    year = year
    
    print('Preprocessing year: {}'.format(year))
    print('Subset: {}'.format(subset))
    print('Output folder: {}'.format(output_path))
    
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
        )

    search = catalog.search(
            collections=["sentinel-2-l2a"],
            bbox= bbox_4326,
            datetime= year + '-01-01/' + year + '-12-31',
            query={
                    #'eo:cloud_cover': {"lt": 100}, 
                    's2:nodata_pixel_percentage': {'lt': 50},
                    's2:mgrs_tile': {'eq': '30PVT'}},
            )
    item = search.item_collection()
    time_steps_pc = len(item)
    
    if subset == 'yes':
        bbox = [415000, 1240000, 415000+1000, 1240000+1000]
    else:    
        bbox = aoi.total_bounds
        
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if subset == 'yes':
            stack = stackstac.stack(item, resolution = 10, bounds = bbox, chunksize= (time_steps_pc, 1, 100, 100))
        else:    
            stack = stackstac.stack(item, resolution = 10, bounds = bbox, chunksize= (time_steps_pc, 1, 'auto', 'auto'))
    
    stack = stack.drop_duplicates(dim = 'time')
    SCL = stack.sel(band = 'SCL')
    stack = stack.sel(band = bands)

    cirrus = xr.where(SCL == 10, 1, 0) # cirrus pixels, binary mask 0, 1
    highclouds = xr.where(SCL == 9, 1, 0) # High probability clouds, binary mask 0, 1
    medclouds = xr.where(SCL == 8, 1, 0) # Medium probability clouds, binary mask 0, 1
    unclass = xr.where(SCL == 7, 1, 0) # unclassified pixels, binary mask 0, 1
    shaclouds = xr.where(SCL == 3, 1, 0) # Cloud shadows, binary mask 0, 1
    saturated = xr.where(SCL == 1, 1, 0) # saturated pixels, binary mask 0, 1

    mask = highclouds + medclouds + shaclouds + saturated + cirrus + unclass # Mask, binary mask 0, 1
    maskedstack = stack.where(mask == 0, np.nan)
    
    print('Masking collection complete')
    
    corr = 5 - maskedstack.time.dt.dayofyear.values[0]
    pc_idx = (maskedstack.time.dt.dayofyear.values + corr) / 5
    missing_idx = np.setdiff1d(np.arange(5, 370, 5) / 5, pc_idx)
    n_missing = missing_idx.shape[0]
    
    pcdata = np.array(maskedstack.values)
    
    print('Loading images into memory complete')
    
    n_bands = pcdata.shape[1]
    n_y = pcdata.shape[2]
    n_x = pcdata.shape[3]
    nandata = np.array([np.nan]*n_missing*n_bands*n_y*n_x).reshape(n_missing, n_bands, n_y, n_x)
    completets = np.zeros((73, n_bands, n_y, n_x))
    
    for n, i in enumerate(pc_idx):
        completets[int(i)-1, :, :, :] = pcdata[n, :, :, :]
    
    for n, i in enumerate(missing_idx):
        completets[int(i)-1, :, :, :] = nandata[n, :, :, :]
    
    print('Complete TS in numpy')
    
    x = maskedstack.indexes['x']
    y = maskedstack.indexes['y']
    band = maskedstack.indexes['band']
    dates = [pd.to_datetime(doy-1, unit='D', origin=str(2018)) for doy in np.arange(5, 370, 5)]
    time = pd.Index(dates, name = 'time')
    
    completets = xr.DataArray(
        data = completets,
        coords=dict(time = time,
                    band = band,
                    y = y,
                    x = x)
        )
    completets._copy_attrs_from(maskedstack)
    
    print('Complete TS in DataArray')
    
    interpolated = completets.interpolate_na(dim="time", method="linear", use_coordinate = 'time')
    
    print('Interpolation complete')
    
    minmax_bandtime = interpolated.quantile([0.02, 0.98], dim=['x', 'y'])
    normalized = (interpolated - minmax_bandtime[0, :, :]) / (minmax_bandtime[1, :, :] - minmax_bandtime[0, :, :])
    normalized = normalized.where(normalized > 0, 0, drop=True)
    normalized = normalized.where(normalized < 1, 1, drop=True)
    
    print('Normalization complete')
    
    for t in tqdm(np.arange(0, 73, 1)):
        for b in np.arange(0, 10, 1):
            normalized[t, b, :, :].rio.to_raster("{}s2_{}_{}.tif".format(output_path, time[t].strftime('%Y%m%d'), bands[b]))
    
    
if __name__ == "__main__":
    
    # parse arguments
    import argparse
    from argparse import RawTextHelpFormatter

    parser = argparse.ArgumentParser(
        description="Preprocess and write Sentinel 2 images time series",
        formatter_class=RawTextHelpFormatter,
    )
    parser.add_argument(
        "--geometry_path",
        type=str,
        default='/home/edgar/DATA/Koumbia_db/Koumbia_JECAM_2018-20-21.shp',
    )
    
    parser.add_argument(
        "--bands",
        type=list,
        default=['B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B11', 'B12'],
    )
    
    parser.add_argument(
        "--year",
        type=str,
        default='2018',
    )
    
    parser.add_argument(
        "--subset",
        type=str,
        default='yes',
    )
    
    parser.add_argument(
        "--output_path",
        type=str,
        default='/home/edgar/DATA/testimages/',
    )
    
    config = vars(parser.parse_args())

    # Write Sentinel 2 images
    extract_ts(**config)
    
