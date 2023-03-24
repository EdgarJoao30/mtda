import rioxarray
import numpy as np
import geopandas as gpd
import glob

c2018 = rioxarray.open_rasterio("/home/edgar/DATA/GTimages/c2018.tif")
gt_idx = rioxarray.open_rasterio("/home/edgar/DATA/GTimages/polygon_ID.tif")
geotiff_list = glob.glob('/home/edgar/DATA/Sentinel2images/s2_2018_*.tif')

aoi = gpd.read_file('/home/edgar/DATA/Koumbia_db/Koumbia_JECAM_2018-20-21.shp')
aoi['polygon_ID'] = np.arange(0,len(aoi),1)
aoi.rename(columns = {'CodeL2':'c2018', '20_CodeL2':'c2020','21_CodeL2':'c2021'}, inplace = True)
aoi = aoi[['c2018', 'c2020', 'c2021', 'polygon_ID', 'geometry']]

def extract_samples(path, split):
    # open file
    X = rioxarray.open_rasterio(path)
    # normalization
    minmax = X.quantile([0.02, 0.98])
    X = (X - minmax[0]) / (minmax[1] - minmax[0])
    X = X.where(X > 0, 0)
    X = X.where(X < 1, 1)
    # masking pixels 
    X = X.where(gt_idx.isin(split.index.values).values)
    # reshape
    X = np.array(X)
    X = X[~np.isnan(X)].reshape(73, -1).T
    return X

def extract_samples_labels(path, split):
    X = [extract_samples(path, split) for path in geotiff_list]
    X = np.hstack(X)
    X = X.reshape(-1, 10, 73)
    X = X[:, (3, 6, 4, 5, 8, 2, 0, 1, 9, 7), :]
    
    y = c2018.where(gt_idx.isin(split.index.values))
    y = np.array(y).flatten()
    y = y[~np.isnan(y)]
    
    groups = np.array(gt_idx).flatten()
    groups = groups[~np.isnan(groups)]
    
    return X, y, groups

X, y, groups = extract_samples_labels(geotiff_list, aoi)

np.save('/home/edgar/DATA/splits/2018_X.npy', X)
np.save('/home/edgar/DATA/splits/2018_y.npy', y)
np.save('/home/edgar/DATA/splits/2018_groups.npy', groups)