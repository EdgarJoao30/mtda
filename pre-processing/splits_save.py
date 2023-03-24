import rioxarray
import xarray as xr
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import geopandas as gpd
import glob

c2018 = rioxarray.open_rasterio("/home/edgar/DATA/GTimages/c2018.tif")
gt_idx = rioxarray.open_rasterio("/home/edgar/DATA/GTimages/polygon_ID.tif")
geotiff_list = glob.glob('/home/edgar/DATA/Sentinel2images/s2_2018_*.tif')

aoi = gpd.read_file('/home/edgar/DATA/Koumbia_db/Koumbia_JECAM_2018-20-21.shp')
aoi['polygon_ID'] = np.arange(0,len(aoi),1)
aoi.rename(columns = {'CodeL2':'c2018', '20_CodeL2':'c2020','21_CodeL2':'c2021'}, inplace = True)
aoi = aoi[['c2018', 'c2020', 'c2021', 'polygon_ID', 'geometry']]

train = []
val = []
test = []

for random_state in range(5):
    train_temp, test_temp = train_test_split(aoi, test_size=0.3, random_state=random_state, stratify=aoi[['c2018']])
    train_temp, val_temp = train_test_split(train_temp, test_size=0.2/0.7, random_state=random_state, stratify=train_temp[['c2018']])
    train.append(train_temp)
    val.append(val_temp)
    test.append(test_temp)
    

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
    y = c2018.where(gt_idx.isin(split.index.values))
    y = np.array(y).flatten()
    y = y[~np.isnan(y)]
    return X, y

for n in tqdm(range(5)):
    X_train, y_train = extract_samples_labels(geotiff_list, train[n])
    X_val, y_val = extract_samples_labels(geotiff_list, val[n])
    X_test, y_test = extract_samples_labels(geotiff_list, test[n])
    
    np.save('/home/edgar/DATA/splits/2018_x_train_{}.npy'.format(n), X_train)
    np.save('/home/edgar/DATA/splits/2018_y_train_{}.npy'.format(n), y_train)
    
    np.save('/home/edgar/DATA/splits/2018_x_val_{}.npy'.format(n), X_val)
    np.save('/home/edgar/DATA/splits/2018_y_val_{}.npy'.format(n), y_val)
    
    np.save('/home/edgar/DATA/splits/2018_x_test_{}.npy'.format(n), X_test)
    np.save('/home/edgar/DATA/splits/2018_y_test_{}.npy'.format(n), y_test)