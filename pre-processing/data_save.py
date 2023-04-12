import rioxarray
import numpy as np
import geopandas as gpd
import glob

def get_arrays(data_path, aoi_path, year):

    gt_label = rioxarray.open_rasterio(f"{data_path}GTimages/c{year}.tif")
    gt_idx = rioxarray.open_rasterio(f"{data_path}GTimages/polygon_ID.tif")
    geotiff_list = glob.glob(f'{data_path}Sentinel2images/s2_{year}_*.tif')

    aoi = gpd.read_file(aoi_path)
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
        if year == '2018':
            X = X[:, (3, 6, 4, 5, 8, 2, 0, 1, 9, 7), :]
        elif year == '2020':
            X = X[:, (5, 8, 1, 7, 0, 2, 9, 3, 1, 6), :]
        elif year == '2021':
            X = X[:, (5, 1, 9, 8, 3, 7, 4, 2, 0, 6), :]
        
        y = gt_label.where(gt_idx.isin(split.index.values))
        y = np.array(y).flatten()
        y = y[~np.isnan(y)]
        
        groups = np.array(gt_idx).flatten()
        groups = groups[~np.isnan(groups)]
        
        return X, y, groups

    X, y, groups = extract_samples_labels(geotiff_list, aoi)
    
    np.save(f'{data_path}all_data/{year}/{year}_X.npy', X)
    np.save(f'{data_path}all_data/{year}/{year}_y.npy', y)
    np.save(f'{data_path}all_data/{year}/{year}_groups.npy', groups)

if __name__ == "__main__":
    
    # parse arguments
    import argparse
    from argparse import RawTextHelpFormatter

    parser = argparse.ArgumentParser(
        description="Arrays time series data",
        formatter_class=RawTextHelpFormatter,
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default='/home/edgar/DATA/',
    )
    
    parser.add_argument(
        "--aoi_path",
        type=str,
        default='/home/edgar/DATA/Koumbia_db/Koumbia_JECAM_2018-20-21.shp',
    )
    
    parser.add_argument(
        "--year",
        type=str,
        default='2020',
    )
    
    config = vars(parser.parse_args())

    # Write arrays to disk
    get_arrays(**config)

