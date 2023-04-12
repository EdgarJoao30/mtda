
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

def split_data(data_path, year):
    # Load the data
    X = np.load(f'{data_path}{year}/{year}_X.npy')
    y = np.load(f'{data_path}{year}/{year}_y.npy')
    groups = np.load(f'{data_path}{year}/{year}_groups.npy')

    sorted_indices = groups.argsort()

    # use the sorted indices to reorder arrays
    X = X[sorted_indices]
    y = y[sorted_indices]
    groups = groups[sorted_indices]


    # Define the stratified group k-fold 
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    sgkf_inner = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)

    for i, (train_index, test_index) in enumerate(sgkf.split(X, y, groups)):
        print(f"Fold {i}:")
        print(f"  Test:  index={test_index}, size={len(test_index)}, proportion={len(test_index)/len(X)}")
        print(f"         group={groups[test_index]}")
        X_test, y_test, groups_test = X[test_index], y[test_index], groups[test_index]
        for n, (train_index, val_index) in enumerate(sgkf_inner.split(X[train_index], y[train_index], groups[train_index])):
            if n == 0:
                print(f"Fold {i}:")
                print(f"  Train: index={train_index}, size={len(train_index)}, proportion={len(train_index)/len(X)}")
                print(f"         group={groups[train_index]}")
                print(f"  validation:  index={val_index}, size={len(val_index)}, proportion={len(val_index)/len(X)}")
                print(f"         group={groups[val_index]}")
                X_val, y_val, groups_val = X[val_index], y[val_index], groups[val_index]
                X_train, y_train, groups_train = X[train_index], y[train_index], groups[train_index]
                
                np.save(f'{data_path}{year}/splits/{year}_x_train_{i}.npy', X_train)
                np.save(f'{data_path}{year}/splits/{year}_y_train_{i}.npy', y_train)
                np.save(f'{data_path}{year}/splits/{year}_groups_train_{i}.npy', groups_train)
                
                np.save(f'{data_path}{year}/splits/{year}_x_val_{i}.npy', X_val)
                np.save(f'{data_path}{year}/splits/{year}_y_val_{i}.npy', y_val)
                np.save(f'{data_path}{year}/splits/{year}_groups_val_{i}.npy', groups_val)
                
                np.save(f'{data_path}{year}/splits/{year}_x_test_{i}.npy', X_test)
                np.save(f'{data_path}{year}/splits/{year}_y_test_{i}.npy', y_test)
                np.save(f'{data_path}{year}/splits/{year}_groups_test_{i}.npy', groups_test)
                
if __name__ == "__main__":
    
    # parse arguments
    import argparse
    from argparse import RawTextHelpFormatter

    parser = argparse.ArgumentParser(
        description="Split time series data",
        formatter_class=RawTextHelpFormatter,
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default='/home/edgar/DATA/all_data/',
    )
    
    parser.add_argument(
        "--year",
        type=str,
        default='2018',
    )
    
    config = vars(parser.parse_args())

    # Write splits to disk
    split_data(**config)
    