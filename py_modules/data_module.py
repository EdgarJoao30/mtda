from torch.utils.data import DataLoader, TensorDataset
from pytorch_lightning import LightningDataModule
import numpy as np
import torch

class Koumbia_DataModule(LightningDataModule):
    def __init__(
        self,
        batch_size = 16,
        split = 0,
        data_path = '/home/edgar/DATA/splits/',
        year = '2018'
    ):
        super().__init__()
        self.batch_size = batch_size
        self.split = split
        self.data_path = data_path
        self.year = year

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        if stage == "fit" or stage == "validate":
            self.train_dataset = TensorDataset(
                torch.as_tensor(np.load('{}{}_x_train_{}.npy'.format(self.data_path, self.year, self.split)), dtype=torch.float32),
                torch.as_tensor(np.load('{}{}_y_train_{}.npy'.format(self.data_path, self.year, self.split)), dtype=torch.int64)
                )

            self.val_dataset = TensorDataset(
                torch.as_tensor(np.load('{}{}_x_val_{}.npy'.format(self.data_path, self.year, self.split)), dtype=torch.float32),
                torch.as_tensor(np.load('{}{}_y_val_{}.npy'.format(self.data_path, self.year, self.split)), dtype=torch.int64)
                )

        elif stage == "test":
            self.test_dataset = TensorDataset(
                torch.as_tensor(np.load('{}{}_x_test_{}.npy'.format(self.data_path, self.year, self.split)), dtype=torch.float32),
                torch.as_tensor(np.load('{}{}_y_test_{}.npy'.format(self.data_path, self.year, self.split)), dtype=torch.int64)
                )
            
        # elif stage == 'predict':
        #     self.predict_dataset = TensorDataset(
        #         torch.as_tensor(np.load('{}{}_x_predict_{}.npy'.format(self.data_path, self.year, self.split)), dtype=torch.float32),
        #         torch.as_tensor(np.load('{}{}_y_predict_{}.npy'.format(self.data_path, self.year, self.split)), dtype=torch.int64)
        #         )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
        )
