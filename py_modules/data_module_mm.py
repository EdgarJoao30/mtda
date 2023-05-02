from torch.utils.data import DataLoader, TensorDataset
from pytorch_lightning import LightningDataModule
import numpy as np
import torch

class Koumbia_DataModule(LightningDataModule):
    def __init__(
        self,
        batch_size = 16,
        split = 0,
        in_domain = '2018',
        data_path = '/home/edgar/DATA/all_data/'
    ):
        super().__init__()
        self.batch_size = batch_size
        self.split = split
        self.in_domain = in_domain
        self.data_path = data_path

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        if stage == "fit" or stage == "validate":
            self.train_dataset = TensorDataset(
                torch.as_tensor(np.load(f'{self.data_path}{self.in_domain}/splits/{self.in_domain}_x_train_{self.split}.npy'), dtype=torch.float32),
                torch.as_tensor(np.load(f'{self.data_path}{self.in_domain}/splits/s1_{self.in_domain}_x_train_{self.split}.npy'), dtype=torch.float32),
                torch.as_tensor(np.load(f'{self.data_path}{self.in_domain}/splits/{self.in_domain}_y_train_{self.split}.npy'), dtype=torch.int64)
                )

            self.val_dataset = TensorDataset(
                torch.as_tensor(np.load(f'{self.data_path}{self.in_domain}/splits/{self.in_domain}_x_val_{self.split}.npy'), dtype=torch.float32),
                torch.as_tensor(np.load(f'{self.data_path}{self.in_domain}/splits/s1_{self.in_domain}_x_val_{self.split}.npy'), dtype=torch.float32),
                torch.as_tensor(np.load(f'{self.data_path}{self.in_domain}/splits/{self.in_domain}_y_val_{self.split}.npy'), dtype=torch.int64)
                )

        elif stage == "test":
            self.test_dataset2018 = TensorDataset(
                torch.as_tensor(np.load(f'{self.data_path}2018/splits/2018_x_test_{self.split}.npy'), dtype=torch.float32),
                torch.as_tensor(np.load(f'{self.data_path}2018/splits/s1_2018_x_test_{self.split}.npy'), dtype=torch.float32),
                torch.as_tensor(np.load(f'{self.data_path}2018/splits/2018_y_test_{self.split}.npy'), dtype=torch.int64)
                )
            self.test_dataset2020 = TensorDataset(
                torch.as_tensor(np.load(f'{self.data_path}2020/splits/2020_x_test_{self.split}.npy'), dtype=torch.float32),
                torch.as_tensor(np.load(f'{self.data_path}2020/splits/s1_2020_x_test_{self.split}.npy'), dtype=torch.float32),
                torch.as_tensor(np.load(f'{self.data_path}2020/splits/2020_y_test_{self.split}.npy'), dtype=torch.int64)
                )
            self.test_dataset2021 = TensorDataset(
                torch.as_tensor(np.load(f'{self.data_path}2021/splits/2021_x_test_{self.split}.npy'), dtype=torch.float32),
                torch.as_tensor(np.load(f'{self.data_path}2021/splits/s1_2021_x_test_{self.split}.npy'), dtype=torch.float32),
                torch.as_tensor(np.load(f'{self.data_path}2021/splits/2021_y_test_{self.split}.npy'), dtype=torch.int64)
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
        return [DataLoader(dataset=self.test_dataset2018,batch_size=self.batch_size,shuffle=False,num_workers=4),
                DataLoader(dataset=self.test_dataset2020,batch_size=self.batch_size,shuffle=False,num_workers=4),
                DataLoader(dataset=self.test_dataset2021,batch_size=self.batch_size,shuffle=False,num_workers=4),
               ]
