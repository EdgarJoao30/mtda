from torch.utils.data import DataLoader, TensorDataset
from pytorch_lightning import LightningDataModule
import numpy as np
import torch

class Koumbia_DataModule(LightningDataModule):
    def __init__(
        self,
        batch_size = 16,
        in_domain = '2018',
        out_domain = '2020',
        data_path = '/home/edgar/DATA/all_data/'
    ):
        super().__init__()
        self.batch_size = batch_size
        self.in_domain = in_domain
        self.out_domain = out_domain
        self.data_path = data_path

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        if stage == "fit":
            train_source_X = torch.as_tensor(np.load(f'{self.data_path}{self.in_domain}/{self.in_domain}_X.npy'), dtype=torch.float32)
            train_source_y = torch.as_tensor(np.load(f'{self.data_path}{self.in_domain}/{self.in_domain}_y.npy'), dtype=torch.int64)
            train_target_X = torch.as_tensor(np.load(f'{self.data_path}{self.out_domain}/{self.out_domain}_X.npy'), dtype=torch.float32)
            self.train_dataset = TensorDataset(train_source_X, train_source_y, train_target_X)
        elif stage == "test":
            self.test_dataset2018 = TensorDataset(
                torch.as_tensor(np.load(f'{self.data_path}2018/2018_X.npy'), dtype=torch.float32),
                torch.as_tensor(np.load(f'{self.data_path}2018/2018_y.npy'), dtype=torch.int64)
                )
            self.test_dataset2020 = TensorDataset(
                torch.as_tensor(np.load(f'{self.data_path}2020/2020_X.npy'), dtype=torch.float32),
                torch.as_tensor(np.load(f'{self.data_path}2020/2020_y.npy'), dtype=torch.int64)
                )
            self.test_dataset2021 = TensorDataset(
                torch.as_tensor(np.load(f'{self.data_path}2021/2021_X.npy'), dtype=torch.float32),
                torch.as_tensor(np.load(f'{self.data_path}2021/2021_y.npy'), dtype=torch.int64)
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

    def test_dataloader(self):
        return [DataLoader(dataset=self.test_dataset2018,batch_size=self.batch_size,shuffle=False,num_workers=4),
                DataLoader(dataset=self.test_dataset2020,batch_size=self.batch_size,shuffle=False,num_workers=4),
                DataLoader(dataset=self.test_dataset2021,batch_size=self.batch_size,shuffle=False,num_workers=4),
               ]
