import os
import numpy as np

import torch

from utils import standardize


class TrainLoaders(object):

    def __init__(self, data_path, batch_size):
        super(TrainLoaders, self).__init__()
        self.data_path = data_path
        self.batch_size = batch_size

        self.train_loader_inn, self.val_loader_inn, self.train_loader_fcn, self.val_loader_fcn = self.construct_loaders()

    def construct_loaders(self):
        ### train data
        train_band  = torch.Tensor(np.load(os.path.join(self.data_path, 'x_train.npy')))
        train_e     = torch.Tensor(np.load(os.path.join(self.data_path, 'y_sigma_train.npy')))
        train_n     = torch.Tensor(np.load(os.path.join(self.data_path, 'y_n_train.npy')))
        train_S     = torch.Tensor(np.load(os.path.join(self.data_path, 'y_seebeck_train.npy')))
        ### val data
        val_band    = torch.Tensor(np.load(os.path.join(self.data_path, 'x_eval.npy')))
        val_e       = torch.Tensor(np.load(os.path.join(self.data_path, 'y_sigma_eval.npy')))
        val_n       = torch.Tensor(np.load(os.path.join(self.data_path, 'y_n_eval.npy')))
        val_S       = torch.Tensor(np.load(os.path.join(self.data_path, 'y_seebeck_eval.npy')))

        ### standardization of data points
        train_band_norm, val_band_norm  = standardize(train_band, val_band, "x")
        train_e_norm, val_e_norm        = standardize(train_e, val_e, "y_sigma")
        train_n_norm, val_n_norm        = standardize(train_n, val_n, "y_n")
        train_S_norm, val_S_norm        = standardize(train_S, val_S, "y_seebeck")

        ### create dataloaders
        train_loader_inn = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(train_band_norm, train_e_norm, train_n_norm, train_S_norm),
            batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)

        val_loader_inn = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(val_band_norm, val_e_norm, val_n_norm, val_S_norm),
            batch_size=self.batch_size, shuffle=False, drop_last=True)

        train_loader_fcn = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(train_band_norm, train_S_norm),
            batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)

        val_loader_fcn = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(val_band_norm, val_S_norm),
            batch_size=self.batch_size, shuffle=False, drop_last=True)

        return train_loader_inn, val_loader_inn, train_loader_fcn, val_loader_fcn

    def get_fcn_loaders(self):
        return self.train_loader_fcn, self.val_loader_fcn
    
    def get_inn_loaders(self):
        return self.train_loader_inn, self.val_loader_inn


class TestLoaders(object):

    def __init__(self, data_path, batch_size):
        super(TestLoaders, self).__init__()
        self.data_path = data_path
        self.batch_size = batch_size

    def get_dataloaders(self):
        
        self.test_band = torch.Tensor(np.load(os.path.join(self.data_path, 'x_test.npy')))
        self.band_mean = torch.load(os.path.join("mean_std", 'x_mean.pt'))
        self.band_std = torch.load(os.path.join("mean_std", 'x_std.pt'))
        test_band_norm = (self.test_band - self.band_mean) / self.band_std

        test_e = torch.Tensor(np.load(os.path.join(self.data_path, 'y_sigma_test.npy')))
        e_mean = torch.load(os.path.join("mean_std", 'y_sigma_mean.pt'))
        e_std = torch.load(os.path.join("mean_std", 'y_sigma_std.pt'))
        test_norm_e = (test_e - e_mean) / e_std

        test_n = torch.Tensor(np.load(os.path.join(self.data_path, 'y_n_test.npy')))
        n_mean = torch.load(os.path.join("mean_std", 'y_n_mean.pt'))
        n_std = torch.load(os.path.join("mean_std", 'y_n_std.pt'))
        test_norm_n = (test_n - n_mean) / n_std

        test_S = torch.Tensor(np.load(os.path.join(self.data_path, 'y_seebeck_test.npy')))
        S_mean = torch.load(os.path.join("mean_std", 'y_seebeck_mean.pt'))
        S_std = torch.load(os.path.join("mean_std", 'y_seebeck_std.pt'))
        test_norm_S = (test_S - S_mean) / S_std

        self.test_size = test_e.shape[0]
        print("Test dataset size: ", self.test_size)

        return torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(test_band_norm, test_norm_e, test_norm_n, test_norm_S),
            batch_size=self.batch_size, shuffle=False, drop_last=True)
