import os
import numpy as np

import torch
import torch.utils.data

import config as c


x_train_ = torch.Tensor(np.load(os.path.join(c.data_path, 'x_train.npy')))
y_train_ = torch.Tensor(np.load(os.path.join(c.data_path, 'y_train.npy')))
train_size = x_train_.shape[0]

x_val_ = torch.Tensor(np.load(os.path.join(c.data_path, 'x_val.npy')))
y_val_ = torch.Tensor(np.load(os.path.join(c.data_path, 'y_val.npy')))
val_size = x_val_.shape[0]

x = torch.cat((x_train_, x_val_), dim=0)
x_mean = x.mean(dim=0, keepdim=True)
x_std = x.std(dim=0, keepdim=True)
x_norm = (x - x_mean) / x_std
x_train = x_norm[:train_size,:]
x_test = x_norm[train_size:train_size+val_size,:]

y = torch.cat((y_train_, y_val_), dim=0)
y_mean = y.mean(dim=0, keepdim=True)
y_std = y.std(dim=0, keepdim=True)
y_norm = (y - y_mean) / y_std
y_train = y_norm[:train_size,:]
y_test = y_norm[train_size:train_size+val_size,:]

tr_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(x_train, y_train),
    batch_size=c.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)

tst_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(x_test, y_test),
    batch_size=c.batch_size, shuffle=False, drop_last=True)
