import os
import numpy as np

import torch
import torch.utils.data

import config as c

x_train = torch.Tensor(np.load(os.path.join(c.data_path, 'I_train.npy')))
x_train = x_train[:, None]
y_train = torch.Tensor(np.load(os.path.join(c.data_path, 'coeff_train.npy')))

x_test = torch.Tensor(np.load(os.path.join(c.data_path, 'I_test.npy')))
x_test = x_test[:, None]
y_test = torch.Tensor(np.load(os.path.join(c.data_path, 'coeff_test.npy')))

tr_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(x_train, y_train),
    batch_size=c.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)

tst_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(x_test, y_test),
    batch_size=c.batch_size, shuffle=False, drop_last=True)

