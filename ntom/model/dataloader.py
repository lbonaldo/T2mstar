import os
import numpy as np

import torch
import torch.utils.data

import config as c

y_train = torch.Tensor(np.load(os.path.join(c.data_path, 'I_train.npy')))
y_train = y_train[:, None]
x_train = torch.Tensor(np.load(os.path.join(c.data_path, 'coeff_train.npy')))

y_test = torch.Tensor(np.load(os.path.join(c.data_path, 'I_test.npy')))
y_test = y_test[:, None]
x_test = torch.Tensor(np.load(os.path.join(c.data_path, 'coeff_test.npy')))

tr_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(x_train, y_train),
    batch_size=c.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)

tst_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(x_test, y_test),
    batch_size=c.batch_size, shuffle=False, drop_last=True)

