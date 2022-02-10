import os
import numpy as np

import torch
import torch.utils.data

import config as c

def standardize(train_tensor, val_tensor, tensor_name):
    train_size = train_tensor.shape[0]
    val_size = val_tensor.shape[0]
    var = torch.cat((train_tensor, val_tensor), dim=0)
    var_mean = var.mean(dim=0, keepdim=True)
    var_std = var.std(dim=0, keepdim=True)
    var_norm = (var - var_mean) / var_std
    var_train = var_norm[:train_size,:]
    var_val = var_norm[train_size:train_size+val_size,:]
    torch.save(var_mean, tensor_name+'_mean.pt')
    torch.save(var_std, tensor_name+'_std.pt')
    return var_train, var_val   

### train
x_train_e = torch.Tensor(np.load(os.path.join(c.data_path, 'x_sigma_train.npy')))
x_train_s = torch.Tensor(np.load(os.path.join(c.data_path, 'x_seebeck_train.npy')))
x_train_n = torch.Tensor(np.load(os.path.join(c.data_path, 'x_n_train.npy')))
y_train = torch.Tensor(np.load(os.path.join(c.data_path, 'y_train.npy')))

### val
x_val_e = torch.Tensor(np.load(os.path.join(c.data_path, 'x_sigma_val.npy')))
x_val_s = torch.Tensor(np.load(os.path.join(c.data_path, 'x_seebeck_val.npy')))
x_val_n = torch.Tensor(np.load(os.path.join(c.data_path, 'x_n_val.npy')))
y_val = torch.Tensor(np.load(os.path.join(c.data_path, 'y_val.npy')))

x_train_norm_e, x_val_norm_e = standardize(x_train_e, x_val_e, "x_sigma")
x_train_norm_s, x_val_norm_s = standardize(x_train_s, x_val_s, "x_seebeck")
x_train_norm_n, x_val_norm_n = standardize(x_train_n, x_val_n, "x_n")
y_train_norm, y_val_norm = standardize(y_train, y_val, "y")

train_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(x_train_norm_e, x_train_norm_s, x_train_norm_n, y_train_norm),
    batch_size=c.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)

val_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(x_val_norm_e, x_val_norm_s, x_val_norm_n, y_val_norm),
    batch_size=c.batch_size, shuffle=False, drop_last=True)
