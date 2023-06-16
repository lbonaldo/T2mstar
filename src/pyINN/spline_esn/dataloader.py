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
x_train = torch.Tensor(np.load(os.path.join(c.data_path, 'x_train.npy')))
y_train_e = torch.Tensor(np.load(os.path.join(c.data_path, 'y_sigma_train.npy')))
y_train_s = torch.Tensor(np.load(os.path.join(c.data_path, 'y_seebeck_train.npy')))
y_train_n = torch.Tensor(np.load(os.path.join(c.data_path, 'y_n_train.npy')))

### val
x_val = torch.Tensor(np.load(os.path.join(c.data_path, 'x_val.npy')))
y_val_e = torch.Tensor(np.load(os.path.join(c.data_path, 'y_sigma_val.npy')))
y_val_s = torch.Tensor(np.load(os.path.join(c.data_path, 'y_seebeck_val.npy')))
y_val_n = torch.Tensor(np.load(os.path.join(c.data_path, 'y_n_val.npy')))

x_train_norm, x_val_norm = standardize(x_train, x_val, "x")
y_train_norm_e, y_val_norm_e = standardize(y_train_e, y_val_e, "y_sigma")
y_train_norm_s, y_val_norm_s = standardize(y_train_s, y_val_s, "y_seebeck")
y_train_norm_n, y_val_norm_n = standardize(y_train_n, y_val_n, "y_n")

train_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(x_train_norm, y_train_norm_e, y_train_norm_s, y_train_norm_n),
    batch_size=c.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)

val_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(x_val_norm, y_val_norm_e, y_val_norm_s, y_val_norm_n),
    batch_size=c.batch_size, shuffle=False, drop_last=True)
