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
    torch.save(var_mean, os.path.join("mean_std", tensor_name+'_mean.pt'))
    torch.save(var_std, os.path.join("mean_std", tensor_name+'_std.pt'))
    return var_train, var_val   

### train
bandparams_train = torch.Tensor(np.load(os.path.join(c.data_path, 'x_train.npy')))
train_e = torch.Tensor(np.load(os.path.join(c.data_path, 'y_sigma_train.npy')))
train_s = torch.Tensor(np.load(os.path.join(c.data_path, 'y_seebeck_train.npy')))
train_n = torch.Tensor(np.load(os.path.join(c.data_path, 'y_n_train.npy')))

### val
bandparams_val = torch.Tensor(np.load(os.path.join(c.data_path, 'x_eval.npy')))
val_e = torch.Tensor(np.load(os.path.join(c.data_path, 'y_sigma_eval.npy')))
val_s = torch.Tensor(np.load(os.path.join(c.data_path, 'y_seebeck_eval.npy')))
val_n = torch.Tensor(np.load(os.path.join(c.data_path, 'y_n_eval.npy')))

bandparams_train_norm, bandparams_val_norm = standardize(bandparams_train, bandparams_val, "x")
train_norm_e, val_norm_e = standardize(train_e, val_e, "y_sigma")
train_norm_s, val_norm_s = standardize(train_s, val_s, "y_seebeck")
train_norm_n, val_norm_n = standardize(train_n, val_n, "y_n")

train_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(bandparams_train_norm, train_norm_e, train_norm_s, train_norm_n),
    batch_size=c.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)

val_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(bandparams_val_norm, val_norm_e, val_norm_s, val_norm_n),
    batch_size=c.batch_size, shuffle=False, drop_last=True)

