import os
import numpy as np

import torch
import torch.utils.data

import config as c


x_train_ = torch.Tensor(np.load(os.path.join(c.data_path, 'x_train.npy')))
train_size = x_train_.shape[0]
y_train_ = torch.Tensor(np.load(os.path.join(c.data_path, 'y_train.npy')))

x_eval_ = torch.Tensor(np.load(os.path.join(c.data_path, 'x_eval.npy')))
eval_size = x_eval_.shape[0]
y_eval_ = torch.Tensor(np.load(os.path.join(c.data_path, 'y_eval.npy')))

print("Training dataset size: ", train_size)
print("Validation dataset size: ", eval_size)

x = torch.cat((x_train_, x_eval_), dim=0)
x_mean = x.mean(dim=0, keepdim=True)
x_std = x.std(dim=0, keepdim=True)
x_norm = (x - x_mean) / x_std
x_train = x_norm[:train_size,:]
x_eval = x_norm[train_size:train_size+eval_size,:]
torch.save(x_mean, 'x_mean.pt')
torch.save(x_std, 'x_std.pt')

y = torch.cat((y_train_, y_eval_), dim=0)
y_mean = y.mean(dim=0, keepdim=True)
y_std = y.std(dim=0, keepdim=True)
y_norm = (y - y_mean) / y_std
y_train = y_norm[:train_size,:]
y_eval = y_norm[train_size:train_size+eval_size,:]
torch.save(y_mean, 'y_mean.pt')
torch.save(y_std, 'y_std.pt')

train_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(x_train, y_train),
    batch_size=c.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)

eval_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(x_eval, y_eval),
    batch_size=c.batch_size, shuffle=False, drop_last=True)

