import numpy as np
import torch
import torch.utils.data

import config as c

x_train = torch.Tensor(np.load('../data/x_train.npy'))
y_train = torch.Tensor(np.load('../data/y_train.npy'))
y_train = y_train[:, None]

x_test = torch.Tensor(np.load('../data/x_test.npy'))
y_test = torch.Tensor(np.load('../data/y_test.npy'))
y_test = y_test[:, None]

tr_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(x_train, y_train),
    batch_size=c.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)

tst_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(x_test, y_test),
    batch_size=c.batch_size, shuffle=False, drop_last=True)

if __name__ == "__main__":
    import train
    train.main()
