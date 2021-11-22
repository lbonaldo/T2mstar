import os
import sys
import numpy as np
import matplotlib.pyplot as plt

import torch

import test_config as c
import model

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"                                                                                                                                                                      
os.environ["CUDA_VISIBLE_DEVICES"]="2"  

def noise_batch(ndim):
    return torch.randn(c.batch_size, ndim).to(c.device)


def inference(model_path):
    export_path = os.path.join(model_path,"data")
    if not os.path.isdir(export_path):
        os.mkdir(export_path)
    # PARAMETERS
    batch_size = 1
    workers = 4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.model.train()

    # MODEL INITIALIZATION
    model.load(os.path.join(model_path, 'inn.pt'))

    # DATASET IMPORT
    x_train_ = torch.Tensor(np.load(os.path.join(c.data_path, 'x_train.npy')))
    x_val_ = torch.Tensor(np.load(os.path.join(c.data_path, 'x_val.npy')))
    x = torch.cat((x_train_, x_val_), dim=0)
    x_mean = x.mean(dim=0, keepdim=True)
    x_std = x.std(dim=0, keepdim=True)

    y_train_ = torch.Tensor(np.load(os.path.join(c.data_path, 'y_train.npy')))
    y_val_ = torch.Tensor(np.load(os.path.join(c.data_path, 'y_val.npy')))
    y = torch.cat((y_train_, y_val_), dim=0)
    y_mean = y.mean(dim=0, keepdim=True)
    y_std = y.std(dim=0, keepdim=True)

    x_test = torch.Tensor(np.load(os.path.join(c.data_path,'x_test.npy')))
    y_test = torch.Tensor(np.load(os.path.join(c.data_path,'y_test.npy')))
    
    y_test_norm = (y_test - y_mean) / y_std

    tst_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_test, y_test_norm),
        batch_size=c.batch_size, shuffle=True, drop_last=True)
    
    # INFERENCE
    results = []
    with torch.set_grad_enabled(False):
        for x_true, y_test in tst_loader:

            y_test = torch.autograd.Variable(y_test).to(c.device)            
            cat_inputs = [c.add_z_noise * noise_batch(c.ndim_z)]
            if c.ndim_pad_zy:
                cat_inputs.append(c.add_pad_noise * noise_batch(c.ndim_pad_zy))
            cat_inputs.append(y_test + c.add_y_noise * noise_batch(c.ndim_y))

            x_pred, _ = model.model(torch.cat(cat_inputs, 1), rev=True)
            results.append((x_pred, x_true))
            break
    
    for i in range(10):
        true = results[0][1][i,:]
        pred = results[0][0][i, :c.ndim_x].detach().cpu()

        pred_norm = results[0][0][i, :c.ndim_x].detach().cpu()
        
        x_mean = torch.squeeze(x_mean)
        x_std = torch.squeeze(x_std)
        pred = pred_norm*x_std + x_mean

        print(pred)
        print(true)

        np.savetxt(export_path+"/coeff_{}.csv".format(i), np.stack((pred, true)), delimiter=",")
    return


if __name__ == "__main__":
    if len(sys.argv) == 1:
        exit("Export folder name")
    else:
        output = inference(sys.argv[1])