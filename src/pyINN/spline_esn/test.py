import os
import sys
from time import time
from tkinter import X

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.autograd import Variable

import config as c
import model

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"                                                                            
os.environ["CUDA_VISIBLE_DEVICES"]="2"  

def noise_batch(ndim):
    return torch.randn(c.batch_size, ndim).to(c.device)


# test coeff reconstruction: coeff_pred, _ = model.model(torch.cat(cat_inputs, 1), rev=True)
def inference(model_path):
    export_path = os.path.join(c.test_path, "test")
    if not os.path.isdir(export_path):
        os.mkdir(export_path)

    # MODEL INITIALIZATION
    model.model_e.eval()
    model.model_s.eval()
    model.model_n.eval()
    #model.load("/mnt/scratch/bonal1lCMICH/inverse/spline/combults/Jan-14-2022/06-51-44/inn.pt")
    model.load(os.path.join(model_path, 'inn.pt'))

    # DATASET IMPORT
    x_test = torch.Tensor(np.load(os.path.join(c.data_path, 'x_test.npy')))
    x_mean = torch.load('x_mean.pt')
    x_std = torch.load('x_std.pt')
    x_test_norm = (x_test - x_mean) / x_std

    y_test_e = torch.Tensor(np.load(os.path.join(c.data_path, 'y_sigma_test.npy')))
    y_sigma_mean = torch.load('y_sigma_mean.pt')
    y_sigma_std = torch.load('y_sigma_std.pt')
    y_test_norm_e = (y_test_e - y_sigma_mean) / y_sigma_std

    y_test_s = torch.Tensor(np.load(os.path.join(c.data_path, 'y_seebeck_test.npy')))
    y_seebeck_mean = torch.load('y_seebeck_mean.pt')
    y_seebeck_std = torch.load('y_seebeck_std.pt')
    y_test_norm_s = (y_test_s - y_seebeck_mean) / y_seebeck_std

    y_test_n = torch.Tensor(np.load(os.path.join(c.data_path, 'y_n_test.npy')))
    y_n_mean = torch.load('y_n_mean.pt')
    y_n_std = torch.load('y_n_std.pt')
    y_test_norm_n = (y_test_n - y_n_mean) / y_n_std

    test_size = y_test_e.shape[0]
    print("Test dataset size: ", test_size)

    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_test_norm, y_test_norm_e, y_test_norm_s, y_test_norm_n),
        batch_size=c.batch_size, shuffle=False, drop_last=True)
    
    # INFERENCE
    batch_idx = 0
    batch_loss = []
    batch_num = int(np.floor(test_size / c.batch_size)) # drop last, see TensorDataset
    final_model_norm = torch.empty((batch_num*c.batch_size, c.ndim_x))
    with torch.set_grad_enabled(False):
        for (x, y_e, y_s, y_n) in test_loader:

            x, y_e, y_s, y_n = Variable(x).to(c.device), Variable(y_e).to(c.device), Variable(y_s).to(c.device), Variable(y_n).to(c.device)

            if c.ndim_pad_x:
                x = torch.cat((x, c.add_pad_noise * noise_batch(c.ndim_pad_x)), dim=1)
            if c.add_y_noise > 0:
                y_e += c.add_y_noise * noise_batch(c.ndim_y)
                y_s += c.add_y_noise * noise_batch(c.ndim_y)
                y_n += c.add_y_noise * noise_batch(c.ndim_y)
            if c.ndim_pad_zy:
                y_e = torch.cat((c.add_pad_noise * noise_batch(c.ndim_pad_zy), y_e), dim=1)
                y_s = torch.cat((c.add_pad_noise * noise_batch(c.ndim_pad_zy), y_s), dim=1)
                y_n = torch.cat((c.add_pad_noise * noise_batch(c.ndim_pad_zy), y_n), dim=1)
            y_e = torch.cat((noise_batch(c.ndim_z), y_e), dim=1)
            y_s = torch.cat((noise_batch(c.ndim_z), y_s), dim=1)
            y_n = torch.cat((noise_batch(c.ndim_z), y_n), dim=1)

            x_pred_e_batch,_ = model.model_e(y_e, rev=True)
            x_pred_s_batch,_ = model.model_s(y_s, rev=True)
            x_pred_n_batch,_ = model.model_n(y_n, rev=True)

            x_pred_batch = (x_pred_e_batch+x_pred_s_batch+x_pred_n_batch)/3

            loss_x = torch.nn.functional.mse_loss(x_pred_batch[:, :c.ndim_x], x[:, :c.ndim_x]).detach().cpu().numpy()
            batch_loss.append(loss_x)
            final_model_norm[batch_idx*c.batch_size:(batch_idx+1)*c.batch_size, :] = x_pred_batch[:,:c.ndim_x]
            batch_idx += 1
    
    x_pred = (final_model_norm.detach().cpu()*x_std + x_mean)
    x_true = x_test[:x_pred.shape[0],:] # dataloader skip last batch
    last_loss = torch.nn.functional.mse_loss(x_pred, x_true).numpy()
    np.savetxt(os.path.join(export_path, "x_pred.txt"), x_pred.numpy(), delimiter=',')
    comb = np.column_stack([x_true, x_pred])
    np.savetxt(os.path.join(export_path, "x_comb.txt"), comb, delimiter=',')
    err = np.abs(x_true-x_pred)
    np.savetxt(os.path.join(export_path, "x_abs_err.txt"), err, delimiter=',')

    return last_loss, np.mean(batch_loss)


if __name__ == '__main__':
    torch.multiprocessing.freeze_support()

    if len(sys.argv) < 2:
        exit("Missing model path. Exit.") 

    if c.device == "cuda":
        print(torch.cuda.get_device_name(0))        
        c.log(torch.cuda.get_device_name(0), c.logfile)

    try:
        t_start = time()
        last_loss, test_losses  = inference(sys.argv[1])
        print("Last loss: ", last_loss)
        c.log("Last loss: %f" % last_loss, c.logfile)
        print("Test results: ", test_losses)
        c.log("Test results: %f" % test_losses, c.logfile)

    finally:
        print("\n\nTesting took %f minutes\n\n" % ((time()-t_start)/60.))        
        c.log("\n\nTesting took %f minutes\n\n" % ((time()-t_start)/60.), c.logfile)
