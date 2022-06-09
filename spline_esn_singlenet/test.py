import os
import sys
from time import time

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
    model.model.train()

    #model.load("/mnt/scratch/bonal1lCMICH/inverse/spline/results/Jan-14-2022/06-51-44/inn.pt")
    model.load(os.path.join(model_path, 'inn.pt'))

    # DATASET IMPORT
    x_test_e = torch.Tensor(np.load(os.path.join(c.data_path, 'x_sigma_test.npy')))
    x_sigma_mean = torch.load('x_sigma_mean.pt')
    x_sigma_std = torch.load('x_sigma_std.pt')
    x_test_norm_e = (x_test_e - x_sigma_mean) / x_sigma_std

    x_test_s = torch.Tensor(np.load(os.path.join(c.data_path, 'x_seebeck_test.npy')))
    x_seebeck_mean = torch.load('x_seebeck_mean.pt')
    x_seebeck_std = torch.load('x_seebeck_std.pt')
    x_test_norm_s = (x_test_s - x_seebeck_mean) / x_seebeck_std

    x_test_n = torch.Tensor(np.load(os.path.join(c.data_path, 'x_n_test.npy')))
    x_n_mean = torch.load('x_n_mean.pt')
    x_n_std = torch.load('x_n_std.pt')
    x_test_norm_n = (x_test_n - x_n_mean) / x_n_std

    y_test = torch.Tensor(np.load(os.path.join(c.data_path, 'y_test.npy')))
    y_mean = torch.load('y_mean.pt')
    y_std = torch.load('y_std.pt')
    y_test_norm = (y_test - y_mean) / y_std

    test_size = x_test_e.shape[0]
    print("Test dataset size: ", test_size)

    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_test_norm_e, x_test_norm_s, x_test_norm_n, y_test_norm),
        batch_size=c.batch_size, shuffle=False, drop_last=True)
    
    # INFERENCE
    batch_idx = 0
    batch_loss = []
    batch_num = int(np.floor(test_size / c.batch_size)) # drop last, see TensorDataset
    final_coeff_norm = torch.empty((batch_num*c.batch_size, c.ndim_x))
    with torch.set_grad_enabled(False):
        for (x_e, x_s, x_n, y) in test_loader:

            x_e, x_s, x_n, y = x_e.to(c.device), x_s.to(c.device), x_n.to(c.device), y.to(c.device)

            x = torch.cat((x_e, x_s, x_n), dim=1)

            if c.ndim_pad_x:
                x = torch.cat((x, c.add_pad_noise * noise_batch(c.ndim_pad_x)), dim=1)
            if c.add_y_noise > 0:
                y += c.add_y_noise * noise_batch(c.ndim_y)
            if c.ndim_pad_zy:
                y = torch.cat((c.add_pad_noise * noise_batch(c.ndim_pad_zy), y), dim=1)
            y = torch.cat((noise_batch(c.ndim_z), y), dim=1)

            # forward step
            pred_x_batch, _ = model.model(y, rev=True)
            batch_loss.append(torch.nn.functional.mse_loss(pred_x_batch[:, :c.ndim_x], x[:, :c.ndim_x]).detach().cpu().numpy())
            final_coeff_norm[batch_idx*c.batch_size:(batch_idx+1)*c.batch_size, :] = pred_x_batch[:, :c.ndim_x]
            batch_idx += 1
    

    x_pred_e = (final_coeff_norm[:,:6].detach().cpu()*x_sigma_std + x_sigma_mean).numpy()
    x_pred_s = (final_coeff_norm[:,6:12].detach().cpu()*x_seebeck_std + x_seebeck_mean).numpy()
    x_pred_n = (final_coeff_norm[:,12:].detach().cpu()*x_n_std + x_n_mean).numpy()
    x_pred = np.column_stack([x_pred_e, x_pred_s, x_pred_n])
    np.savetxt(os.path.join(export_path, "x_pred.txt"), x_pred, delimiter=',')
    x_true = np.column_stack([x_test_e[:x_pred.shape[0],:], x_test_s[:x_pred.shape[0],:], x_test_n[:x_pred.shape[0],:]]) 
    comb = np.column_stack([x_true, x_pred])
    np.savetxt(os.path.join(export_path, "x_comb.txt"), comb, delimiter=',')
    err = np.abs(x_true-x_pred)
    np.savetxt(os.path.join(export_path, "x_rel_err.txt"), err, delimiter=',')

    return np.mean(batch_loss)


if __name__ == '__main__':
    torch.multiprocessing.freeze_support()

    if len(sys.argv) < 2:
        exit("Missing model path. Exit.") 

    if c.device == "cuda":
        print(torch.cuda.get_device_name(0))        
        c.log(torch.cuda.get_device_name(0), c.logfile)

    try:
        t_start = time()
        test_losses  = inference(sys.argv[1])
        print("Test results: ", test_losses)
        c.log("Test results: %f" % test_losses, c.logfile)

    finally:
        print("\n\nTesting took %f minutes\n\n" % ((time()-t_start)/60.))        
        c.log("\n\nTesting took %f minutes\n\n" % ((time()-t_start)/60.), c.logfile)

# source ~/.bashrc
# conda activate inverse
# module load CUDA/10.1.105
# python test.py /mnt/scratch/bonal1lCMICH/inverse/spline_esn_singlenet/results/Feb-17-2022/09-49-49