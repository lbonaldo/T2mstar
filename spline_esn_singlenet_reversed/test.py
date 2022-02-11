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
    model.model.eval()

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
    final_coeff_norm = torch.empty((batch_num*c.batch_size, y_test.shape[1]))
    with torch.set_grad_enabled(False):
        for (x_e, x_s, x_n, y) in test_loader:

            x_e, x_s, x_n, y = Variable(x_e).to(c.device), Variable(x_s).to(c.device), Variable(x_n).to(c.device), Variable(y).to(c.device) 

            x = torch.cat((x_e, x_s, x_n), dim=1)

            if c.ndim_pad_x:
                x = torch.cat((x, c.add_pad_noise * noise_batch(c.ndim_pad_x)), dim=1)
            # if c.add_y_noise > 0:
            #     y += c.add_y_noise * noise_batch(c.ndim_y)
            # if c.ndim_pad_zy:
            #     y = torch.cat((c.add_pad_noise * noise_batch(c.ndim_pad_zy), y), dim=1)
            # y = torch.cat((noise_batch(c.ndim_z), y), dim=1)

            # forward step
            pred_y_batch, _ = model.model(x)
            batch_loss.append(torch.nn.functional.mse_loss(pred_y_batch[:, -c.ndim_y:], y[:, -c.ndim_y:]).detach().cpu().numpy())
            final_coeff_norm[batch_idx*c.batch_size:(batch_idx+1)*c.batch_size, :] = pred_y_batch[:, -c.ndim_y:]
            batch_idx += 1
    
    y_pred = (final_coeff_norm.detach().cpu()*y_std + y_mean).numpy()
    np.savetxt(os.path.join(export_path, "y_pred.txt"), y_pred, delimiter=',')
    y_true = y_test[:y_pred.shape[0],:]
    res = np.column_stack([y_true, y_pred, y_true-y_pred])
    np.savetxt(os.path.join(export_path, "y_results.txt"), res, delimiter=',')

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
