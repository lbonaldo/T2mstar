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


def inference(model_path, rev=True):
    export_path = os.path.join(model_path,"data")
    if not os.path.isdir(export_path):
        os.mkdir(export_path)
    # PARAMETERS
    model.model.eval()

    # MODEL INITIALIZATION
    model.load(os.path.join(model_path, 'inn.pt'))

    # DATASET IMPORT
    x_train_ = torch.Tensor(np.load(os.path.join(c.data_path, '{}_train.npy'.format('x' if rev is True else 'y'))))
    x_val_ = torch.Tensor(np.load(os.path.join(c.data_path, '{}_val.npy'.format('x' if rev is True else 'y'))))
    x = torch.cat((x_train_, x_val_), dim=0)
    x_mean = x.mean(dim=0, keepdim=True)
    x_std = x.std(dim=0, keepdim=True)

    y_train_ = torch.Tensor(np.load(os.path.join(c.data_path, '{}_train.npy'.format(('y' if rev is True else 'x')))))
    y_val_ = torch.Tensor(np.load(os.path.join(c.data_path, '{}_val.npy'.format(('y' if rev is True else 'x')))))
    y = torch.cat((y_train_, y_val_), dim=0)
    y_mean = y.mean(dim=0, keepdim=True)
    y_std = y.std(dim=0, keepdim=True)

    x_test = torch.Tensor(np.load(os.path.join(c.data_path,'{}_test.npy'.format('x' if rev is True else 'y'))))
    y_test = torch.Tensor(np.load(os.path.join(c.data_path,'{}_test.npy'.format('y' if rev is True else 'x'))))
    
    x_test_norm = (x_test - x_mean) / x_std
    y_test_norm = (y_test - y_mean) / y_std

    x_mean = torch.squeeze(x_mean)
    x_std = torch.squeeze(x_std)
    y_mean = torch.squeeze(y_mean)
    y_std = torch.squeeze(y_std)

    # INFERENCE
    results = []
    if rev == True:

        tst_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(x_test, y_test_norm),
            batch_size=c.batch_size, shuffle=False, drop_last=True)

        x_pred = np.empty(x_test.shape) 

        with torch.set_grad_enabled(False):
            batch_idx = 0
            for _, y_test in tst_loader:

                y_test = torch.autograd.Variable(y_test).to(c.device)            
                cat_inputs = [c.add_z_noise * noise_batch(c.ndim_z)]
                if c.ndim_pad_zy:
                    cat_inputs.append(c.add_pad_noise * noise_batch(c.ndim_pad_zy))
                cat_inputs.append(y_test + c.add_y_noise * noise_batch(c.ndim_y))

                x_pred_b, _ = model.model(torch.cat(cat_inputs, 1), rev=rev)
                pred_norm = ((x_pred_b[:, :c.ndim_x].detach().cpu())*x_std + x_mean).numpy()
                x_pred[batch_idx*c.batch_size:(batch_idx+1)*c.batch_size,:] = pred_norm
                batch_idx += 1
                
            np.save("pred.npy", x_pred)
            np.save("true.npy", x_test.numpy())
        # diff = 0.0

        # for i in range(10):
        #     pred_norm = results[0][0][i, :c.ndim_x].detach().cpu()
        #     pred = (pred_norm*x_std + x_mean).numpy()
        #     true_norm = results[0][1][i,:]
        #     true = (true_norm*x_std + x_mean).numpy()

        #     mae = np.mean(np.abs(pred-true))
        #     print("MAE: ", mae)
        #     diff += mae
        #     print(pred)
        #     print(true)

        #     np.savetxt(export_path+"/coeff_{}.csv".format(i), np.stack((pred, true)), delimiter=",")
        # print("total MAE: ", diff/10)
        return

    else:
        with torch.set_grad_enabled(False):
            for x_test, y_true in tst_loader:
                x_test = torch.autograd.Variable(x_test).to(c.device)          

                if c.ndim_pad_x:
                    x_test = torch.cat((x_test, c.add_pad_noise * noise_batch(c.ndim_pad_x)), dim=1)

                y_pred, _ = model.model(x_test)
                results.append((y_pred, y_true))
                break
                    
        diff = 0.0
        for i in range(10):
            pred_norm = results[0][0][i, -c.ndim_y:].detach().cpu()
            pred = (pred_norm*y_std + y_mean).numpy()
            true_norm = results[0][1][i,:]
            true = (true_norm*y_std + y_mean).numpy()
            
            mae = np.mean(np.abs(pred-true))
            print("MAE: ", mae)
            diff += mae
            print(pred)
            print(true)

            np.savetxt(export_path+"/coeff_{}.csv".format(i), np.stack((pred, true)), delimiter=",")
        print("total MAE: ", diff/10)
        return


if __name__ == "__main__":
    if len(sys.argv) < 3:
        exit("Check params.")
    else:
        rev = None
        if sys.argv[2] == "true":
            rev = True
        elif sys.argv[2] == "false":
            rev = False
        else:
            exit("Check second param")

        output = inference(sys.argv[1], rev)
