import os
import numpy as np
import matplotlib.pyplot as plt

import torch

import test_config as c
import model

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"                                                                                                                                                                      
os.environ["CUDA_VISIBLE_DEVICES"]="2"  

def noise_batch(ndim):
    return torch.randn(c.batch_size, ndim).to(c.device)

# Defines the Left function.
def l(x, a, b):
    if -a*x - b > 0:
        return np.sqrt(-a*x - b)
    else:
        return 0


# Defines the Right function.
def r(x, c, d):
    if c*x - d > 0:
        return np.sqrt(c*x - d)
    else:
        return 0


def g(x, beta, mu):
    return np.sqrt(beta/np.pi)*np.exp(-beta*(x - mu)**2)


def inference(data_path, model_path):
    # PARAMETERS
    batch_size = 1
    workers = 4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.model.train()

    # MODEL INITIALIZATION
    model.load(os.path.join(model_path, 'inn.pt'))

    # DATASET IMPORT
    I_test = torch.Tensor(np.load(os.path.join(data_path,'I_test.npy')))
    coeff_test = torch.Tensor(np.load(os.path.join(data_path,'coeff_test.npy')))
    
    idxs = tuple(np.random.randint(0,len(I_test),size=500))
    I_test = I_test[idxs, None]
    coeff_test = coeff_test[idxs, :]

    tst_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(I_test, coeff_test),
        batch_size=c.batch_size, shuffle=False, drop_last=True)
    
    # INFERENCE
    results = []
    with torch.set_grad_enabled(False):
        for I, coeff_true in tst_loader:

            I = torch.autograd.Variable(I).to(c.device)            
            cat_inputs = [c.add_z_noise * noise_batch(c.ndim_z)]
            if c.ndim_pad_zy:
                cat_inputs.append(c.add_pad_noise * noise_batch(c.ndim_pad_zy))
            cat_inputs.append(I + c.add_y_noise * noise_batch(c.ndim_y))

            coeff_pred, _ = model.model(torch.cat(cat_inputs, 1), rev=True)
            results.append((coeff_pred, coeff_true))
            break

    xarr = np.arange(-5,5,0.05)
    l_true = np.empty(len(xarr))
    r_true = np.empty(len(xarr))
    g_true = np.empty(len(xarr))
    l_pred = np.empty(len(xarr))
    r_pred = np.empty(len(xarr))
    g_pred = np.empty(len(xarr))

    for i in range(10):
        a_pred = results[0][0][i, :c.ndim_x].detach().cpu().numpy()[0]
        b_pred = results[0][0][i, :c.ndim_x].detach().cpu().numpy()[1]
        c_pred = results[0][0][i, :c.ndim_x].detach().cpu().numpy()[2]
        d_pred = results[0][0][i, :c.ndim_x].detach().cpu().numpy()[3]
        mu_pred = results[0][0][i, :c.ndim_x].detach().cpu().numpy()[4]
        beta_pred = results[0][0][i, :c.ndim_x].detach().cpu().numpy()[5]
        a_true = results[0][1][i,0]
        b_true = results[0][1][i,1]
        c_true = results[0][1][i,2]
        d_true = results[0][1][i,3]
        mu_true = results[0][1][i,4]
        beta_true = results[0][1][i,5]
        
        for j,x in enumerate(xarr):
            l_true[j] = l(x, a_true, b_true)
            r_true[j] = r(x, c_true, d_true)
            g_true[j] = g(x, beta_true, mu_true)
            l_pred[j] = l(x, a_pred, b_pred)
            r_pred[j] = r(x, c_pred, d_pred)
            g_pred[j] = g(x, beta_pred, mu_pred)

        fig = plt.figure(figsize=(10,6))
        plt.plot(xarr,l_true, color='#1f77b4', label="l_true")
        plt.plot(xarr,r_true, color='#1f77b4',  label="r_true")
        plt.plot(xarr,g_true, color='#1f77b4',  label="g_true")
        plt.plot(xarr,l_pred, color='#ff7f0e', label="l_pred")
        plt.plot(xarr,r_pred, color='#ff7f0e',  label="r_pred")
        plt.plot(xarr,g_pred, color='#ff7f0e',  label="g_pred")
        plt.legend()
        plt.savefig("plot_pred_{}".format(i))

    return


def inference_rev(data_path, model_path):
    # PARAMETERS
    batch_size = 1
    workers = 4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.model.train()

    # MODEL INITIALIZATION
    model.load(os.path.join(model_path, 'inn_rev.pt'))

    # DATASET IMPORT
    I_test = torch.Tensor(np.load(os.path.join(data_path,'I_test.npy')))
    coeff_test = torch.Tensor(np.load(os.path.join(data_path,'coeff_test.npy')))
    
    idxs = tuple(np.random.randint(0,len(I_test),size=500))
    I_test = I_test[idxs, None]
    coeff_test = coeff_test[idxs, :]

    tst_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(I_test, coeff_test),
        batch_size=c.batch_size, shuffle=False, drop_last=True)
    
    # INFERENCE
    results = []
    with torch.set_grad_enabled(False):
        for I, coeff_true in tst_loader:
            I = torch.autograd.Variable(I).to(c.device)
            if c.ndim_pad_x:
                I = torch.cat((I, c.add_pad_noise * noise_batch(c.ndim_pad_x)), dim=1)    
            coeff_pred, _ = model.model(I)
            results.append((coeff_pred, coeff_true))
            break

    xarr = np.arange(-5,5,0.05)
    l_true = np.empty(len(xarr))
    r_true = np.empty(len(xarr))
    g_true = np.empty(len(xarr))
    l_pred = np.empty(len(xarr))
    r_pred = np.empty(len(xarr))
    g_pred = np.empty(len(xarr))

    for i in range(10):
        
        a_pred = results[0][0][i, c.ndim_z:].detach().cpu().numpy()[0]
        b_pred = results[0][0][i, c.ndim_z:].detach().cpu().numpy()[1]
        c_pred = results[0][0][i, c.ndim_z:].detach().cpu().numpy()[2]
        d_pred = results[0][0][i, c.ndim_z:].detach().cpu().numpy()[3]
        mu_pred = results[0][0][i, c.ndim_z:].detach().cpu().numpy()[4]
        beta_pred = results[0][0][i, c.ndim_z:].detach().cpu().numpy()[5]
        a_true = results[0][1][i,0]
        b_true = results[0][1][i,1]
        c_true = results[0][1][i,2]
        d_true = results[0][1][i,3]
        mu_true = results[0][1][i,4]
        beta_true = results[0][1][i,5]
        
        for j,x in enumerate(xarr):
            l_true[j] = l(x, a_true, b_true)
            r_true[j] = r(x, c_true, d_true)
            g_true[j] = g(x, beta_true, mu_true)
            l_pred[j] = l(x, a_pred, b_pred)
            r_pred[j] = r(x, c_pred, d_pred)
            g_pred[j] = g(x, beta_pred, mu_pred)

        fig = plt.figure(figsize=(10,6))
        plt.plot(xarr,l_true, 'b', label="l_true")
        plt.plot(xarr,r_true, 'b',  label="r_true")
        plt.plot(xarr,g_true, 'b',  label="g_true")
        plt.plot(xarr,l_pred, 'r', label="l_pred")
        plt.plot(xarr,r_pred, 'r',  label="r_pred")
        plt.plot(xarr,g_pred, 'r',  label="g_pred")
        plt.legend()
        plt.savefig("plot_pred_{}".format(i))

    return

if __name__ == "__main__":
    data_path = "/mnt/scratch/bonal1lCMICH/data"
    model_path = "."
    output = inference(data_path, model_path)
