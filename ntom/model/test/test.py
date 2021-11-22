import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import model


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
    end
end


def g(x, beta, mu):
    return np.sqrt(beta/np.pi)*exp(-beta*(x - mu)^2)


def inference(data_path, model_path):
    # PARAMETERS
    batch_size = 1
    workers = 4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.model.train()

    # MODEL INITIALIZATION
    model.load_state_dict(torch.load(os.path.join(model_path, 'inn_rev.pt'), map_location=device))

    # DATASET IMPORT

    I_test = torch.Tensor(np.load(os.path.join(data_path,'/I_test.npy')))
    coeff_test = torch.Tensor(np.load(os.path.join(data_path,'coeff_test.npy')))
    
    idxs = tuple(np.random.randint(0,len(I_test),size=10))
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
            coeff_pred, _ = model.model(I, rev=True)
            results.append(coeff_pred, coeff_true)

    xarr = np.arange(-5,5,0.05)
    l_true = np.empty(len(x))
    r_true = np.empty(len(x))
    g_true = np.empty(len(x))
    l_pred = np.empty(len(x))
    r_pred = np.empty(len(x))
    g_pred = np.empty(len(x))
    
    for i in len(results):
        a_pred,b_pred,c_pred,d_pred,mu_pred,beta_pred = results[i][0]
        a_true,b_true,c_true,d_true,mu_true,beta_true = results[i][1]
        for x in xarr:
            l_true[j] = l(x, a_true, b_true)
            r_true[j] = r(x, c_true, d_true)
            g_true[j] = g(x, beta_true, mu_true)
            l_pred[j] = l(x, a_pred, b_pred)
            r_pred[j] = r(x, c_pred, d_pred)
            g_pred[j] = g(x, beta_pred, mu_pred)

        fig = plt.figure(figsize=(10,6))
        plt.plot(x,l_true)
        plt.plot(x,r_true)
        plt.plot(x,g_true)
        plt.plot(x,l_pred)
        plt.plot(x,r_pred)
        plt.plot(x,g_pred)
        plt.savefig("plot_pred_{}".format(i))

    return

if __name__ == "__main__":
    data_path = "../../data"
    model_path = "."
    output = inference(data_path, model_path)