import torch
import config as c

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MSE = torch.nn.MSELoss

def cstMSE(output, target):
    wi = torch.Tensor(c.loss_weights)
    wi.to(device)
    # assert(sum(wi)==1)
    loss = torch.zeros(tuple(output.size()), device=device)
    for i in range(output.size(dim=1)):
        loss[:,i] += torch.mul(((output[:,i]-target[:,i])**2),wi[i])
    return torch.mean(loss)
