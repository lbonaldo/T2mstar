import numpy as np
import torch
import matplotlib.pyplot as plt

import config as c


def MMD_matrix_multiscale(x, y, widths_exponents):
    xx, yy, xy = torch.mm(x,x.t()), torch.mm(y,y.t()), torch.mm(x,y.t())

    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))

    dxx = torch.clamp(rx.t() + rx - 2.*xx, 0, np.inf)
    dyy = torch.clamp(ry.t() + ry - 2.*yy, 0, np.inf)
    dxy = torch.clamp(rx.t() + ry - 2.*xy, 0, np.inf)

    XX, YY, XY = (torch.zeros(xx.shape).to(c.device),
                  torch.zeros(xx.shape).to(c.device),
                  torch.zeros(xx.shape).to(c.device))

    for h,e in widths_exponents:
        XX += h**e * (h**e + dxx)**-1
        YY += h**e * (h**e + dyy)**-1
        XY += h**e * (h**e + dxy)**-1

#    debug_mmd_terms(XX.detach().cpu(), YY.detach().cpu(), XY.detach().cpu())

    return XX + YY - 2.*XY


def l2_dist_matrix(x, y):
    xx, yy, xy = torch.mm(x,x.t()), torch.mm(y,y.t()), torch.mm(x,y.t())

    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))

    return torch.clamp(rx.t() + ry - 2.*xy, 0, np.inf)

def forward_mmd(y0, y1):
    return MMD_matrix_multiscale(y0, y1, c.mmd_forw_kernels)


def backward_mmd(x0, x1):
    return MMD_matrix_multiscale(x0, x1, c.mmd_back_kernels)


def l2_fit(pred, true):
    return torch.nn.functional.mse_loss(pred,true)


def debug_mmd_terms(XX, YY, XY):

    plt.subplot(2,2,1)
    plt.imshow((XX + YY - XY - XY.t()).data.numpy(), cmap='jet')
    plt.title('Tot')
    plt.colorbar()

    plt.subplot(2,2,2)
    plt.imshow(XX.data.numpy(), cmap='jet')
    plt.title('XX')
    plt.colorbar()

    plt.subplot(2,2,3)
    plt.imshow(YY.data.numpy(), cmap='jet')
    plt.title('YY')
    plt.colorbar()

    plt.subplot(2,2,4)
    plt.imshow(XY.data.numpy(), cmap='jet')
    plt.title('XY')
    plt.colorbar()

    plt.savefig("./results/debug_mmd_terms1.png")
    plt.close()
    plt.cla()
    plt.clf()
