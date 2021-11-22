import numpy as np
import torch
from torch.autograd import Variable

import config as c

import losses
import model
import monitoring

assert c.train_loader and c.test_loader, "No data loaders supplied"


def noise_batch(ndim):
    return torch.randn(c.batch_size, ndim).to(c.device)


def loss_max_likelihood(out, jac, y):
    neg_log_likeli = ( 0.5 / c.y_uncertainty_sigma**2 * torch.sum((out[:, -c.ndim_y:] - y[:, -c.ndim_y:])**2, 1)
                     + 0.5 / c.zeros_noise_scale**2   * torch.sum((out[:, c.ndim_z:-c.ndim_y] - y[:, c.ndim_z:-c.ndim_y])**2, 1)
                     + 0.5 * torch.sum(out[:, :c.ndim_z]**2, 1)
                     - jac)

    return c.lambd_max_likelihood * torch.mean(neg_log_likeli)


def loss_forward_mmd(out, y):
    # Shorten output, and remove gradients wrt y, for latent loss
    output_block_grad = torch.cat((out[:, :c.ndim_z],
                                   out[:, -c.ndim_y:].data), dim=1)
    y_short = torch.cat((y[:, :c.ndim_z], y[:, -c.ndim_y:]), dim=1)
    l_forw_mmd = c.lambd_mmd_forw  * torch.mean(losses.forward_mmd(output_block_grad, y_short))

    l_forw_fit = c.lambd_fit_forw * losses.l2_fit(out[:, c.ndim_z:], y[:, c.ndim_z:])

    return l_forw_fit, l_forw_mmd


def loss_backward_mmd(x, y):
    x_samples, jac = model.model(y, rev=True)
    MMD = losses.backward_mmd(x, x_samples)
    if c.mmd_back_weighted:
        MMD *= torch.exp(- 0.5 / c.y_uncertainty_sigma**2 * losses.l2_dist_matrix(y, y))
    return c.lambd_mmd_back * torch.mean(MMD)


def loss_reconstruction(out_y, y, x):
    cat_inputs = [out_y[:, :c.ndim_z] + c.add_z_noise * noise_batch(c.ndim_z)]
    if c.ndim_pad_zy:
        cat_inputs.append(out_y[:, c.ndim_z:-c.ndim_y] + c.add_pad_noise * noise_batch(c.ndim_pad_zy))
    cat_inputs.append(out_y[:, -c.ndim_y:] + c.add_y_noise * noise_batch(c.ndim_y))

    x_reconstructed, jac = model.model(torch.cat(cat_inputs, 1), rev=True)
    return c.lambd_reconstruct * losses.l2_fit(x_reconstructed, x)


def train_epoch(i_epoch, test=False):

    if not test:
        model.model.train()
        loader = c.train_loader

    if test:
        model.model.eval()
        loader = c.test_loader
        nograd = torch.no_grad()
        nograd.__enter__()

    batch_idx = 0
    loss_history = []

    for x, y in loader:
        if batch_idx > c.n_its_per_epoch:
            break

        batch_losses = []
        batch_idx += 1

        x, y = Variable(x).to(c.device), Variable(y).to(c.device)

        if c.ndim_pad_x:
            x = torch.cat((x, c.add_pad_noise * noise_batch(c.ndim_pad_x)), dim=1)
        if c.add_y_noise > 0:
            y += c.add_y_noise * noise_batch(c.ndim_y)
        if c.ndim_pad_zy:
            y = torch.cat((c.add_pad_noise * noise_batch(c.ndim_pad_zy), y), dim=1)
        y = torch.cat((noise_batch(c.ndim_z), y), dim=1)

        # forward step
        out_y, jac = model.model(x)

        l_forw = 0.0
        if c.train_max_likelihood:
            lml = loss_max_likelihood(out_y, jac, y)
            batch_losses.append(lml)
            l_forw += lml

        if c.train_forward_mmd:
            l_mmd_f = loss_forward_mmd(out_y, y)
            batch_losses.extend(l_mmd_f)
            l_forw += sum(l_mmd_f)

        if not test:
            l_forw.backward()

        l_back = 0.0
        if c.train_backward_mmd:
            l_mmd_b = loss_backward_mmd(x, y)
            batch_losses.append(l_mmd_b)
            l_back += l_mmd_b

        if c.train_reconstruction:
            l_rec = loss_reconstruction(out_y.data, y, x)
            batch_losses.append(l_rec)
            l_back += l_rec

        loss_history.append([l.item() for l in batch_losses])

        if not test:
            l_back.backward()
            model.optim_step()

    if test:
        monitoring.show_hist(out_y[:, :c.ndim_z])
        monitoring.show_cov(out_y[:, :c.ndim_z])

        if c.test_time_functions:
            out_x, jac = model.model(y, rev=True)
            for f in c.test_time_functions:
                f(out_x, out_y, x, y)

        nograd.__exit__(None, None, None)

    return np.mean(loss_history, axis=0)

