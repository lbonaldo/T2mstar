import numpy as np
import torch
from torch.autograd import Variable

import config as c

import losses
import model
import monitoring

assert c.train_loader and c.val_loader, "No data loaders supplied"


def noise_batch(ndim):
    return torch.randn(c.batch_size, ndim).to(c.device)


def loss_max_likelihood(out, jac, y):
    neg_log_likeli = (0.5/c.y_uncertainty_sigma**2 * torch.sum((out[:, -c.ndim_y:] - y[:, -c.ndim_y:])**2, 1)
                     + 0.5/c.zeros_noise_scale**2 * torch.sum((out[:, c.ndim_z:-c.ndim_y] - y[:, c.ndim_z:-c.ndim_y])**2, 1)
                     + 0.5 * torch.sum(out[:, :c.ndim_z]**2, 1)
                     - jac)

    return c.lambd_max_likelihood * torch.mean(neg_log_likeli)


def loss_forward_mmd(out, y):
    # Shorten output, and remove gradients wrt y, for latent loss
    output_block_grad = torch.cat((out[:, :c.ndim_z],
                                   out[:, -c.ndim_y:].data), dim=1)
    y_short = torch.cat((y[:, :c.ndim_z], y[:, -c.ndim_y:]), dim=1)
    l_forw_mmd = c.lambd_mmd_forw  * torch.mean(losses.forward_mmd(output_block_grad, y_short))

    l_forw_fit = c.lambd_fit_forw * losses.l2_fit(out[:, -c.ndim_y:], y[:, -c.ndim_y:])

    return l_forw_fit, l_forw_mmd


def loss_backward_mmd(x, x_sample, y):
    MMD = losses.backward_mmd(x, x_sample)
    if c.mmd_back_weighted:
        MMD *= torch.exp(- 0.5 / c.y_uncertainty_sigma**2 * losses.l2_dist_matrix(y, y))
    return c.lambd_mmd_back * torch.mean(MMD)


def get_reconstruction(out_y, model):
    # z_var
    cat_inputs = [out_y[:, :c.ndim_z] + c.add_z_noise * noise_batch(c.ndim_z)]
    # zy_pad_var
    if c.ndim_pad_zy:
        cat_inputs.append(out_y[:, c.ndim_z:-c.ndim_y] + c.add_pad_noise * noise_batch(c.ndim_pad_zy))
    # y_var
    cat_inputs.append(out_y[:, -c.ndim_y:] + c.add_y_noise * noise_batch(c.ndim_y))

    x_reconstructed, _ = model(torch.cat(cat_inputs, 1), rev=True)
    return x_reconstructed


def loss_reconstruction(x, x_reconstructed):
    return c.lambd_reconstruct * losses.l2_fit(x, x_reconstructed)


def train_epoch(eval=False):

    if not eval:
        model.model_e.train()
        model.model_s.train()
        model.model_n.train()

        batch_idx = 0
        loss_history = []

        for (x, y_e, y_s, y_n) in c.train_loader:
            if batch_idx > c.n_its_per_epoch:
                break

            batch_losses = []
            batch_idx += 1

            x, y_e, y_s, y_n = x.to(c.device), y_e.to(c.device), y_s.to(c.device), y_n.to(c.device)

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
            
            # forward step
            out_y_e, jac_e = model.model_e(x)
            out_y_s, jac_s = model.model_s(x)
            out_y_n, jac_n = model.model_n(x)

            l_forw_e = 0.0
            l_forw_s = 0.0
            l_forw_n = 0.0

            if c.train_max_likelihood:
                lml_e = loss_max_likelihood(out_y_e, jac_e, y_e)
                lml_s = loss_max_likelihood(out_y_s, jac_s, y_s)
                lml_n = loss_max_likelihood(out_y_n, jac_n, y_n)
                batch_losses.extend([lml_e,lml_s,lml_n])
                l_forw_e += lml_e
                l_forw_s += lml_s
                l_forw_n += lml_n

            if c.train_forward_mmd:
                l_mmd_f_e = loss_forward_mmd(out_y_e, y_e)
                l_mmd_f_s = loss_forward_mmd(out_y_s, y_s)
                l_mmd_f_n = loss_forward_mmd(out_y_n, y_n)
                batch_losses.extend(l_mmd_f_e)
                batch_losses.extend(l_mmd_f_s)
                batch_losses.extend(l_mmd_f_n)
                l_forw_e += sum(l_mmd_f_e)
                l_forw_s += sum(l_mmd_f_s)
                l_forw_n += sum(l_mmd_f_n)

            l_forw_e.backward(retain_graph=True)
            l_forw_s.backward(retain_graph=True)
            l_forw_n.backward(retain_graph=True)

            x, y_e, y_s, y_n = x.detach().clone(), y_e.detach().clone(), y_s.detach().clone(), y_n.detach().clone()
            out_y_e, out_y_s, out_y_n = out_y_e.clone(), out_y_s.clone(), out_y_n.clone()

            l_back_e = 0.0
            l_back_s = 0.0
            l_back_n = 0.0
            x_sample_e,_ = model.model_e(y_e, rev=True)
            x_sample_s,_ = model.model_s(y_s, rev=True)
            x_sample_n,_ = model.model_n(y_n, rev=True)
            x_sample = (x_sample_e+x_sample_s+x_sample_n)/3

            if c.train_backward_mmd:
                l_mmd_b_e = loss_backward_mmd(x, x_sample, out_y_e)
                l_mmd_b_s = loss_backward_mmd(x, x_sample, out_y_s)
                l_mmd_b_n = loss_backward_mmd(x, x_sample, out_y_n)
                batch_losses.extend([l_mmd_b_e,l_mmd_b_s,l_mmd_b_n])
                l_back_e += l_mmd_b_e
                l_back_s += l_mmd_b_s
                l_back_n += l_mmd_b_n

            if c.train_reconstruction:
                x_rec_e = get_reconstruction(out_y_e.data, model.model_e)
                x_rec_s = get_reconstruction(out_y_s.data, model.model_s)
                x_rec_n = get_reconstruction(out_y_n.data, model.model_n)
                x_rec = (x_rec_e+x_rec_s+x_rec_n)/3
                l_rec = loss_reconstruction(x,x_rec)
                batch_losses.append(l_rec)
                l_back_e += l_rec
                l_back_s += l_rec
                l_back_n += l_rec

            loss_history.append([l.item() for l in batch_losses])

            l_back_e.backward(retain_graph=True)
            l_back_s.backward(retain_graph=True)
            l_back_n.backward()

            model.optim_step()

        return np.mean(loss_history, axis=0)

    else:
        model.model_e.train()
        model.model_s.train()
        model.model_n.train()

        batch_idx = 0
        loss_history = []

        for (x, y_e, y_s, y_n) in c.val_loader:
            if batch_idx > c.n_its_per_epoch:
                break

            batch_losses = []
            batch_idx += 1

            x, y_e, y_s, y_n = x.to(c.device), y_e.to(c.device), y_s.to(c.device), y_n.to(c.device)

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

            # forward step
            out_y_e, jac_e = model.model_e(x)
            out_y_s, jac_s = model.model_s(x)
            out_y_n, jac_n = model.model_n(x)

            if c.train_max_likelihood:
                lml_e = loss_max_likelihood(out_y_e, jac_e, y_e)
                lml_s = loss_max_likelihood(out_y_s, jac_s, y_s)
                lml_n = loss_max_likelihood(out_y_n, jac_n, y_n)
                batch_losses.extend([lml_e,lml_s,lml_n])    

            if c.train_forward_mmd:
                l_mmd_f_e = loss_forward_mmd(out_y_e, y_e)
                l_mmd_f_s = loss_forward_mmd(out_y_s, y_s)
                l_mmd_f_n = loss_forward_mmd(out_y_n, y_n)
                batch_losses.extend(l_mmd_f_e)
                batch_losses.extend(l_mmd_f_s)
                batch_losses.extend(l_mmd_f_n)
                
            x_sample_e,_ = model.model_e(y_e, rev=True)
            x_sample_s,_ = model.model_s(y_s, rev=True)
            x_sample_n,_ = model.model_n(y_n, rev=True)
            x_sample = (x_sample_e+x_sample_s+x_sample_n)/3

            if c.train_backward_mmd:
                l_mmd_b_e = loss_backward_mmd(x, x_sample, out_y_e)
                l_mmd_b_s = loss_backward_mmd(x, x_sample, out_y_s)
                l_mmd_b_n = loss_backward_mmd(x, x_sample, out_y_n)
                batch_losses.extend([l_mmd_b_e,l_mmd_b_s,l_mmd_b_n])

            if c.train_reconstruction:
                x_rec_e = get_reconstruction(out_y_e.data, model.model_e)
                x_rec_s = get_reconstruction(out_y_s.data, model.model_s)
                x_rec_n = get_reconstruction(out_y_n.data, model.model_n)
                x_rec = (x_rec_e+x_rec_s+x_rec_n)/3
                l_rec = loss_reconstruction(x,x_rec)
                batch_losses.append(l_rec)

            loss_history.append([l.item() for l in batch_losses])

        return np.mean(loss_history, axis=0)
