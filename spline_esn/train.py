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

    l_forw_fit = c.lambd_fit_forw * losses.l2_fit(out[:, -c.ndim_y:], y[:, -c.ndim_y:])

    return l_forw_fit, l_forw_mmd


def loss_backward_mmd(x, y, model):
    x_samples, jac = model(y, rev=True)
    MMD = losses.backward_mmd(x, x_samples)
    if c.mmd_back_weighted:
        MMD *= torch.exp(- 0.5 / c.y_uncertainty_sigma**2 * losses.l2_dist_matrix(y, y))
    return c.lambd_mmd_back * torch.mean(MMD)


def loss_reconstruction(out_y, x, model):
    # z_var
    cat_inputs = [out_y[:, :c.ndim_z] + c.add_z_noise * noise_batch(c.ndim_z)]
    # zy_pad_var
    if c.ndim_pad_zy:
        cat_inputs.append(out_y[:, c.ndim_z:-c.ndim_y] + c.add_pad_noise * noise_batch(c.ndim_pad_zy))
    # y_var
    cat_inputs.append(out_y[:, -c.ndim_y:] + c.add_y_noise * noise_batch(c.ndim_y))

    x_reconstructed, _ = model(torch.cat(cat_inputs, 1), rev=True)
    return c.lambd_reconstruct * losses.l2_fit(x_reconstructed, x)


def train_epoch(i_epoch, test=False):

    if not test:
        model.model_e.train()
        model.model_s.train()
        model.model_n.train()

        batch_idx = 0
        loss_history = []

        for (x_e, x_s, x_n, y) in c.train_loader:
            if batch_idx > c.n_its_per_epoch:
                break

            batch_losses = []
            batch_idx += 1

            x_e, x_s, x_n, y = Variable(x_e).to(c.device), Variable(x_s).to(c.device), Variable(x_n).to(c.device), Variable(y).to(c.device) 

            if c.ndim_pad_x:
                x_e = torch.cat((x_e, c.add_pad_noise * noise_batch(c.ndim_pad_x)), dim=1)
                x_s = torch.cat((x_s, c.add_pad_noise * noise_batch(c.ndim_pad_x)), dim=1)
                x_n = torch.cat((x_n, c.add_pad_noise * noise_batch(c.ndim_pad_x)), dim=1)
            if c.add_y_noise > 0:
                y += c.add_y_noise * noise_batch(c.ndim_y)
            if c.ndim_pad_zy:
                y = torch.cat((c.add_pad_noise * noise_batch(c.ndim_pad_zy), y), dim=1)
            y = torch.cat((noise_batch(c.ndim_z), y), dim=1)

            # forward step
            out_y_e, jac_e = model.model_e(x_e)
            out_y_s, jac_s = model.model_s(x_s)
            out_y_n, jac_n = model.model_n(x_n)
            out_y = (out_y_e+out_y_s+out_y_n)/3

            l_forw_e = 0.0
            l_forw_s = 0.0
            l_forw_n = 0.0
            if c.train_max_likelihood:
                lml_e = loss_max_likelihood(out_y, jac_e, y)
                lml_s = loss_max_likelihood(out_y, jac_s, y)
                lml_n = loss_max_likelihood(out_y, jac_n, y)
                batch_losses.extend([lml_e,lml_s,lml_n])
                l_forw_e += lml_e
                l_forw_s += lml_s
                l_forw_n += lml_n

            if c.train_forward_mmd:
                l_mmd_f = loss_forward_mmd(out_y, y)
                batch_losses.extend(l_mmd_f)
                l_forw_e += sum(l_mmd_f)
                l_forw_s += sum(l_mmd_f)
                l_forw_n += sum(l_mmd_f)

            l_forw_e.backward(retain_graph=True)
            l_forw_s.backward(retain_graph=True)
            l_forw_n.backward(retain_graph=True)

            l_back_e = 0.0
            l_back_s = 0.0
            l_back_n = 0.0
            if c.train_backward_mmd:
                l_mmd_b_e = loss_backward_mmd(x_e, y, model.model_e)
                l_mmd_b_s = loss_backward_mmd(x_s, y, model.model_s)
                l_mmd_b_n = loss_backward_mmd(x_n, y, model.model_n)
                batch_losses.extend([l_mmd_b_e,l_mmd_b_s,l_mmd_b_n])
                l_back_e += l_mmd_b_e
                l_back_s += l_mmd_b_s
                l_back_n += l_mmd_b_n

            if c.train_reconstruction:
                l_rec_e = loss_reconstruction(out_y.data, x_e, model.model_e)
                l_rec_s = loss_reconstruction(out_y.data, x_s, model.model_s)
                l_rec_n = loss_reconstruction(out_y.data, x_n, model.model_n)
                batch_losses.extend([l_rec_e,l_rec_s,l_rec_n])
                l_back_e += l_rec_e
                l_back_s += l_rec_s
                l_back_n += l_rec_n

            loss_history.append([l.item() for l in batch_losses])

            l_back_e.backward()
            l_back_s.backward()
            l_back_n.backward()
            model.optim_step()

        return np.mean(loss_history, axis=0)

    else:
        model.model_e.eval()
        model.model_s.eval()
        model.model_n.eval()

        with torch.no_grad():

            batch_idx = 0
            loss_history = []

            for (x_e, x_s, x_n, y) in c.val_loader:
                if batch_idx > c.n_its_per_epoch:
                    break

                batch_losses = []
                batch_idx += 1

                x_e, x_s, x_n, y = Variable(x_e).to(c.device), Variable(x_s).to(c.device), Variable(x_n).to(c.device), Variable(y).to(c.device) 

                if c.ndim_pad_x:
                    x_e = torch.cat((x_e, c.add_pad_noise * noise_batch(c.ndim_pad_x)), dim=1)
                    x_s = torch.cat((x_s, c.add_pad_noise * noise_batch(c.ndim_pad_x)), dim=1)
                    x_n = torch.cat((x_n, c.add_pad_noise * noise_batch(c.ndim_pad_x)), dim=1)
                if c.add_y_noise > 0:
                    y += c.add_y_noise * noise_batch(c.ndim_y)
                if c.ndim_pad_zy:
                    y = torch.cat((c.add_pad_noise * noise_batch(c.ndim_pad_zy), y), dim=1)
                y = torch.cat((noise_batch(c.ndim_z), y), dim=1)

                # forward step
                out_y_e, jac_e = model.model_e(x_e)
                out_y_s, jac_s = model.model_s(x_s)
                out_y_n, jac_n = model.model_n(x_n)
                out_y = (out_y_e+out_y_s+out_y_n)/3

                if c.train_max_likelihood:
                    lml_e = loss_max_likelihood(out_y, jac_e, y)
                    lml_s = loss_max_likelihood(out_y, jac_s, y)
                    lml_n = loss_max_likelihood(out_y, jac_n, y)
                    batch_losses.extend([lml_e,lml_s,lml_n])

                if c.train_forward_mmd:
                    l_mmd_f = loss_forward_mmd(out_y, y)
                    batch_losses.extend(l_mmd_f)

                if c.train_backward_mmd:
                    l_mmd_b_e = loss_backward_mmd(x_e, y)
                    l_mmd_b_s = loss_backward_mmd(x_s, y)
                    l_mmd_b_n = loss_backward_mmd(x_n, y)
                    batch_losses.extend([l_mmd_b_e,l_mmd_b_s,l_mmd_b_n])

                if c.train_reconstruction:
                    l_rec_e = loss_reconstruction(out_y.data, x_e)
                    l_rec_s = loss_reconstruction(out_y.data, x_s)
                    l_rec_n = loss_reconstruction(out_y.data, x_n)
                    batch_losses.extend([l_rec_e,l_rec_s,l_rec_n])

                loss_history.append([l.item() for l in batch_losses])

            #monitoring.show_hist(out_y[:, :c.ndim_z])
            #monitoring.show_cov(out_y[:, :c.ndim_z])

            # if c.test_time_functions:
            #     out_x, jac = model.model(y, rev=True)
            #     for f in c.test_time_functions:
            #         f(out_x, out_y, x, y)

            return np.mean(loss_history, axis=0)
