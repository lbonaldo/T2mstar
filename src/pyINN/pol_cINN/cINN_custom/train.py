import os
from time import time

import numpy as np
import torch

from losses import Losses

from utils import log,noise_batch


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


class Trainer(object):
    def __init__(self,paths,models,device="cpu"):
        torch.multiprocessing.freeze_support()

        self.models = models
        self.paths = paths
        self.device = device

        if device is "cuda":
            print(torch.cuda.get_device_name(0))        
            log(torch.cuda.get_device_name(0), self.paths.logfile)

    def exec(self):
        self.train_fcn()
        self.train_inn()

    def train_fcn(self):

        t_start = time()
        wrapper = self.models["S"]["fcn"]
        hyper = wrapper.hyparams
        train_loader,val_loader = wrapper.loaders.get_fcn_loaders()
        model = wrapper.model
        optim = wrapper.optim
        scheduler = wrapper.scheduler
        visualizer = wrapper.visualizer

        running_loss = np.Inf
        try:
            visualizer.print_config(self.paths.model_name["fcn"])
            train_losses = np.empty(hyper.epochs)
            val_losses = np.empty(hyper.epochs)

            # start the training
            for i in range(hyper.epochs):

                model.train()
                epoch_loss = []

                for (band, S) in train_loader:

                    band, S = band.to(self.device), S.to(self.device)
                    # separate mstars with Fermi level
                    mu = band[:,-1]

                    optim.zero_grad()
                    with torch.set_grad_enabled(True):    
                        # forward step
                        pred_mu = torch.squeeze(model(S))
                        l_mu = torch.nn.functional.mse_loss(mu,pred_mu)
                        # backward step
                        l_mu.backward()
                        optim.step()

                    epoch_loss.append(l_mu.detach().cpu().numpy())

                current_train_loss = np.mean(epoch_loss)
                train_losses[i] = current_train_loss

                # valdation
                model.eval()

                epoch_loss = []
                for (band, S) in val_loader:

                    band, S = band.to(self.device), S.to(self.device)
                    # separate mstars with Fermi level
                    mu = band[:,-1]

                    with torch.set_grad_enabled(False):    
                        pred_mu = torch.squeeze(model(S))
                        l_mu = torch.nn.functional.mse_loss(mu,pred_mu)

                    epoch_loss.append(l_mu.detach().cpu().numpy())

                current_val_loss = np.mean(epoch_loss)
                val_losses[i] = current_val_loss
                visualizer.print_single_loss(get_lr(optim), current_train_loss, current_val_loss)

                if  current_val_loss < running_loss:
                    print("Best loss: ", running_loss)
                    print("New loss: ", current_val_loss, " -->  model saved.")
                    torch.save({'opt': optim.state_dict(), 'net': model.state_dict()}, os.path.join(self.paths.train_path,self.paths.model_name["fcn"]+".pt"))
                    running_loss = current_val_loss

                scheduler.step(current_val_loss)

            visualizer.plot_single_loss(hyper.epochs, train_losses, val_losses, logscale=False)

            print("\n\nTraining fcn took %f minutes\n\n" % ((time()-t_start)/60.))        
            log("\n\nTraining fcn took %f minutes\n\n" % ((time()-t_start)/60.), self.paths.logfile)

        except:
            torch.save({'opt': optim.state_dict(), 'net': model.state_dict()}, os.path.join(self.paths.train_path,self.paths.model_name["fcn"]+".pt"+'_ABORT'))
            raise

    def train_inn(self):

        t_start = time()

        # cond inn
        e_wrapper = self.models["e"]
        e_model = e_wrapper.model
        e_optim = e_wrapper.optim
        e_scheduler = e_wrapper.scheduler

        # cc inn
        n_wrapper = self.models["n"]
        n_model = n_wrapper.model
        n_optim = n_wrapper.optim
        n_scheduler = n_wrapper.scheduler

        # S inn
        S_wrapper = self.models["S"]["inn"]
        S_model = S_wrapper.model
        S_optim = S_wrapper.optim
        S_scheduler = S_wrapper.scheduler

        # hyperparams and data are the same for both the nets
        hyper = e_wrapper.hyparams
        lossparams = e_wrapper.lossparams
        inn_params = e_wrapper.others
        train_loader,val_loader = e_wrapper.loaders.get_inn_loaders()
        visualizer = e_wrapper.visualizer

        # loss functions
        losses = Losses(lossparams,self.device,inn_params)

        running_loss = np.Inf
        try:
            visualizer.print_config(" ".join(self.paths.model_name["inn"]))

            ndim_y_e = inn_params.ndim_ys["inn"][0]
            ndim_y_n = inn_params.ndim_ys["inn"][1]
            ndim_y_S = inn_params.ndim_ys["inn"][2]

            train_losses = np.empty((hyper.epochs,3*inn_params.num_losses))
            val_losses = np.empty((hyper.epochs,3*inn_params.num_losses))

            # start the training
            for i in range(hyper.epochs):

                e_model.train()
                n_model.train()
                S_model.train()

                epoch_loss = []
                for (band, e, n, S) in train_loader:
            
                    band, e, n, S = band.to(self.device), e.to(self.device), n.to(self.device), S.to(self.device)

                    # separate mstars with Fermi level
                    mstars = band[:,:-1]

                    if inn_params.ndim_pad_x:
                        mstars = torch.cat((mstars, inn_params.add_pad_noise * noise_batch(hyper.batch_size,inn_params.ndim_pad_x,self.device)), dim=1)
                    if inn_params.add_y_noise > 0:
                        e += inn_params.add_y_noise * noise_batch(hyper.batch_size,ndim_y_e,self.device)
                        n += inn_params.add_y_noise * noise_batch(hyper.batch_size,ndim_y_n,self.device)
                        S += inn_params.add_y_noise * noise_batch(hyper.batch_size,ndim_y_S,self.device)

                    e_optim.zero_grad()
                    n_optim.zero_grad()
                    S_optim.zero_grad()

                    with torch.set_grad_enabled(True):    
                        # cond inn
                        e_pred_z, e_jac = e_model(mstars,e)
                        e_loss = losses.loss_max_likelihood(e_pred_z, e_jac)
                        e_loss.backward()
                        if hyper.clipping_gradient:    
                            torch.nn.utils.clip_grad_value_(e_model.parameters(), clip_value=hyper.clip_value)
                        e_optim.step()

                        # cc inn
                        n_pred_z, n_jac = n_model(mstars,n)
                        n_loss = losses.loss_max_likelihood(n_pred_z, n_jac)
                        n_loss.backward()
                        if hyper.clipping_gradient:    
                            torch.nn.utils.clip_grad_value_(n_model.parameters(), clip_value=hyper.clip_value)
                        n_optim.step()

                        # S inn
                        S_pred_z, S_jac = S_model(mstars,S)
                        S_loss = losses.loss_max_likelihood(S_pred_z, S_jac)
                        S_loss.backward()
                        if hyper.clipping_gradient:    
                            torch.nn.utils.clip_grad_value_(S_model.parameters(), clip_value=hyper.clip_value)
                        S_optim.step()

                    epoch_loss.append([e_loss.detach().cpu().numpy(),n_loss.detach().cpu().numpy(),S_loss.detach().cpu().numpy()])

                current_train_loss = np.mean(epoch_loss, axis=0)
                train_losses[i,:] = current_train_loss

                # validation
                e_model.eval()
                n_model.eval()
                S_model.eval()
        
                epoch_loss = []
                for (band, e, n, S) in val_loader:

                    band, e, n, S = band.to(self.device), e.to(self.device), n.to(self.device), S.to(self.device)
                    # separate mstars with Fermi level
                    mstars = band[:,:-1]

                    if inn_params.ndim_pad_x:
                        mstars = torch.cat((mstars, inn_params.add_pad_noise * noise_batch(hyper.batch_size,inn_params.ndim_pad_x,self.device)), dim=1)
                    if inn_params.add_y_noise > 0:
                        e += inn_params.add_y_noise * noise_batch(hyper.batch_size,ndim_y_e,self.device)
                        n += inn_params.add_y_noise * noise_batch(hyper.batch_size,ndim_y_n,self.device)
                        S += inn_params.add_y_noise * noise_batch(hyper.batch_size,ndim_y_S,self.device)

                    with torch.set_grad_enabled(False):    
                       
                        e_pred_z, e_jac = e_model(mstars,e)
                        e_loss = losses.loss_max_likelihood(e_pred_z, e_jac)
                        n_pred_z, n_jac = n_model(mstars,n)
                        n_loss = losses.loss_max_likelihood(n_pred_z, n_jac)
                        S_pred_z, S_jac = S_model(mstars,S)
                        S_loss = losses.loss_max_likelihood(S_pred_z, S_jac)

                    epoch_loss.append([e_loss.detach().cpu().numpy(),n_loss.detach().cpu().numpy(),S_loss.detach().cpu().numpy()])

                current_val_loss = np.mean(epoch_loss, axis=0)
                val_losses[i,:] = current_val_loss
                lrs = (get_lr(e_optim), get_lr(n_optim), get_lr(S_optim))
                visualizer.print_losses(lrs, current_train_loss, current_val_loss)

                eval_loss = (current_val_loss[0]+current_val_loss[1]+current_val_loss[2])/3

                if  eval_loss < running_loss:
                    print("Best loss: ", running_loss)
                    print("New loss: ", eval_loss, " -->  model saved.")
                    for optim,model,model_name in zip((e_optim,n_optim,S_optim),(e_model,n_model,S_model),self.paths.model_name["inn"]):
                        torch.save({'opt': optim.state_dict(), 'net': model.state_dict()}, os.path.join(self.paths.train_path,model_name+".pt"))
                    running_loss = eval_loss

                e_scheduler.step(current_val_loss[0])
                n_scheduler.step(current_val_loss[1])
                S_scheduler.step(current_val_loss[2])

            visualizer.plot_losses(hyper.epochs, train_losses, val_losses, logscale=True)

        except:
            for optim,model,model_name in zip((e_optim,n_optim,S_optim),(e_model,n_model,S_model),self.paths.model_name["inn"]):
                torch.save({'opt': optim.state_dict(), 'net': model.state_dict()}, os.path.join(self.paths.train_path,model_name+".pt_ABORT"))
            raise

        finally:
            print("\nBest loss: ", running_loss)        
            log("\nBest loss: {}".format(running_loss), self.paths.logfile)
            print("\n\nTraining inn took %f minutes\n\n" % ((time()-t_start)/60.))        
            log("\n\nTraining inn took %f minutes\n\n" % ((time()-t_start)/60.), self.paths.logfile)
            for optim,model,model_name in zip((e_optim,n_optim,S_optim),(e_model,n_model,S_model),self.paths.model_name["inn"]):
                torch.save({'opt': optim.state_dict(), 'net': model.state_dict()}, os.path.join(self.paths.train_path,model_name+".pt"))
