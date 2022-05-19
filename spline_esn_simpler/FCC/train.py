from time import time

from torchsummary import summary

import numpy as np
import matplotlib.pyplot as plt
import torch

import config as c
from dataloader import train_loader, val_loader

from losses import MSE, cstMSE


class Trainer(object):
    def __init__(self, NNet):
        torch.multiprocessing.freeze_support()

        if c.device is "cuda":
            print(torch.cuda.get_device_name(0))        
            c.log(torch.cuda.get_device_name(0), c.logfile)

        self.model = NNet(c.ndim_x,c.ndim_y,c.dim_layers,c.dropout_perc)
        self.model.to(c.device)
        
        # if isinstance(c.ndim_x,int):
        #     summary(self.model, c.ndim_x)
        # else:
        #     dim = [(i) for i in c.ndim_x]
        #     summary(self.model, dim)

        params = list(filter(lambda p: p.requires_grad, self.model.parameters()))
        for p in params:
            p.data = c.init_scale * torch.randn(p.data.shape).to(c.device)
        
        self.optimizer = torch.optim.Adam(params, lr=c.lr_init, betas=c.adam_betas, eps=1e-6, weight_decay=c.l2_weight_reg)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=c.sch_factor, patience=c.patience, threshold=0.001, threshold_mode='rel', cooldown=0, min_lr=1e-06, eps=1e-08)
        self.criterion = cstMSE if c.Loss == "cstMSE" else MSE()

    def exec(self, epochs=c.n_epochs):
        running_loss = np.Inf
        t_start = time()
        try:
            train_losses = []
            val_losses = []
            c.visualizer.print_config()
            for i_epoch in range(epochs):

                epoch_train_loss = self.train()
                epoch_val_loss  = self.eval()

                if  epoch_val_loss < running_loss:
                    print("Best loss: ", running_loss)
                    print("New loss: ", epoch_val_loss, " -->  model saved.")
                    self.model.save(c.filename_out)
                    running_loss = epoch_val_loss

                c.visualizer.print_losses(self.scheduler.optimizer.param_groups[0]['lr'],[epoch_train_loss, epoch_val_loss])
                train_losses.append(epoch_train_loss)
                val_losses.append(epoch_val_loss)
                self.scheduler.step(epoch_val_loss)
            c.visualizer.plot_losses(train_losses, val_losses, logscale=True)
            
        except:
            self.model.save(c.filename_out + '_ABORT')
            raise

        finally:
            print("\nBest loss: ", running_loss)        
            c.log("\nBest loss: {}".format(running_loss), c.logfile)
            print("\n\nTraining took %f minutes\n\n" % ((time()-t_start)/60.))        
            c.log("\n\nTraining took %f minutes\n\n" % ((time()-t_start)/60.), c.logfile)
            self.model.save(c.filename_out)

    def train(self):
        self.model.train()
        
        batch_idx = 0
        batch_losses = []

        for (band, e, s, n) in train_loader:
            if batch_idx > c.n_its_per_epoch:
                break
            batch_idx += 1

            band, e, s, n = band.to(c.device), e.to(c.device), s.to(c.device), n.to(c.device)

            self.optimizer.zero_grad()
            with torch.set_grad_enabled(True):    
                # forward step
                out_band = self.model(e,s,n)
                loss = self.criterion(out_band,band)
                # backward step
                loss.backward()
                self.optimizer.step()
                batch_losses.append(loss.detach().cpu().numpy())
        return np.mean(batch_losses,axis=0)
    
    def eval(self):
        self.model.eval()
        
        batch_losses = []

        for (band, e, s, n) in val_loader:

            band, e, s, n = band.to(c.device), e.to(c.device), s.to(c.device), n.to(c.device)

            with torch.set_grad_enabled(False):    
                out_band = self.model(e,s,n)
                loss = self.criterion(out_band,band)
                batch_losses.append(loss.detach().cpu().numpy())
        return np.mean(batch_losses,axis=0)

