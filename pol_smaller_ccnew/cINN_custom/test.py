import os
import sys
from time import time

import numpy as np
import torch

import losses
from utils import log, noise_batch
from dataloader import TestLoaders



class Tester(object):

    def __init__(self,paths,models,device="cpu"):
        torch.multiprocessing.freeze_support()
        os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"

        self.models = models
        self.paths = paths
        self.device = device

    def exec(self):
        t_start = time()
        try:
            _ = self.test_fcn()
            e_test_losses,n_test_losses,S_test_losses,enS_test_losses = self.run_test()
            print("Test results: e = {}, n = {}, S = {}, enS = {}".format(e_test_losses,n_test_losses,S_test_losses,enS_test_losses))
            log("Test results: e = %f, n = %f, S = %f, enS = %f" % (e_test_losses,n_test_losses,S_test_losses,enS_test_losses), self.paths.logfile)

        finally:
            print("\n\nTesting took %f minutes\n\n" % ((time()-t_start)/60.))        
            log("\n\nTesting took %f minutes\n\n" % ((time()-t_start)/60.), self.paths.logfile)
        

    def run_test(self):
        # cond cinn
        e_wrapper = self.models["e"]
        e_model = e_wrapper.model
        e_model.eval()
        state_dicts = torch.load(os.path.join(self.paths.train_path,self.paths.model_name["inn"][0]+".pt"),map_location=torch.device(self.device))
        e_model.load_state_dict(state_dicts['net'])

        # nn cinn
        n_model = self.models["n"].model
        n_model.eval()
        state_dicts = torch.load(os.path.join(self.paths.train_path,self.paths.model_name["inn"][1]+".pt"),map_location=torch.device(self.device))
        n_model.load_state_dict(state_dicts['net'])

        # seebeck cinn
        S_cinn_model = self.models["S"]["inn"].model
        S_cinn_model.eval()
        state_dicts = torch.load(os.path.join(self.paths.train_path,self.paths.model_name["inn"][2]+".pt"),map_location=torch.device(self.device))
        S_cinn_model.load_state_dict(state_dicts['net'])

        # Seebeck fcn
        S_fcn_model = self.models["S"]["fcn"].model
        S_fcn_model.eval()
        state_dicts = torch.load(os.path.join(self.paths.train_path,self.paths.model_name["fcn"]+".pt"),map_location=torch.device(self.device))
        S_fcn_model.load_state_dict(state_dicts['net'])

        # hyper and data are the same for both the nets
        hyper = e_wrapper.hyparams
        nn_params = e_wrapper.others

        ndim_y_e = nn_params.ndim_ys["inn"][0]
        ndim_y_n = nn_params.ndim_ys["inn"][1]
        ndim_y_S = nn_params.ndim_ys["inn"][2]

        # input is 6mstars +mu
        ndim_x = nn_params.ndim_x["inn"] + nn_params.ndim_ys["fcn"]

        # DATASET IMPORT
        loader = TestLoaders(self.paths.data_path,hyper.batch_size)
        test_loader = loader.get_dataloaders()

        batch_num = int(np.floor(loader.test_size / hyper.batch_size)) # drop last, see TensorDataset
        
        e_batch_losses = []
        n_batch_losses = []
        S_batch_losses = []
        enS_batch_losses = []
        
        e_final_model_norm = torch.empty((batch_num*hyper.batch_size, ndim_x))
        n_final_model_norm = torch.empty((batch_num*hyper.batch_size, ndim_x))
        S_final_model_norm = torch.empty((batch_num*hyper.batch_size, ndim_x))
        enS_final_model_norm = torch.empty((batch_num*hyper.batch_size, ndim_x))

        batch_idx = 0
        with torch.set_grad_enabled(False):
            for (band_norm_true, e, n, S) in test_loader:

                band_norm_true, e, n, S = band_norm_true.to(self.device), e.to(self.device), n.to(self.device), S.to(self.device)

                if nn_params.add_y_noise > 0:
                    e += nn_params.add_y_noise * losses.noise_batch(hyper.batch_size,ndim_y_e,self.device)
                    n += nn_params.add_y_noise * losses.noise_batch(hyper.batch_size,ndim_y_n,self.device)
                    S += nn_params.add_y_noise * losses.noise_batch(hyper.batch_size,ndim_y_S,self.device)

                z = noise_batch(hyper.batch_size,nn_params.ndim_z,self.device)

                # mstars predictions
                mstars_e_pred,_ = e_model(z, e, rev=True)
                mstars_n_pred,_ = n_model(z, n, rev=True)
                mstars_S_pred,_ = S_cinn_model(z, S, rev=True)

                pred_mstars = (mstars_e_pred[:,:nn_params.ndim_x["inn"]] + 
                                mstars_n_pred[:,:nn_params.ndim_x["inn"]] + 
                                mstars_S_pred[:,:nn_params.ndim_x["inn"]])/3

                # mu predictions
                mu_pred = S_fcn_model(S)
                
                # band prediction
                band_norm_e_pred = torch.cat((mstars_e_pred[:,:nn_params.ndim_x["inn"]],mu_pred), dim=1)
                band_norm_n_pred = torch.cat((mstars_n_pred[:,:nn_params.ndim_x["inn"]],mu_pred), dim=1)
                band_norm_S_pred = torch.cat((mstars_S_pred[:,:nn_params.ndim_x["inn"]],mu_pred), dim=1)
                band_norm_enS_pred = torch.cat((pred_mstars,mu_pred), dim=1)

                # errors
                e_batch_losses.append(torch.nn.functional.mse_loss(band_norm_e_pred,band_norm_true).detach().cpu().numpy())
                n_batch_losses.append(torch.nn.functional.mse_loss(band_norm_n_pred,band_norm_true).detach().cpu().numpy())
                S_batch_losses.append(torch.nn.functional.mse_loss(band_norm_S_pred,band_norm_true).detach().cpu().numpy())
                enS_batch_losses.append(torch.nn.functional.mse_loss(band_norm_enS_pred,band_norm_true).detach().cpu().numpy())

                # save predictions
                e_final_model_norm[batch_idx*hyper.batch_size:(batch_idx+1)*hyper.batch_size, :] = band_norm_e_pred
                n_final_model_norm[batch_idx*hyper.batch_size:(batch_idx+1)*hyper.batch_size, :] = band_norm_n_pred
                S_final_model_norm[batch_idx*hyper.batch_size:(batch_idx+1)*hyper.batch_size, :] = band_norm_S_pred
                enS_final_model_norm[batch_idx*hyper.batch_size:(batch_idx+1)*hyper.batch_size, :] = band_norm_enS_pred
                batch_idx += 1

        # true band (not normalized)
        band_true = loader.test_band[:e_final_model_norm.shape[0],:] # dataloader skip last batch
        print("Band mean: ", loader.band_mean)
        print("Band std: ", loader.band_std)
        # predictions
        for name,final_model_norm in zip(("e","n","S","enS"),(e_final_model_norm,n_final_model_norm,S_final_model_norm,enS_final_model_norm)):
            # unormalized prediction
            pred_band = (final_model_norm.detach().cpu()*loader.band_std + loader.band_mean)
            np.savetxt(os.path.join(self.paths.test_path, name+"_band_pred.txt"), pred_band.numpy(), delimiter=',')
            # abs errors
            err = np.abs(band_true-pred_band)
            np.savetxt(os.path.join(self.paths.test_path, name+"_band_abs_err.txt"), err, delimiter=',')
            # combine true and pred for visualization
            comb = np.empty((2*(band_true.shape[0]),band_true.shape[1]))
            for i in range(band_true.shape[0]):
                comb[2*i,:] = band_true[i,:]
                comb[2*i+1,:] = pred_band[i,:]
            np.savetxt(os.path.join(self.paths.test_path, name+"_band_comb.txt"), comb, delimiter=',')

        return np.mean(e_batch_losses,axis=0),np.mean(n_batch_losses,axis=0),np.mean(S_batch_losses,axis=0),np.mean(enS_batch_losses,axis=0)

    def test_fcn(self):
        # load the best model
        wrapper = self.models["S"]["fcn"]
        model = wrapper.model
        model.eval()
        state_dicts = torch.load(os.path.join(self.paths.train_path,self.paths.model_name["fcn"]+".pt"),map_location=torch.device(self.device))
        model.load_state_dict(state_dicts['net'])

        # DATASET IMPORT
        loader = TestLoaders(self.paths.data_path,wrapper.hyparams.batch_size)
        test_loader = loader.get_dataloaders()

        batch_num = int(np.floor(loader.test_size / wrapper.hyparams.batch_size))
        mu_norm_pred = torch.empty(batch_num*wrapper.hyparams.batch_size)

        loss = []
        batch_idx = 0
        with torch.set_grad_enabled(False):
            for (band,_,_,S) in test_loader:
                band, S = band.to(self.device), S.to(self.device)
                # separate mstars with Fermi level
                mu = band[:,-1]
                # get mu prediction 
                pred_mu = torch.squeeze(model(S))
                loss.append(torch.nn.functional.mse_loss(mu,pred_mu).detach().cpu().numpy())
                # save mu prediction
                mu_norm_pred[batch_idx*wrapper.hyparams.batch_size:(batch_idx+1)*wrapper.hyparams.batch_size] = pred_mu
                batch_idx += 1
        
        # record test loss
        print("Loss test: ", np.mean(loss,axis=0))
        log("Loss test: %f" % np.mean(loss,axis=0), self.paths.logfile)
        # mean and std of mu_true
        print(loader.band_mean)
        mu_mean = loader.band_mean[0,-1].item()
        mu_std = loader.band_std[0,-1].item()
        print("True mu mean: ", mu_mean)
        log("True mu mean: %f" % mu_mean, self.paths.logfile)
        print("True mu std: ", mu_std)
        log("True mu std: %f" % mu_std, self.paths.logfile)
        # true mu (not normalized)
        mu_true = loader.test_band[:batch_num*wrapper.hyparams.batch_size,-1] # dataloader skip last batch
        # unormalized prediction
        mu_pred = (mu_norm_pred.detach().cpu()*mu_std + mu_mean)
        # mean and std of mu_pred
        print(mu_pred)
        mu_pred_mean = torch.mean(mu_pred)
        mu_pred_std = torch.std(mu_pred)
        print("Pred mu mean: ", mu_pred_mean)
        log("Pred mu mean: %f" % mu_pred_mean, self.paths.logfile)
        print("Pred mu std: ", mu_pred_std)
        log("Pred mu std: %f" % mu_pred_std, self.paths.logfile)
        # abs errors
        err = torch.abs(mu_true-mu_pred).numpy()
        print("Mean absolute error: ", np.mean(err,axis=0))
        log("Mean absolute error: %f" % np.mean(err,axis=0), self.paths.logfile)
    
        return np.mean(loss,axis=0),np.mean(err,axis=0)
