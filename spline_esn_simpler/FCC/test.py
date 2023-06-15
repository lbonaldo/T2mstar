import os
import sys
from time import time

import numpy as np

import torch

#from losses import MSE
from models import FCC1, FCC2, FCC3


class Tester(object):

    def __init__(self, NNet, exp_path):
        torch.multiprocessing.freeze_support()

        os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
        #os.environ["CUDA_VISIBLE_DEVICES"]="2"

        test_path = os.path.join(exp_path, "test")
        sys.path.insert(1,test_path)

        import test_config as c
        self.data_path = c.data_path
        self.test_path = c.test_path
        self.device = c.device
        self.ndim_x = c.ndim_x
        self.ndim_y = c.ndim_y
        self.dim_layers = c.dim_layers
        self.dropout_perc = c.dropout_perc
        self.batch_size = c.batch_size
        self.log = c.log
        self.logfile = c.logfile

        self.model = NNet(self.ndim_x,self.ndim_y,self.dim_layers,self.dropout_perc)
        self.model.load(os.path.join(exp_path, "train", 'fcc.pt'))
        self.model.to(self.device)

        self.criterion = torch.nn.MSELoss()

        # DATASET IMPORT
        self.bandparams_test = torch.Tensor(np.load(os.path.join(self.data_path, 'x_test.npy')))
        self.bandparams_mean = torch.load(os.path.join("mean_std", 'x_mean.pt'))
        self.bandparams_std = torch.load(os.path.join("mean_std", 'x_std.pt'))
        bandparams_test_norm = (self.bandparams_test - self.bandparams_mean) / self.bandparams_std

        test_e = torch.Tensor(np.load(os.path.join(self.data_path, 'y_sigma_test.npy')))
        sigma_mean = torch.load(os.path.join("mean_std", 'y_sigma_mean.pt'))
        sigma_std = torch.load(os.path.join("mean_std", 'y_sigma_std.pt'))
        test_norm_e = (test_e - sigma_mean) / sigma_std

        test_s = torch.Tensor(np.load(os.path.join(self.data_path, 'y_seebeck_test.npy')))
        seebeck_mean = torch.load(os.path.join("mean_std", 'y_seebeck_mean.pt'))
        seebeck_std = torch.load(os.path.join("mean_std", 'y_seebeck_std.pt'))
        test_norm_s = (test_s - seebeck_mean) / seebeck_std

        test_n = torch.Tensor(np.load(os.path.join(self.data_path, 'y_n_test.npy')))
        n_mean = torch.load(os.path.join("mean_std", 'y_n_mean.pt'))
        n_std = torch.load(os.path.join("mean_std", 'y_n_std.pt'))
        test_norm_n = (test_n - n_mean) / n_std

        self.test_size = test_e.shape[0]
        print("Test dataset size: ", self.test_size)

        self.test_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(bandparams_test_norm, test_norm_e, test_norm_s, test_norm_n),
            batch_size=self.batch_size, shuffle=False, drop_last=True)

    # exec test
    def exec(self):
        t_start = time()
        try:
            test_losses  = self.test()
            print("Test results: ", test_losses)
            self.log("Test results: %f" % test_losses, self.logfile)

        finally:
            print("\n\nTesting took %f minutes\n\n" % ((time()-t_start)/60.))        
            self.log("\n\nTesting took %f minutes\n\n" % ((time()-t_start)/60.), self.logfile)

    def test(self):
        self.model.eval()
        batch_idx = 0
        batch_num = int(np.floor(self.test_size / self.batch_size)) # drop last, see TensorDataset
        final_model_norm = torch.empty((batch_num*self.batch_size, self.ndim_y))
        batch_losses = []
        for (band, e, s, n) in self.test_loader:

            band, e, s, n = band.to(self.device), e.to(self.device), s.to(self.device), n.to(self.device)

            with torch.set_grad_enabled(False):    
                out_band_batch = self.model(e,s,n)
                loss = self.criterion(out_band_batch,band)
                batch_losses.append(loss.detach().cpu().numpy())
                final_model_norm[batch_idx*self.batch_size:(batch_idx+1)*self.batch_size, :] = out_band_batch[:,:self.ndim_y]

        band_pred = (final_model_norm.detach().cpu()*self.bandparams_std + self.bandparams_mean)
        print(self.bandparams_mean)
        print(self.bandparams_std)
        band_true = self.bandparams_test[:band_pred.shape[0],:] # dataloader skip last batch
        np.savetxt(os.path.join(self.test_path, "band_pred.txt"), band_pred.numpy(), delimiter=',')
        err = np.abs(band_true-band_pred)
        np.savetxt(os.path.join(self.test_path, "band_abs_err.txt"), err, delimiter=',')
        comb = np.empty((2*(band_true.shape[0]),band_true.shape[1]))
        for i in range(band_true.shape[0]):
            comb[2*i,:] = band_true[i,:]
            comb[2*i+1,:] = band_pred[i,:]
        np.savetxt(os.path.join(self.test_path, "band_comb.txt"), comb, delimiter=',')

        return np.mean(batch_losses,axis=0)


if __name__ == '__main__':
    torch.multiprocessing.freeze_support()

    if len(sys. argv) < 2:
        exit("Experiment path missing") 
    Tester(FCC1,sys.argv[1]).exec()

