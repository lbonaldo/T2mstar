import numpy as np
import torch

# from utils import noise_batch


class Losses(object):
    '''Losses class for training fcn and inn'''

    def __init__(self, lossparams, device="cpu",others=None):
        super(Losses, self).__init__()

        self.params = lossparams

    def loss_max_likelihood(self, z, jac):
        loss_z = torch.sum(z**2, dim=1)   # z^2
        loss = 0.5*loss_z  - jac
        return self.params.lambd_max_likelihood * torch.mean(loss)









