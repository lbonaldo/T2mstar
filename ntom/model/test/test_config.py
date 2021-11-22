'''Global configuration'''
import os
import sys
from datetime import date, datetime

import torch

######################
#  General settings  #
######################

# Compute device to perform the training on, 'cuda' or 'cpu'
use_cuda        = torch.cuda.is_available()
device          = torch.device("cuda" if use_cuda else "cpu")

#######################
#  Test schedule  #
#######################

# Batch size
batch_size      = 300

ndim_x     = 6
ndim_pad_x = 0

ndim_y     = 1
ndim_z     = 5
ndim_pad_zy = 0

assert (ndim_x + ndim_pad_x
        == ndim_y + ndim_z + ndim_pad_zy), "Dimensions don't match up"


# Both for fitting, and for the reconstruction, perturb y with Gaussian
# noise of this sigma
add_y_noise     = 0.0
# For reconstruction, perturb z
add_z_noise     = 2e-3
# In all cases, perturb the zero padding
add_pad_noise   = 0.0

zeros_noise_scale = 1e3

###########
#  Model  #
###########

# Initialize the model parameters from a normal distribution with this sigma
init_scale = 0.10
#
N_blocks   = 6
#
exponent_clamping = 2.0
#
hidden_layer_sizes = 256
#
dropout_perc = 0.2
#
batch_norm = True
#
use_permutation = True
#
verbose_construction = False
