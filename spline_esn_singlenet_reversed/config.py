'''Global configuration'''
import os
import sys
from datetime import date, datetime

import torch

def create_testfolder():
    tpath = "/mnt/scratch/bonal1lCMICH/inverse/spline_esn_singlenet_reversed/results"
    if not os.path.isdir(tpath): # if there is not a results folder -> create it
        os.mkdir(tpath)
    today_date = date.today().strftime("%b-%d-%Y")
    tpath = os.path.join(tpath, today_date)
    if not os.path.isdir(tpath): # if there is not a today folder -> create it
        os.mkdir(tpath)
    test_name = datetime.now().strftime("%H-%M-%S")
    tpath = os.path.join(tpath, test_name)
    os.mkdir(tpath)
    return tpath

def log(string, filepath):
    with open(filepath, 'a') as f:
        f.write(string)
    

test_path = create_testfolder()

######################
#  General settings  #
######################

# Filename to export print
logfile         = os.path.join(test_path, "out.txt")
# Filename to save the model under
filename_out    = os.path.join(test_path, 'inn.pt')
# Data folder
data_path       = "/mnt/scratch/bonal1lCMICH/inverse/spline_esn_singlenet_reversed/data"
# Model to load and continue training. Ignored if empty string
filename_in     = ''
# Compute device to perform the training on, 'cuda' or 'cpu'
use_cuda        = torch.cuda.is_available()
device          = torch.device("cuda" if use_cuda else "cpu")
# Use interactive visualization of losses and other plots. Requires visdom
interactive_visualization = False
# Run a list of python functions at test time after eacch epoch
# See toy_modes_train.py for reference example
test_time_functions = []

#######################
#  Training schedule  #
#######################

# Initial learning rate
lr_init         = 1e-3
#Batch size
batch_size      = 100
# Total number of epochs to train for
n_epochs        = 60
# End the epoch after this many iterations (or when the train loader is exhausted)
n_its_per_epoch = 5000
# For the first n epochs, train with a lower learning rate (lr_init*0.1). This can be
# helpful if the model immediately explodes.
pre_low_lr      = 5
# Decay exponentially each epoch, to final_decay*lr_init at the last epoch.
final_decay     = 0.01
# L2 weight regularization of model parameters
l2_weight_reg   = 5e-2
# Parameters beta1, beta2 of the Adam optimizer
adam_betas = (0.9, 0.95)

#####################
#  Data dimensions  #
#####################

ndim_x     = 9  
ndim_pad_x = 13

ndim_y     = 18 
ndim_z     = 4
ndim_pad_zy = 0

assert (ndim_x + ndim_pad_x
        == ndim_y + ndim_z + ndim_pad_zy), "Dimensions don't match up"

from dataloader import train_loader, val_loader

############
#  Losses  #
############

train_forward_mmd    = True
train_backward_mmd   = True
train_reconstruction = True
train_max_likelihood = True

lambd_fit_forw       = 1e-3
lambd_mmd_forw       = 1.
lambd_reconstruct    = 1.
lambd_mmd_back       = 10.
lambd_max_likelihood = 1e-3

# Both for fitting, and for the reconstruction, perturb y with Gaussian
# noise of this sigma
add_y_noise     = 1e-2
# For reconstruction, perturb z
add_z_noise     = 1e-2
# In all cases, perturb the zero padding
add_pad_noise   = 1e-4
# increse/decrease contribution pad_yz to lml
zeros_noise_scale = 100

# For noisy forward processes, the sigma on y (assumed equal in all dimensions).
# This is only used if mmd_back_weighted of train_max_likelihoiod are True.
y_uncertainty_sigma = 0.1

mmd_forw_kernels = [(1, 2), (1, 2), (1, 2)]
mmd_back_kernels = [(1, 0.1), (1, 0.5), (1, 2)]
mmd_back_weighted = True

###########
#  Model  #
###########

# Initialize the model parameters from a normal distribution with this sigma
init_scale = 0.30
#
N_blocks   = 6
#
exponent_clamping = 2.0
#
hidden_layer_sizes = 128
#
dropout_perc = 0.2
#
batch_norm = False
#
use_permutation = False
#
verbose_construction = False