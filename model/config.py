'''Global configuration'''
import torch

######################
#  General settings  #
######################

# Filename to save the model under
filename_out    = 'output/inn.pt'
# Model to load and continue training. Ignored if empty string
filename_in     = ''
# Compute device to perform the training on, 'cuda' or 'cpu'
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
# Use interactive visualization of losses and other plots. Requires visdom
interactive_visualization = False
# Run a list of python functions at test time after eacch epoch
# See toy_modes_train.py for reference example
test_time_functions = []

#######################
#  Training schedule  #
#######################

# Initial learning rate
lr_init         = 1.0e-3
#Batch size
batch_size      = 200
# Total number of epochs to train for
n_epochs        = 100
# End the epoch after this many iterations (or when the train loader is exhausted)
n_its_per_epoch = 1000
# For the first n epochs, train with a much lower learning rate. This can be
# helpful if the model immediately explodes.
pre_low_lr      = 0
# Decay exponentially each epoch, to final_decay*lr_init at the last epoch.
final_decay     = 0.02
# L2 weight regularization of model parameters
l2_weight_reg   = 1e-5
# Parameters beta1, beta2 of the Adam optimizer
adam_betas = (0.9, 0.95)

#####################
#  Data dimensions  #
#####################

ndim_x     = 6
ndim_pad_x = 2

ndim_y     = 1
ndim_z     = 5
ndim_pad_zy = 2

# Overwrite or import data loaders here.
from dataloader import tr_loader, tst_loader

train_loader = tr_loader
test_loader = tst_loader

assert (ndim_x + ndim_pad_x
        == ndim_y + ndim_z + ndim_pad_zy), "Dimensions don't match up"

############
#  Losses  #
############

train_forward_mmd    = True
train_backward_mmd   = True
train_reconstruction = True
train_max_likelihood = True

lambd_fit_forw       = 1.
lambd_mmd_forw       = 50.
lambd_reconstruct    = 1.
lambd_mmd_back       = 500.
lambd_max_likelihood = 1.

# Both for fitting, and for the reconstruction, perturb y with Gaussian
# noise of this sigma
add_y_noise     = 5e-2
# For reconstruction, perturb z
add_z_noise     = 2e-2
# In all cases, perturb the zero padding
add_pad_noise   = 1e-2

zeros_noise_scale = 5e-2

# For noisy forward processes, the sigma on y (assumed equal in all dimensions).
# This is only used if mmd_back_weighted of train_max_likelihoiod are True.
y_uncertainty_sigma = 0.12 * 4

mmd_forw_kernels = [(0.2, 2), (1.5, 2), (3.0, 2)]
mmd_back_kernels = [(0.2, 0.1), (0.2, 0.5), (0.2, 2)]
mmd_back_weighted = True

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
hidden_layer_sizes = 128
#
dropout_perc = 0.2
#
batch_norm = True
#
use_permutation = True
#
verbose_construction = False