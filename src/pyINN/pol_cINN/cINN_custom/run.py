#import os

#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
#os.environ["CUDA_VISIBLE_DEVICES"]="1"

import os
import subprocess

import torch

from utils import *
from dataloader import TrainLoaders
from visualizer import Visualizer
from train import Trainer
from test import Tester

from models import cINN, FCSeebeck, ModelWrapper



# 1. PARAMETERS/HYPERPARAMETERS SETTING

##### PATH SETTINGS #####
ROOT = "/mnt/home/bonal1lCMICH/Documents/inverse_problem/inverse/pol_smaller_ccnew/cINN_custom"
model_name = {"inn": ["e_cINN","n_cINN","S_cINN"], "fcn": "S_fcn"}
#########################

train_path, test_path = create_expfolder(ROOT)

#########################
#  Filesystem settings  #
#########################

# Filename to export print
logfile         = os.path.join(train_path, "out.txt")
# Data folder
data_path       = os.path.join(ROOT,"data")

######################
#  Pytorch settings  #
######################

# Compute device to perform the training on, 'cuda' or 'cpu'
use_cuda        = torch.cuda.is_available()
device          = torch.device("cuda" if use_cuda else "cpu")
torch.manual_seed(random.randint(1,1234))

################################
#  Hyperparameters - TRAINING  #
################################

# Batch size
batch_size      = 300
# learning rate
lr_init         = 0.001
# Total number of epochs to train for
epochs          = 500
# wait this # of epochs before decresing lr
patience        = 65
# after patience, lr *= sch_factor
sch_factor      = 0.1
# L2 weight regularization of model parameters
l2_weight_reg   = 1e-4
#
exponent_clamping = 2.0
# sigma to initialize the model parameters
init_sigma      = 0.20
# Parameters beta1, beta2 of the Adam optimizer
adam_betas      = (0.9, 0.99)
#
use_permutation = True

#####################
#  Data dimensions  #
#####################

ndim_x      = {"inn": 6, "fcn": 8}     # 6 mstar + mu
ndim_pad_x  = 0

ndim_en     = 7     # en with a polyfit of degree 7
ndim_S      = 8     # s with a polyfit of degree 8
polydegree_str = "{}_{}_{}".format(ndim_en,ndim_en,ndim_S)

ndim_ys     = {"inn": [ndim_en,ndim_en,ndim_S], "fcn": 1}


ndim_z      = ndim_x["inn"] + ndim_pad_x

##############################
#  Hyperparameters - MODELS  #
##############################

# inn
N_blocks        = 6
#
hidden_layer_sizes = 64 # WARNING! Modify it on models.py
#
dropout_perc    = 0.4
#
batch_norm      = False
# gradient value clipping
clipping_gradient   = True
clip_value          = 2.0

# fcn
fcs_hidden_layer_sizes = [32,64,16]
#
fcs_dropout_perc = [0.1,0.2,0.2,0.1]

############
#  Losses  #
############

lambd_max_likelihood = 1e-3
num_losses           = 1

# Both for fitting, and for the reconstruction, perturb y with Gaussian
# noise of this sigma
add_y_noise     = 0.0
# For reconstruction, perturb z
add_z_noise     = 0.0
# In all cases, perturb the zero padding
add_pad_noise   = 0.0
# increse/decrease contribution pad_yz to lml
zeros_noise_scale = 100

# For noisy forward processes, the sigma on y (assumed equal in all dimensions).
# This is only used if mmd_back_weighted of train_max_likelihoiod are True.
y_uncertainty_sigma = 0.1

#################
#  Visualizers  #
#################

# fcn
lr_labels = ['lr']
loss_labels_train = ["mse"]
loss_labels_val = [l + '(val)' for l in loss_labels_train]
config_file_params = (train_path,test_path,logfile,data_path,device,batch_size,lr_init,epochs,patience,sch_factor,l2_weight_reg,exponent_clamping,init_sigma,adam_betas,use_permutation,fcs_hidden_layer_sizes,fcs_dropout_perc)
config_file_params_name = []
for par in config_file_params:
    config_file_params_name.append(get_varname(par)[:-1])
visualizer_fcn = Visualizer(lr_labels,loss_labels_train,loss_labels_val,config_file_params,config_file_params_name,train_path)

# inn
lr_labels = ['lr_e','lr_n','lr_S']
loss_labels_train = ['L_ML_e','L_ML_n','L_ML_S']
loss_labels_val = [l + '(val)' for l in loss_labels_train]
config_file_params = (train_path,test_path,logfile,data_path,device,batch_size,lr_init,epochs,patience,sch_factor,l2_weight_reg,exponent_clamping,init_sigma,adam_betas,use_permutation,ndim_x,ndim_ys,ndim_z,ndim_pad_x, N_blocks,hidden_layer_sizes,dropout_perc,lambd_max_likelihood,add_y_noise,add_z_noise,add_pad_noise,zeros_noise_scale,y_uncertainty_sigma)
config_file_params_name = []
for par in config_file_params:
    config_file_params_name.append(get_varname(par)[:-1])
visualizer_inn = Visualizer(lr_labels,loss_labels_train,loss_labels_val,config_file_params,config_file_params_name,train_path)

# 2. Export test config file for future tests
export_text_config(train_path,test_path,logfile,data_path,device,batch_size,ndim_x,ndim_ys,ndim_z,ndim_pad_x,add_y_noise,add_z_noise,add_pad_noise)

# 3. create models
e_cInn = cINN(ndim_x["inn"], ndim_pad_x, ndim_ys["inn"][0], N_blocks, exponent_clamping, use_permutation)
n_cInn = cINN(ndim_x["inn"], ndim_pad_x, ndim_ys["inn"][1], N_blocks, exponent_clamping, use_permutation)
S_cInn = cINN(ndim_x["inn"], ndim_pad_x, ndim_ys["inn"][2], N_blocks, exponent_clamping, use_permutation)
S_fcn = FCSeebeck(ndim_x["fcn"], ndim_ys["fcn"], fcs_hidden_layer_sizes, fcs_dropout_perc)

# 4. Create containers for training
train_loaders   = TrainLoaders(data_path,batch_size)
hyperparams     = HyperParams(batch_size,lr_init,epochs,init_sigma,adam_betas,patience,l2_weight_reg, clipping_gradient, clip_value)
paths           = Paths(logfile,train_path,data_path,test_path,model_name)
params_inn      = NNparams(ndim_x,ndim_ys,ndim_z,ndim_pad_x,add_y_noise,add_z_noise,add_pad_noise,num_losses)
lossparams      = LossParams(lambd_max_likelihood,y_uncertainty_sigma,zeros_noise_scale,batch_size)

hyperparams_fcn  = HyperParams(300,0.01,250,0.2,(0.9, 0.99),50,0.001,clipping_gradient, clip_value)


# 5. create models' wrappers
e_wrapper = ModelWrapper(e_cInn,train_loaders,hyperparams,device,visualizer_inn,lossparams,params_inn)
n_wrapper = ModelWrapper(n_cInn,train_loaders,hyperparams,device,visualizer_inn,lossparams,params_inn)
S_wrapper = {"fcn": ModelWrapper(S_fcn,train_loaders,hyperparams_fcn,device,visualizer_fcn),
             "inn": ModelWrapper(S_cInn,train_loaders,hyperparams,device,visualizer_inn,lossparams,params_inn)}

models    =	{"e": e_wrapper,
             "n": n_wrapper,
             "S": S_wrapper}

# 6. Training
Trainer(paths,models,device).exec()

# 7. Test
Tester(paths,models,device).exec()

# 8. Plot test results
subprocess.run(["julia","--project=.","visualize_results.jl",test_path,polydegree_str])

# 9. Plot accuracy
subprocess.run(["julia","--project=.","accuracy.jl",test_path,polydegree_str])