#import os

#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
#os.environ["CUDA_VISIBLE_DEVICES"]="1"

import os
import subprocess

import pandas as pd
import torch

from utils import *
from dataloader import TrainLoaders
from visualizer import Visualizer
from train import Trainer
from test import Tester

from models import FCSeebeck, ModelWrapper


# 1. PARAMETERS/HYPERPARAMETERS SETTING

##### PATH SETTINGS #####
ROOT = "/mnt/home/bonal1lCMICH/Documents/inverse_problem/inverse/pol_smaller_ccnew/cINN_custom"
#########################

train_path, test_path = create_expfolder(ROOT,True)

#########################
#  Filesystem settings  #
#########################

# Filename to export print
logfile         = os.path.join(train_path,"out.txt")
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

# Total number of epochs to train for
epochs          = 250
# after patience, lr *= sch_factor
sch_factor      = 0.1
#
exponent_clamping = 2.0
# sigma to initialize the model parameters
init_sigma      = 0.20
# Parameters beta1, beta2 of the Adam optimizer
adam_betas      = (0.9, 0.99)

#####################
#  Data dimensions  #
#####################

ndim_x      = 8
ndim_y      = 1

#  Visualizer  #
lr_labels = ['lr']
loss_labels_train = ["mse"]
loss_labels_val = [l + '(val)' for l in loss_labels_train]

def single_run(batch_size,lr_init,l2_weight_reg,fcs_dropout_perc,fcs_hidden_layer_sizes,counter):  

    patience = 65
    config_file_params = (train_path,test_path,logfile,data_path,device,batch_size,lr_init,epochs,patience,sch_factor,l2_weight_reg,exponent_clamping,init_sigma,adam_betas,False,fcs_hidden_layer_sizes,fcs_dropout_perc)
    config_file_params_name = ('train_path','test_path','logfile','data_path','device','batch_size','lr_init','epochs','patience','sch_factor','l2_weight_reg','exponent_clamping','init_sigma','adam_betas','permuation','fcs_hidden_layer_sizes','fcs_dropout_perc')
    visualizer_fcn = Visualizer(lr_labels,loss_labels_train,loss_labels_val,config_file_params,config_file_params_name,train_path)
    model_name = {"inn": ["e_cINN","n_cINN","S_cINN"], "fcn": "S_fcn_{}".format(counter)}

    # 3. create models
    S_fcn = FCSeebeck(ndim_x, ndim_y, fcs_hidden_layer_sizes, fcs_dropout_perc)

    # 4. Create containers for training
    train_loaders   = TrainLoaders(data_path,batch_size)
    hyperparams     = HyperParams(batch_size,lr_init,epochs,init_sigma,adam_betas,patience,l2_weight_reg,False,0.0)
    paths           = Paths(logfile,train_path,data_path,test_path,model_name)

    # 5. create models' wrappers
    wrapper = {"S": {"fcn": ModelWrapper(S_fcn,train_loaders,hyperparams,device,visualizer_fcn)}}

    # 6. Training
    Trainer(paths,wrapper,device).train_fcn()

    # 7. Test
    loss, error = Tester(paths,wrapper,device).test_fcn()

    return loss,error


### HYPERPARAMER RANGES
patience_v                  = [50,76,25]
# fcs_hidden_layer_sizes_v    = [[32,64,32,16],[32,64,16],[32,64,64,32,16],[16,32,32,16],[16,32,64,32,16]]
# fcs_dropout_perc_v          = [[.2,.3,.2,.1],[.1,.2,.1],[.1,.2,.2,.2,.1],[.1,.2,.2,.1],[.1,.2,.3,.2,.1]]
fcs_hidden_layer_sizes_v    = [[16,32,32,16]] #,,,
fcs_dropout_perc_v          = [[.1,.2,.2,.2,.1]] #,,,
l2_weight_reg_v             = [1e-4,2e-4,5e-4,1e-3]
lr_init_v                   = [0.01,0.005,0.002,0.001,0.0005]
batch_size_v                = range(150,350,50)


### create grid
table = pd.DataFrame(columns = ['batch_size','lr_init','l2_weight_reg','fcs_dropout_perc','fcs_hidden_layer_sizes','patience','loss','error'])
  
### populate grid
counter = 0
# for patience in patience_v:
for i,(fcs_hidden_layer_sizes,fcs_dropout_perc) in enumerate(zip(fcs_hidden_layer_sizes_v,fcs_dropout_perc_v)):
    for l2_weight_reg in l2_weight_reg_v:
        for lr_init in lr_init_v:
            for batch_size in batch_size_v:
                # loss, error = single_run(batch_size,lr_init,l2_weight_reg,fcs_dropout_perc,fcs_hidden_layer_sizes,patience,counter)
                loss, error = single_run(batch_size,lr_init,l2_weight_reg,fcs_dropout_perc,fcs_hidden_layer_sizes,counter)
                
                table = table.append({'batch_size' : batch_size, 
                                        'lr_init' : lr_init, 
                                        'l2_weight_reg' : l2_weight_reg,
                                        'fcs_hidden_layer_sizes' : "model_{}".format(i), 
                                        'fcs_dropout_perc' : "drop_{}".format(i), 
                                        # 'patience' : patience,
                                        'loss' : loss, 
                                        'error' : error},
                                        ignore_index = True)
                    
                counter += 1
                # export grid
                table.to_csv(os.path.join(test_path, "grid_search.csv"))





