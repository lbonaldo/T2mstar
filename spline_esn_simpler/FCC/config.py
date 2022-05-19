'''Global configuration'''
import os
import string,random
from datetime import date, datetime
from models import FCC1,FCC2,FCC3
from monitoring import Visualizer

import torch

def create_expfolder():
    tpath = "/mnt/scratch/bonal1lCMICH/inverse/spline_esn_simpler/results"
    if not os.path.isdir(tpath): # if there is not a results folder -> create it
        os.mkdir(tpath)
    today_date = date.today().strftime("%b-%d-%Y")
    tpath = os.path.join(tpath, today_date)
    if not os.path.isdir(tpath): # if there is not a today folder -> create it
        os.mkdir(tpath)
    exp_name = datetime.now().strftime("%H-%M-%S")+"-"+''.join(random.choice(string.ascii_letters) for i in range(4))
    tpath = os.path.join(tpath, exp_name)
    os.mkdir(tpath)
    # train folder
    train_path = os.path.join(tpath, "train")
    os.mkdir(train_path)
    # test folder
    test_path = os.path.join(tpath, "test")
    os.mkdir(test_path)
    return train_path,test_path

def log(string, filepath):
    with open(filepath, 'a') as f:
        f.write(string)

def export_text_config(test_path,test_config):
    with open(os.path.join(test_path, "test_config.py"), 'w') as f:
        f.write(test_config)

train_path,test_path = create_expfolder()

######################
#  General settings  #
######################

# Filename to export print
logfile         = os.path.join(train_path, "out.txt")
# Filename to save the model under
filename_out    = os.path.join(train_path, 'fcc.pt')
# Data folder
data_path       = "/mnt/scratch/bonal1lCMICH/inverse/spline_esn_simpler/data"
# Compute device to perform the training on, 'cuda' or 'cpu'
use_cuda        = torch.cuda.is_available()
device          = torch.device("cuda" if use_cuda else "cpu")

#####################
#  Data dimensions  #
#####################

ndim_x     = [8,8,8] # esn with a polyfit of degree 5
ndim_y     = 7  # seven parameters for band structure

#######################
#  Training schedule  #
#######################

# Initialize the model parameters from a normal distribution with this sigma
init_scale      = 0.20
# Initial learning rate
lr_init         = 0.1
#Batch size
batch_size      = 500
# Total number of epochs to train for
n_epochs        = 500
# End the epoch after this many iterations (or when the train loader is exhausted)
n_its_per_epoch = 5000
# wait this # of epochs before decresing lr
patience        = 50
# after patience, lr *= sch_factor
sch_factor      = 0.1
# L2 weight regularization of model parameters
l2_weight_reg   = 1e-15
# Parameters beta1, beta2 of the Adam optimizer
adam_betas      = (0.9, 0.99)
# dimension of each layer
# dim_layers  = [64,14]
# NNet = FCC1
# dim_layers1 = [32,64,128,64,32]
# dim_layers2 = [16,32,64,32,16]
# dim_layers  = [dim_layers1,dim_layers2]
# NNet = FCC2
dim_layers1 = [16,32,32,16]
dim_layers2 = [16,32,32,16]
dim_layers3 = [16,32,32,16]
dim_layers  = [dim_layers1,dim_layers2,dim_layers3]
NNet = FCC3
# dropout_perc for each layer
dropout_perc    = [.0,.0,.0,.0]
# loss
Loss = "MSE"
#Loss = "cstMSE"
loss_weights = [0.1,0.1,0.1,0.1,0.1,0.1,.4]

################
#  Visualizer  #
################
loss_labels_train = [Loss]
lr_labels = ['lr']
loss_labels_val = [l + '(val)' for l in loss_labels_train]

visualizer = Visualizer([loss_labels_train,loss_labels_val], lr_labels)

######################
#  Export Test File  #
######################

test_config = """def log(string, filepath):
    with open(filepath, 'a') as f:
        f.write(string)

train_path = "{}"   # path to the model
test_path = "{}"
logfile = "{}"
data_path = "{}"
device = "{}"
Loss = "{}"
loss_weights = {}

batch_size = {}
dropout_perc = {}
dim_layers = {}

ndim_x = {}
ndim_y = {}""".format(train_path,test_path,logfile,data_path,device,Loss,loss_weights,batch_size,dropout_perc,dim_layers,ndim_x,ndim_y)

export_text_config(test_path,test_config)

