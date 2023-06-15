import os
import inspect
import string,random
from datetime import date, datetime

from dataclasses import dataclass

import torch

### data structures
@dataclass
class HyperParams:
    batch_size: int
    lr_init: float
    epochs: int
    init_sigma: float
    adam_betas: tuple
    patience: int
    l2_weight_reg: float
    clipping_gradient: bool
    clip_value: float


@dataclass
class Paths:
    logfile: string
    train_path: string
    data_path: string
    test_path: string
    model_name: string


@dataclass
class NNparams:
    ndim_x: int
    ndim_ys: list
    ndim_z: int
    ndim_pad_x: int
    add_y_noise: float
    add_z_noise: float
    add_pad_noise: float
    num_losses: int


@dataclass
class LossParams:
    lambd_max_likelihood:float
    y_uncertainty_sigma: float
    zeros_noise_scale: float
    batch_size: int


def create_expfolder(root_path, mu=False):
    path = os.path.join(root_path,"results")
    if not os.path.isdir(path): # if there is not a results folder -> create it
        os.mkdir(path)
    today_date = date.today().strftime("%b-%d-%Y")
    path = os.path.join(path, today_date)
    if not os.path.isdir(path): # if there is not a today folder -> create it
        os.mkdir(path)
    exp_name = datetime.now().strftime("%H-%M-%S")+"-"+''.join(random.choice(string.ascii_letters) for i in range(4))
    if mu: 
        exp_name += "_mu_optim"
    path = os.path.join(path, exp_name)
    os.mkdir(path)
    # train folder
    train_path = os.path.join(path, "train")
    os.mkdir(train_path)
    # test folder
    test_path = os.path.join(path, "test")
    os.mkdir(test_path)
    return train_path,test_path


def log(string, filepath):
    with open(filepath, 'a') as f:
        f.write(string)


def export_text_config(train_path,test_path,logfile,data_path,device,batch_size,ndim_x,ndim_y,ndim_z,ndim_pad_x,add_y_noise,add_z_noise,add_pad_noise):

    test_config = """def log(string, filepath):
    with open(filepath, 'a') as f:
        f.write(string)

train_path = "{}"   # path to the model
test_path = "{}"
logfile = "{}"
data_path = "{}"
device = "{}"

batch_size = {}

ndim_x = {}
ndim_y = {}     # ndim_y/3: num params for each tensor
ndim_z = {}
ndim_pad_x     = {}

add_y_noise    = {}
add_z_noise    = {}
add_pad_noise  = {} """.format(train_path,test_path,logfile,data_path,device,batch_size,ndim_x,ndim_y,ndim_z,ndim_pad_x,add_y_noise,add_z_noise,add_pad_noise)

    with open(os.path.join(test_path, "test_config.py"), 'w') as f:
        f.write(test_config)


def standardize(train_tensor, val_tensor, tensor_name):
    train_size = train_tensor.shape[0]
    val_size = val_tensor.shape[0]
    var = torch.cat((train_tensor, val_tensor), dim=0)
    var_mean = var.mean(dim=0, keepdim=True)
    var_std = var.std(dim=0, keepdim=True)
    var_norm = (var - var_mean) / var_std
    var_train = var_norm[:train_size,:]
    var_val = var_norm[train_size:train_size+val_size,:]
    mspath = "mean_std"
    if not os.path.isdir(mspath): # if there is not a mean_std folder -> create it
        os.mkdir(mspath)
    torch.save(var_mean, os.path.join("mean_std", tensor_name+'_mean.pt'))
    torch.save(var_std, os.path.join("mean_std", tensor_name+'_std.pt'))
    return var_train, var_val


def noise_batch(batch_size, ndim, device="cpu"):
    return torch.randn(batch_size, ndim).to(device)

def get_varname(var):
    callers_local_vars = inspect.currentframe().f_back.f_locals.items()
    return [var_name for var_name, var_val in callers_local_vars if var_val is var]