#import os

#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
#os.environ["CUDA_VISIBLE_DEVICES"]="1"

import pathlib
from time import time

import numpy as np
import matplotlib.pyplot as plt
import torch

import config as c

from train import Trainer
from test import Tester

if __name__ == '__main__':
    Trainer(c.NNet).exec()
    exp_path = pathlib.PurePath(c.train_path).parents[0]
    Tester(c.NNet,exp_path).exec()
