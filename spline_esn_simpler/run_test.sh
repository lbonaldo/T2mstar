#!/bin/bash

source ~/.bashrc
conda activate inverse
module load CUDA/10.1.105
python test.py
