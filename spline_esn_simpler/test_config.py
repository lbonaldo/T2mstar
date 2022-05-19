def log(string, filepath):
    with open(filepath, 'a') as f:
        f.write(string)

test_path = "/mnt/scratch/bonal1lCMICH/inverse/spline_esn_simpler/results/May-17-2022/21-21-10"
logfile = "/mnt/scratch/bonal1lCMICH/inverse/spline_esn_simpler/results/May-17-2022/21-21-10/out.txt"
data_path = "/mnt/scratch/bonal1lCMICH/inverse/spline_esn_simpler/data"
device = "cuda"

batch_size = 200
hidden_layer_sizes = 64
N_blocks   = 6
exponent_clamping = 2.0
dropout_perc = 0.2
batch_norm = False
use_permutation = False

ndim_x = 7
ndim_pad_x = 2

ndim_y = 8
ndim_z = 1
ndim_pad_zy = 0

add_pad_noise = 0.0001
add_y_noise = 0.005