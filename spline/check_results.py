import os
import numpy as np

test_path = "/mnt/scratch/bonal1lCMICH/inverse/spline/data/y_test.npy"
pred_path = "/mnt/scratch/bonal1lCMICH/inverse/spline/results/Jan-13-2022/14-22-55/test/y_pred.txt"

y_pred = np.loadtxt(pred_path, delimiter=',')
y_true = np.load(test_path)[:y_pred.shape[0],:]

res = np.column_stack([y_true, y_pred, y_true-y_pred])
np.savetxt("res.txt", res, delimiter=',')