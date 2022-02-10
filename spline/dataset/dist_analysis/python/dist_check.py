import numpy as np
import matplotlib.pyplot as plt

coeff = ['a','b','c','d','mu']

y_train = np.load("y_train.npy")
y_eval = np.load("y_eval.npy")
y_test = np.load("y_test.npy")

fig, ax = plt.subplots(3, 1, figsize=(20,10))
for i in range(len(coeff)):
    plt.suptitle("Pred vs True - "+coeff[i], fontsize=22)
    ax[0].hist(y_train[:,i])
    ax[1].hist(y_eval[:,i])
    ax[2].hist(y_test[:,i])
    plt.savefig("test_dist_{}.jpg".format(coeff[i]))
    ax[0].cla(); ax[1].cla(); ax[2].cla()