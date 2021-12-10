import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
p = 5 # x,y,z

def gaussian(x):
    return np.sqrt(3/np.pi)*np.exp(-3*(x - 2)^2)

def f15(x):
    f1 = np.sin(x)
    f2 = (x+23)/(27)
    f3 = cos(2*x)
    f4 = x^2/9
    f5 = -log(1/(1+x))
    return np.concatenate(f1,f2,f3,f4,f5)

def f16(x):
    f1 = np.sin(x)
    f2 = (x+23)/(27)
    f3 = cos(2*x)
    f4 = x^2/9
    f5 = -log(1/(1+x))
    f6 = gaussian(x)
    return np.concatenate(f1,f2,f3,f4,f5,f6)


# batch_size = 50

# length = int(len(data)/2)
# num_batch = int(length/batch_size)

# r_pred = np.empty((length, 6))
# r_true = np.empty((length, 6))

# j = 0
# for i in range(len(num_batch)):
#     r_pred[j*batch_size:(j+1)*batch_size,:] = data[i*batch_size:(i+1)*batch_size, :]
#     r_true[j*batch_size:(j+1)*batch_size,:] = data[(i+1)*batch_size:(i+2)*batch_size, :]
#     i += 1
#     j += 1

x_pred = np.load("pred.npy")
x_test = np.load("true.npy")

x = np.arange(1,3,0.000001)
fig=plt.figure(figsize=(10,10))
for i in range(6):
    plt.plot(x, x_pred[:,i], color='#1f77b4')
    plt.plot(x, x_test[:,i], color='#ff7f0e')
    plt.savefig("fig_{}.png".format(i+1))
