from xmlrpc.server import DocXMLRPCRequestHandler
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def l(x,a,b):
    if -a*x - b > 0:
        return np.sqrt(-a*x - b)
    else:
        return 0

def r(x, c, d):
    if c*x - d > 0:
        return np.sqrt(c*x - d)
    else:
        return 0

def g(x, beta, mu):
    return np.sqrt(beta/np.pi)*np.exp(-beta*((x - mu)**2))

matplotlib.rcParams['agg.path.chunksize'] = 200000
res = np.loadtxt('y_results.txt', delimiter=',')

x_start = 100
x_stop = 10100
y_true = res[x_start:x_stop,:5]
y_pred = res[x_start:x_stop,5:10]

coeff = ['a','b','c','d','mu']
fig, ax = plt.subplots(2, 2, figsize=(20,20))
for i in range(len(coeff)):
    plt.suptitle("Pred vs True - "+coeff[i], fontsize=22)
    ax[0,0].hist(y_pred[:,i])
    ax[0,1].scatter(y_true[:,i], y_pred[:,i])
    ax[0,0].set_ylabel("y_pred", fontsize=18)
    ax[1,0].scatter(y_pred[:,i], y_true[:,i])
    ax[1,0].set_xlabel("y_pred", fontsize=18)
    ax[1,0].set_ylabel("y_true", fontsize=18)
    ax[1,1].set_xlabel("y_true", fontsize=18)
    ax[1,1].hist(y_true[:,i])
    plt.savefig("scatter_{}.jpg".format(coeff[i]))
    ax[0,0].cla(); ax[0,1].cla(); ax[1,0].cla(); ax[1,1].cla()

x_start = 300
x_stop = 500
x = np.arange(1,x_stop-x_start+1)
y_true = y_true[x_start:x_stop,:]
y_pred = y_pred[x_start:x_stop,:]
y_diff = np.abs(y_true-y_pred)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20,8))
for i in range(len(coeff)):
    ax1.plot(x, y_pred[:,i], 'o', label="pred")
    ax1.plot(x, y_true[:,i], 'o', label="true")
    ax2.plot(x, y_diff[:,i], label="diff")
    ax1.legend()
    ax1.set_ylabel(coeff[i])
    ax1.set_title("Coefficient estimation")
    ax2.set_ylabel('abs error')
    plt.savefig("plot_{}.jpg".format(coeff[i]))
    ax1.cla(); ax2.cla()

y_true_average = np.mean(res[:,:5], axis=0)
y_pred_average = np.mean(res[:,5:10], axis=0)

print("Pred average: ", y_pred_average)
print("True average: ", y_true_average)

xmin = -2.0
xmax = 2.0
dx = 0.01
x = np.arange(xmin, xmax, dx)

l_curve_pred = np.empty(x.shape)
l_curve_true = np.empty(x.shape)
r_curve_pred = np.empty(x.shape)
r_curve_true = np.empty(x.shape)
g_curve_pred = np.empty(x.shape)
g_curve_true = np.empty(x.shape)

for i in range(len(x)):
    l_curve_pred[i] = l(x[i],y_pred_average[0],y_pred_average[1])
    l_curve_true[i] = l(x[i],y_true_average[0],y_true_average[1])
    r_curve_pred[i] = r(x[i],y_pred_average[2],y_pred_average[3])
    r_curve_true[i] = r(x[i],y_true_average[2],y_true_average[3])
    g_curve_pred[i] = g(x[i],2.5,y_pred_average[4])
    g_curve_true[i] = g(x[i],2.5,y_true_average[4])

fig = plt.figure(figsize=(12,8))
plt.plot(x, l_curve_pred, label="l_curve_pred")
plt.plot(x, r_curve_pred, label="r_curve_pred")
plt.plot(x, g_curve_pred, label="g_curve_pred")
plt.plot(x, l_curve_true, label="l_curve_true")
plt.plot(x, r_curve_true, label="r_curve_true")
plt.plot(x, g_curve_true, label="g_curve_true")
plt.legend()
plt.savefig("average_plot.jpg")

