import numpy as np
import matplotlib
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.sans-serif'] = ['Verdana']
plt.rcParams['axes.linewidth'] = 2
plt.rcParams['axes.titlepad'] = 11
plt.rcParams['axes.spines.bottom'] = True
plt.rcParams['axes.spines.left'] = True
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['xtick.major.size'] = 2
plt.rcParams['ytick.major.size'] = 2
plt.rcParams["legend.loc"] = "upper left"
plt.rcParams["lines.linewidth"] = 2
plt.rcParams['lines.markersize'] = 7
plt.rcParams['font.size'] = 13
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 9
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11
plt.rcParams['legend.fontsize'] = 9

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

test_path = "results/test_14/"

x_pred = np.load(test_path+"pred.npy")
x_true = np.load(test_path+"true.npy")

x = np.arange(1,3,0.000001)
fig=plt.figure(figsize=(10,10))
plt.title("INN prediction - true value", fontsize="x-large")
plt.plot(x, x_true[:-1,0], "-.", color='#1f77b4', label="True")
plt.plot(x, x_pred[:-1,0], color='#ff7f0e', label="Pred")
plt.plot(x, x_true[:-1,1], "-.", color='#1f77b4')
plt.plot(x, x_pred[:-1,1], color='#ff7f0e')
plt.plot(x, x_true[:-1,2], "-.", color='#1f77b4')
plt.plot(x, x_pred[:-1,2], color='#ff7f0e')
plt.plot(x, x_true[:-1,3], "-.", color='#1f77b4')
plt.plot(x, x_pred[:-1,3], color='#ff7f0e')
plt.plot(x, x_true[:-1,4], "-.", color='#1f77b4')
plt.plot(x, x_pred[:-1,4], color='green', label="recon")
plt.plot(x, x_true[:-1,5], "-.", color='#1f77b4')
plt.plot(x, x_pred[:-1,5], color='green', label="recon")
plt.legend(fontsize="large")
plt.grid()
plt.savefig(test_path+"fig.png")