import numpy as np
import matplotlib.pyplot as plt

import config as c

def print_config():
    config_str = ""
    config_str += "="*80 + "\n"
    config_str += "Config options:\n\n"

    listname_configscope = dir(c)
    for s in ['torch', 'train_loader', 'val_loader', "os", "sys", "date", "datetime", "create_testfolder", "log"]:
        listname_configscope.remove(s)

    for v in listname_configscope:
        if v[0]=='_': continue
        s=eval('c.%s'%(v))
        config_str += "  {:25}\t{}\n".format(v,s)

    config_str += "="*80 + "\n"
    
    c.log(config_str, c.logfile)
    print(config_str)


class Visualizer:
    def __init__(self, loss_labels):
            self.n_losses = len(loss_labels)
            self.loss_labels = loss_labels
            self.counter = 0

            self.header = 'Epoch '
            for l in loss_labels:
                self.header += ' %15s' % (l)
            
            self.n_plots = 3
            self.figsize = (10,10)

    def update_losses(self, losses, logscale=False):
        if self.header:
            c.log(self.header, c.logfile)
            print(self.header)
            self.header = None

        print('\r', '    '*20, end='')
        line = '\r%6.3i' % (self.counter)
        for l in losses:
            line += '  %14.4f' % (l)

        c.log(line+"\n", c.logfile)
        print(line)
        self.counter += 1
    
    def plot_losses(self, train_losses, eval_losses, logscale=False):
        self.fig, self.axes = plt.subplots(self.n_plots, 1, figsize=self.figsize)
        x = np.arange(1, c.pre_low_lr+c.n_epochs)
        tensor_name = ['e','s','n']
        l_indices = [[0,3,8],[1,3,9],[2,3,10]]
        for indices,name in zip(l_indices,tensor_name):
            for i,idx in enumerate(indices):
                y_t = np.array(train_losses[idx])
                y_e = np.array(eval_losses[idx])
                if logscale: 
                    y_t = np.log10(y_t)
                    y_e = np.log10(y_e)
                self.axes[i+1].plot(x, y_t, label="train")
                self.axes[i+1].plot(x, y_e, label="val")
                self.axes[i+1].set_title(self.loss_labels[idx])
                self.axes[i+1].set_xlabel('epoch')
                self.axes[i+1].set_ylabel('Loss')
                self.axes[i+1].grid(True)
            self.fig.tight_layout()
            plt.savefig(name+"_losses.png")

    def update_hist(self, *args):
        pass

    def update_cov(self, data):
        pass

visualizer = None

def restart():
    global visualizer
    loss_labels = []

    if c.train_max_likelihood:
        loss_labels.extend(['L_ML_e','L_ML_s','L_ML_n'])
    if c.train_forward_mmd:
        loss_labels.extend(['L_fit_e','L_fit_s','L_fit_n','L_mmd_fwd_e','L_mmd_fwd_s','L_mmd_fwd_n'])
    if c.train_backward_mmd:
        loss_labels.extend(['L_mmd_back_e','L_mmd_back_s','L_mmd_back_n'])
    if c.train_reconstruction:
        loss_labels.append('L_reconst')

    loss_labels += [l + '(val)' for l in loss_labels]

    visualizer = Visualizer(loss_labels)


def show_loss(losses, logscale=False):
    visualizer.update_losses(losses, logscale)

def plot_loss(train_losses, eval_losses, logscale=False):
    visualizer.plot_losses(train_losses, eval_losses, logscale)

def show_hist(data):
    visualizer.update_hist(data.data.cpu())

def show_cov(data):
    visualizer.update_cov(data.data.cpu())

def close():
    visualizer.close()

