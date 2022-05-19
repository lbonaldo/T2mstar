import numpy as np
import matplotlib.pyplot as plt

import config as c

def print_config():
    config_str = ""
    config_str += "="*80 + "\n"
    config_str += "Config options:\n\n"

    listname_configscope = dir(c)
    for s in ['torch', 'train_loader', 'val_loader', "os", "sys", "date", "datetime", "create_testfolder", "log", "test_config", "export_text_config"]:
        listname_configscope.remove(s)

    for v in listname_configscope:
        if v[0]=='_': continue
        s=eval('c.%s'%(v))
        config_str += "  {:25}\t{}\n".format(v,s)

    config_str += "="*80 + "\n"
    
    c.log(config_str, c.logfile)
    print(config_str)


class Visualizer:
    def __init__(self, loss_labels, lr_labels):
        loss_labels_train, loss_labels_val = loss_labels
        self.n_losses = len(loss_labels_train)
        self.loss_labels_train = loss_labels_train
        self.loss_labels_val = loss_labels_val
        self.lr_labels = lr_labels    # tuple of lr
        self.counter = 0

        self.header = 'Epoch'
        for l in self.lr_labels:
            self.header += " "+l
        for l in loss_labels_train:
            self.header += " "+l
        for l in loss_labels_val:
            self.header += " "+l
        
        self.n_plots = 3
        self.figsize = (10,10)

    def print_losses(self, lr, losses):
        if self.header:
            c.log(self.header, c.logfile)
            print(self.header)
            self.header = None

        # print('\r', '    '*20, end='')
        line = '\r%6.3i' % (self.counter+1)
        for l in lr:
            line += ' %14.4f' % (l)
        for l in losses:
            line += ' %14.4f' % (l)

        c.log(line+"\n", c.logfile)
        print(line)
        self.counter += 1
    
    def plot_losses(self, train_losses, eval_losses, logscale=False):
        self.fig, self.axes = plt.subplots(self.n_plots, 1, figsize=self.figsize)
        x = np.arange(1, c.pre_low_lr+c.n_epochs)
        # tensor_name = ['e','s','n']
        plots_indices = [[0,1,2],[3,4,5],[6,7,8,9,10,11],[12]]
        # print(train_losses) 
        # print(eval_losses)
        for i,plot_indices in enumerate(plots_indices):
            for idx in plot_indices:
                y_t = np.array(train_losses[:,idx])
                y_e = np.array(eval_losses[:,idx])
                if logscale: 
                    y_t = np.log10(y_t)
                    y_e = np.log10(y_e)
                plt.plot(x, y_t, label=self.loss_labels_train[idx])
                plt.plot(x, y_e, label=self.loss_labels_val[idx])
            plt.xlabel('epoch')
            plt.ylabel('Loss')
            plt.savefig("loss_plot_#{}.png".format(i+1))

    def update_hist(self, *args):
        pass

    def update_cov(self, data):
        pass

visualizer = None

def restart():
    global visualizer
    loss_labels_train = []

    lr_labels = ['lr_e', 'lr_s', 'lr_n']

    if c.train_max_likelihood:
        loss_labels_train.extend(['L_ML_e','L_ML_s','L_ML_n'])
    if c.train_forward_mmd:
        loss_labels_train.extend(['L_fit_e','L_fit_s','L_fit_n','L_mmd_fwd_e','L_mmd_fwd_s','L_mmd_fwd_n'])
    if c.train_backward_mmd:
        loss_labels_train.extend(['L_mmd_back_e','L_mmd_back_s','L_mmd_back_n'])
    if c.train_reconstruction:
        loss_labels_train.append('L_reconst')

    loss_labels_val = [l + '(val)' for l in loss_labels_train]

    visualizer = Visualizer([loss_labels_train,loss_labels_val], lr_labels)


def show_loss(lr, losses):
    visualizer.print_losses(lr, losses)

def plot_loss(train_losses, eval_losses, logscale=False):
    visualizer.plot_losses(train_losses, eval_losses, logscale)

def show_hist(data):
    visualizer.update_hist(data.data.cpu())

def show_cov(data):
    visualizer.update_cov(data.data.cpu())

def close():
    visualizer.close()

