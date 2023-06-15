import os
import numpy as np
import matplotlib.pyplot as plt

from utils import log


class Visualizer(object):
    def __init__(self, lr_labels, loss_labels_train, loss_labels_val, config_file_params,config_file_params_name,train_path):
        self.lr_labels = lr_labels      # list of lr
        self.loss_labels_train = loss_labels_train  # list of loss labels
        self.loss_labels_val = loss_labels_val      # list of loss labels
        self.config_file_params = config_file_params
        self.config_file_params_name = config_file_params_name
        self.train_path = train_path
        self.logfile = os.path.join(train_path, "out.txt")
        
        self.counter = 0
        self.header = 'Epoch'
        for l in self.lr_labels:
            self.header += "\t"+l
        for l in self.loss_labels_train:
            self.header += "\t"+l
        for l in self.loss_labels_val:
            self.header += "\t"+l

    def print_config(self, model_name):
        string_max = len(self.logfile) + 10
        config_str = ""
        config_str += "="*string_max + "\n"
        config_str += "Config options for "+model_name+":\n\n"
    
        # print "var_name = var_value" to stdout and file
        for par_names,par in zip(self.config_file_params_name,self.config_file_params):
            par_name = "" 
            par_names_len = len(par_names)
            if par_names_len == 1:
                par_name = par_names[0]
            else:
                for i in range(par_names_len-1):
                    par_name += par_names[i] + ", "
                par_name += par_names[-1]

            config_str += "{} = {}\n".format(par_name,par)

        config_str += "="*string_max + "\n"
        
        log(config_str, self.logfile)
        print(config_str)

    def print_single_loss(self, lr, train_losses, val_losses):
        if self.header:
            log(self.header, self.logfile)
            print(self.header)
            self.header = None

        line = '\r%5.3i' % (self.counter+1)          
        line += '\t{:.5f}'.format(lr)
        line += '\t{:.5f}'.format(train_losses)
        line += '\t{:.5f}'.format(val_losses)

        log(line+"\n", self.logfile)
        print(line)
        self.counter += 1


    def print_losses(self, lr, train_losses, val_losses):
        if self.header:
            log(self.header, self.logfile)
            print(self.header)
            self.header = None

        line = '\r%5.3i' % (self.counter+1)

        if isinstance(lr, float):
            lr = [lr]            
        for l in lr:
            line += '\t{:.5f}'.format(l)
        for l in train_losses:
            line += '\t{:.5f}'.format(l)
        for l in val_losses:
            line += '\t{:.5f}'.format(l)

        log(line+"\n", self.logfile)
        print(line)
        self.counter += 1

    def plot_single_loss(self, epochs, train_loss, eval_loss, logscale=False):
        self.fig = plt.figure(figsize=(10,8))
        x = np.arange(epochs)

        y_t = np.array(train_loss)
        y_e = np.array(eval_loss)
        if logscale: 
            y_t = np.log10(y_t)
            y_e = np.log10(y_e)

        plt.plot(x, y_t, label="train")
        plt.plot(x, y_e, label="val")
        
        plt.xlabel('epoch')
        plt.ylabel('Loss')

        plt.legend(loc="upper right")
        
        plt.savefig(os.path.join(self.train_path, "singleloss_plot.png"))


    def plot_losses(self, epochs, train_losses, val_losses, logscale=False):
        self.fig = plt.figure(figsize=(10,8))
        x = np.arange(epochs)

        for i in range(train_losses.shape[1]):
            y_t = np.array(train_losses[:,i])
            y_v = np.array(val_losses[:,i])
            if logscale: 
                y_t = np.log10(y_t)
                y_v = np.log10(y_v)

            plt.plot(x, y_t, label=self.loss_labels_train[i])
            plt.plot(x, y_v, label=self.loss_labels_val[i])

        plt.xlabel('epoch')
        plt.ylabel('Loss')

        plt.legend(loc="upper right")
        
        plt.savefig(os.path.join(self.train_path, "losses_plot.png"))

