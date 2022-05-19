import os
import numpy as np
import matplotlib.pyplot as plt

import config as c

class Visualizer(object):
    def __init__(self, loss_labels, lr_labels):
        loss_labels_train, loss_labels_val = loss_labels
        self.loss_labels_train = loss_labels_train
        self.loss_labels_val = loss_labels_val
        self.lr_labels = lr_labels    # tuple of lr
        self.counter = 0

        self.header = 'Epoch'
        for l in self.lr_labels:
            self.header += "\t"+l
        for l in self.loss_labels_train:
            self.header += "\t\t"+l
        for l in self.loss_labels_val:
            self.header += "\t\t"+l

    def print_config(self):
        config_str = ""
        config_str += "="*120 + "\n"
        config_str += "Config options:\n\n"

        listname_configscope = dir(c)
        for s in ['torch', "os", "string", "random", "date", "datetime", "FCC1", "FCC2", "FCC3", "NNet", "create_expfolder", "log", "test_config", "export_text_config", "visualizer", "loss_labels_val", "lr_labels", "loss_labels_train","Visualizer"]:
            listname_configscope.remove(s)
        for s in ["dim_layers1","dim_layers2","dim_layers3"]:
            if s in listname_configscope:
                listname_configscope.remove(s)

        for v in listname_configscope:
            if v[0]=='_': continue
            s=eval('c.%s'%(v))
            config_str += "  {:25}\t{}\n".format(v,s)

        config_str += "="*120 + "\n"
        
        c.log(config_str, c.logfile)
        print(config_str)

    def print_losses(self, lr, losses):
        if self.header:
            c.log(self.header, c.logfile)
            print(self.header)
            self.header = None

        line = '\r%5.3i' % (self.counter+1)
        if isinstance(lr, float):
            lr = [lr]            
        for l in lr:
            line += '\t{:.5f}'.format(l)
        for l in losses:
            line += '\t\t{:.5f}'.format(l)

        c.log(line+"\n", c.logfile)
        print(line)
        self.counter += 1

    def plot_losses(self, train_losses, eval_losses, logscale=False):
        self.fig = plt.figure(figsize=(10,8))
        x = np.arange(c.n_epochs)

        y_t = np.array(train_losses)
        y_e = np.array(eval_losses)
        if logscale: 
            y_t = np.log10(y_t)
            y_e = np.log10(y_e)
        plt.plot(x, y_t, label=self.loss_labels_train)
        plt.plot(x, y_e, label=self.loss_labels_val)
        plt.xlabel('epoch')
        plt.ylabel('Loss')
        plt.savefig(os.path.join(c.train_path, "loss_plot.png"))

