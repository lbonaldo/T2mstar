import numpy as np
import matplotlib.pyplot as plt

import config as c

def print_config():
    config_str = ""
    config_str += "="*80 + "\n"
    config_str += "Config options:\n\n"

    listname_configscope = dir(c)
    for s in ['torch', 'tr_loader', 'train_loader', 'tst_loader', 'test_loader', "os", "sys", "date", "datetime", "create_testfolder", "log"]:
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

    def update_losses(self, losses, logscale=False):
        if self.header:
            c.log(self.header, c.logfile)
            print(self.header)
            self.header = None

        #c.log('\r'+'    '*20, c.logfile)
        print('\r', '    '*20, end='')
        line = '\r%6.3i' % (self.counter)
        for l in losses:
            line += '  %14.4f' % (l)

        c.log(line+"\n", c.logfile)
        print(line)
        self.counter += 1

    def update_hist(self, *args):
        pass

    def update_cov(self, data):
        pass


class LiveVisualizer(Visualizer):
    def __init__(self, loss_labels):
        super().__init__(loss_labels)

        import visdom

        self.n_plots = 3
        self.figsize = (4,4)

        cfg = {"server": "gateway-03", "port": 8099}
        self.vis = visdom.Visdom('http://' + cfg["server"], port = cfg["port"])

        self.viz.close()

        self.l_plots = self.viz.line(X = np.zeros((1,self.n_losses)), 
                                     Y = np.zeros((1,self.n_losses)), 
                                     opts = {'legend':self.loss_labels})
        self.cov_mat = self.viz.heatmap(np.zeros((c.ndim_z, c.ndim_z)))

        self.fig, self.axes = plt.subplots(self.n_plots, self.n_plots, figsize=self.figsize)
        self.hist = self.viz.matplot(self.fig)

    def update_losses(self, losses, logscale=False):
        super().update_losses(losses)
        its = min(len(c.train_loader), c.n_its_per_epoch)
        y = np.array([losses])
        if logscale: 
            y = np.log10(y)

        self.viz.line(X = (self.counter-1) * its * np.ones((1,self.n_losses)),
                      Y = y,
                      opts   = {'legend':self.loss_labels},
                      win    = self.l_plots,
                      update = 'append')

    def update_hist(self, data):
        for i in range(self.n_plots):
            for j in range(self.n_plots):
                try:
                    self.axes[i,j].clear()
                    self.axes[i,j].hist(data[:, i*self.n_plots + j], bins=20, histtype='step')
                except IndexError:
                    break
                except ValueError:
                    continue

        self.fig.tight_layout()
        self.viz.matplot(self.fig, win=self.hist)

    def update_cov(self, data):
        self.viz.heatmap(np.cov(data.numpy(), rowvar=False), win=self.cov_mat)

    def close(self):
        self.viz.close(win=self.hist)
        self.viz.close(win=self.imgs)
        self.viz.close(win=self.l_plots)


visualizer = None


def restart():
    global visualizer
    loss_labels = []

    if c.train_max_likelihood:
        loss_labels.append('L_ML')
    if c.train_forward_mmd:
        loss_labels += ['L_fit', 'L_mmd_fwd']
    if c.train_backward_mmd:
        loss_labels.append('L_mmd_back')
    if c.train_reconstruction:
        loss_labels.append('L_reconst')

    loss_labels += [l + '(test)' for l in loss_labels]

    if c.interactive_visualization:
        visualizer = LiveVisualizer(loss_labels)
    else:
        visualizer = Visualizer(loss_labels)


def show_loss(losses, logscale=False):
    visualizer.update_losses(losses, logscale)


def show_hist(data):
    visualizer.update_hist(data.data.cpu())


def show_cov(data):
    visualizer.update_cov(data.data.cpu())


def close():
    visualizer.close()

