import torch
import torch.nn as nn

import FrEIA.framework
import FrEIA.modules


class FullyConnected(nn.Module):
    '''Fully connected tranformation, not reversible, but used below.'''

    def __init__(self, dims_in, dims_out, internal_size=128, dropout=0.25):
        super(FullyConnected, self).__init__()

        self.hidden1 = FCHiddenLayer(dims_in,internal_size,dropout, batch_norm=False)
        self.hidden2 = FCHiddenLayer(internal_size,internal_size,dropout, batch_norm=False)
        self.hidden3 = FCHiddenLayer(internal_size,internal_size,dropout, batch_norm=False)
        self.last = FCLastLayer(internal_size,dims_out,dropout, batch_norm=False)

    def forward(self, x):
        x = self.hidden3(self.hidden2(self.hidden1(x)))
        return self.last(x)


class cINN(nn.Module):
    '''Conditional Invertible NN class'''

    def __init__(self, ndim_x, ndim_pad_x, ndim_y, N_blocks, exponent_clamping=1.,use_permutation=False):
        super(cINN, self).__init__()

        self.ndim_x = ndim_x
        self.ndim_pad_x = ndim_pad_x
        self.ndim_y = ndim_y
        self.N_blocks = N_blocks

        self.nodes = None
        self.model = None

        self.dims_in = self.ndim_x + self.ndim_pad_x

        self.cond_node = FrEIA.framework.ConditionNode(self.ndim_y)

        # construct the INN
        self.nodes = [FrEIA.framework.InputNode(self.dims_in, name='input')]

        for i in range(self.N_blocks):
            self.nodes.append(FrEIA.framework.Node([self.nodes[-1].out0], 
                                                    FrEIA.modules.GLOWCouplingBlock,
                                                    {'subnet_constructor': FullyConnected,
                                                    'clamp': exponent_clamping},
                                                    conditions=self.cond_node,
                                                    name='coupling_{}'.format(i)))

            if use_permutation:
                self.nodes.append(FrEIA.framework.Node([self.nodes[-1].out0], 
                                                        FrEIA.modules.PermuteRandom,
                                                        {'seed': i},
                                                        name='permute_{}'.format(i)))

        self.nodes.append(FrEIA.framework.OutputNode([self.nodes[-1].out0], name='output'))
        self.nodes.append(self.cond_node)

        self.model = FrEIA.framework.GraphINN(self.nodes)

    def forward(self, x, y, rev=False):
        return self.model(x, y, rev=rev)


class ModelWrapper(object):
    '''Wrapper to a Torch model, Members are (at least): model, optimizer, scheduler.'''

    def __init__(self, model, loaders, hyparams, device, visualizer, lossparams=None,others=None):
        super(ModelWrapper, self).__init__()

        self.model = model
        self.loaders = loaders
        self.hyparams = hyparams
        self.device = device
        self.visualizer = visualizer
        self.lossparams = lossparams
        self.others = others

        self.model.to(self.device)

        params = list(filter(lambda p: p.requires_grad, self.model.parameters()))
        for p in params:
            p.data = hyparams.init_sigma * torch.randn(p.data.shape).to(device)

        self.optim = torch.optim.Adam(params, lr=hyparams.lr_init, betas=hyparams.adam_betas, eps=1e-6, weight_decay=hyparams.l2_weight_reg)

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optim, mode='min', factor=0.1, patience=hyparams.patience, threshold=0.001, threshold_mode='rel', cooldown=0, min_lr=1e-06)


class FCHiddenLayer(nn.Module):
    '''Base unit for a hidden layer of a FC network.'''

    def __init__(self, dims_in, dims_out, dropout, batch_norm=True):
        super(FCHiddenLayer, self).__init__()

        self.batch_norm = batch_norm

        self.layer  = nn.Linear(dims_in, dims_out)
        self.bn     = nn.BatchNorm1d(dims_out)
        self.drop   = nn.Dropout(p=dropout)
        self.act    = nn.LeakyReLU()

    def forward(self, x):
        if self.batch_norm:
            return self.act(self.drop(self.bn(self.layer(x))))
        else:
            return self.act(self.drop(self.layer(x)))

class FCLastLayer(nn.Module):
    '''Base unit for the last layer of a FC network.'''

    def __init__(self, dims_in, dims_out, dropout, batch_norm=True):
        super(FCLastLayer, self).__init__()

        self.batch_norm = batch_norm

        self.layer  = nn.Linear(dims_in, dims_out)
        self.bn     = nn.BatchNorm1d(dims_out)
        self.drop   = nn.Dropout(p=dropout)

    def forward(self, x):
        if self.batch_norm:
            return self.drop(self.bn(self.layer(x)))
        else:
            return self.drop(self.layer(x))


class FCSeebeck(nn.Module):
    '''Fully connected network to learn the Fermi level from the Seebeck.'''

    def __init__(self, dims_in, dims_out, internal_size, dropout, device="cpu"):
        super(FCSeebeck, self).__init__()

        assert len(internal_size)+1 == len(dropout), "Number of hidden layers and dropout vector must have same length"

        n_hidden_layers = len(internal_size)
        
        self.layers = [FCHiddenLayer(dims_in,internal_size[0],dropout[0]).to(device)]  # first layer
        for i in range(1,n_hidden_layers):
            self.layers.append(FCHiddenLayer(internal_size[i-1],internal_size[i],dropout[i]).to(device))
        self.layers.append(FCLastLayer(internal_size[n_hidden_layers-1],dims_out,dropout[n_hidden_layers]).to(device))   # last layer

        self.module_list = nn.ModuleList(self.layers)

    def forward(self, x):
        for layer in self.module_list:
            x = layer(x)
        return x


