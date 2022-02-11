import torch
import FrEIA.framework
import FrEIA.modules

import config as c
from subnets import FullyConnected

gamma = (c.final_decay) ** (1. / c.n_epochs)

nodes = [FrEIA.framework.InputNode(c.ndim_x + c.ndim_pad_x, name='input')]
torch.manual_seed(c.batch_size)

for i in range(c.N_blocks):
    nodes.append(FrEIA.framework.Node([nodes[-1].out0], FrEIA.modules.RNVPCouplingBlock,
                                      {'subnet_constructor': FullyConnected,
                                       'clamp': c.exponent_clamping,
                                       },
                                      name='coupling_{}'.format(i)))

    if c.use_permutation:
        nodes.append(FrEIA.framework.Node([nodes[-1].out0], FrEIA.modules.PermuteRandom,
                                          {'seed': i},
                                          name='permute_{}'.format(i)))

nodes.append(FrEIA.framework.OutputNode([nodes[-1].out0], name='output'))

model = FrEIA.framework.GraphINN(nodes, verbose=c.verbose_construction)
model.to(c.device)

params_trainable = list(filter(lambda p: p.requires_grad, model.parameters()))
for p in params_trainable:
    p.data = c.init_scale * torch.randn(p.data.shape).to(c.device)

gamma = (c.final_decay) ** (1. / c.n_epochs)
optim = torch.optim.Adam(params_trainable, lr=c.lr_init, betas=c.adam_betas, eps=1e-6, weight_decay=c.l2_weight_reg)
weight_scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=1, gamma=gamma)


def optim_step():
    optim.step()
    optim.zero_grad()


def scheduler_step():
    weight_scheduler.step()


def save(name):
    torch.save({'opt': optim.state_dict(),
                'net': model.state_dict()}, name)


def load(name):
    state_dicts = torch.load(name, map_location=torch.device(c.device))
    model.load_state_dict(state_dicts['net'])
    try:
        optim.load_state_dict(state_dicts['opt'])
    except ValueError:
        c.log('Cannot load optimizer for some reason or other', c.logfile)
        print('Cannot load optimizer for some reason or other')
