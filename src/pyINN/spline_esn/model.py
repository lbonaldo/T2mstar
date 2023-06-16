import torch
import FrEIA.framework
import FrEIA.modules

import config as c
from subnets import FullyConnected


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

model_e = FrEIA.framework.GraphINN(nodes, verbose=c.verbose_construction)
model_e.to(c.device)
model_s = FrEIA.framework.GraphINN(nodes, verbose=c.verbose_construction)
model_s.to(c.device)
model_n = FrEIA.framework.GraphINN(nodes, verbose=c.verbose_construction)
model_n.to(c.device)

params_e = list(filter(lambda p: p.requires_grad, model_e.parameters()))
for p in params_e:
    p.data = c.init_scale * torch.randn(p.data.shape).to(c.device)    
optim_e = torch.optim.Adam(params_e, lr=c.lr_init, betas=c.adam_betas, eps=1e-6, weight_decay=c.l2_weight_reg)
scheduler_e = torch.optim.lr_scheduler.ReduceLROnPlateau(optim_e, mode='min', factor=0.1, patience=c.patience, threshold=0.001, threshold_mode='rel', cooldown=0, min_lr=1e-06, eps=1e-08)


params_s = list(filter(lambda p: p.requires_grad, model_s.parameters()))
for p in params_s:
    p.data = c.init_scale * torch.randn(p.data.shape).to(c.device)
optim_s = torch.optim.Adam(params_s, lr=c.lr_init, betas=c.adam_betas, eps=1e-6, weight_decay=c.l2_weight_reg)
scheduler_s = torch.optim.lr_scheduler.ReduceLROnPlateau(optim_s, mode='min', factor=0.1, patience=c.patience, threshold=0.001, threshold_mode='rel', cooldown=0, min_lr=1e-06, eps=1e-08)

params_n = list(filter(lambda p: p.requires_grad, model_n.parameters()))
for p in params_n:
    p.data = c.init_scale * torch.randn(p.data.shape).to(c.device)
optim_n = torch.optim.Adam(params_n, lr=c.lr_init, betas=c.adam_betas, eps=1e-6, weight_decay=c.l2_weight_reg)
scheduler_n = torch.optim.lr_scheduler.ReduceLROnPlateau(optim_n, mode='min', factor=0.1, patience=c.patience, threshold=0.001, threshold_mode='rel', cooldown=0, min_lr=1e-06, eps=1e-08)

models = (model_e,model_s,model_n)
params_trainables = (params_e,params_s,params_n) 
optims = (optim_e,optim_s,optim_n)
schedulers = (scheduler_e,scheduler_s,scheduler_n)


def optim_step():
    for optim in optims:
        optim.step()
        optim.zero_grad()


def scheduler_step(val_loss):
    for scheduler in schedulers:
        scheduler.step(val_loss)


def save(name):
    torch.save(({'opt_e': optim_e.state_dict(), 'net_e': model_e.state_dict()},
        {'opt_s': optim_s.state_dict(), 'net_s': model_s.state_dict()},
        {'opt_n': optim_n.state_dict(), 'net_n': model_n.state_dict()}), name)


def load(name):
    state_dicts = torch.load(name, map_location=torch.device(c.device))
    model_e.load_state_dict(state_dicts[0]['net_e'])
    model_s.load_state_dict(state_dicts[1]['net_s'])
    model_n.load_state_dict(state_dicts[2]['net_n'])
    try:
        optim_e.load_state_dict(state_dicts[0]['opt_e'])
        optim_s.load_state_dict(state_dicts[1]['opt_s'])
        optim_n.load_state_dict(state_dicts[2]['opt_n'])
    except ValueError:
        c.log('Cannot load optimizer for some reason or other', c.logfile)
        print('Cannot load optimizer for some reason or other')
