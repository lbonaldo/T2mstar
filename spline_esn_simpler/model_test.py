import torch
import FrEIA.framework
import FrEIA.modules

import test_config as c
from subnets_test import FullyConnected, subnet_conv


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

model_e = FrEIA.framework.GraphINN(nodes)
model_e.to(c.device)
model_s = FrEIA.framework.GraphINN(nodes)
model_s.to(c.device)
model_n = FrEIA.framework.GraphINN(nodes)
model_n.to(c.device)

models = (model_e,model_s,model_n)

def load(name):
    state_dicts = torch.load(name, map_location=torch.device(c.device))
    model_e.load_state_dict(state_dicts[0]['net_e'])
    model_s.load_state_dict(state_dicts[1]['net_s'])
    model_n.load_state_dict(state_dicts[2]['net_n'])
