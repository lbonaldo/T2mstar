import torch
import torch.nn as nn


class BaseFCC(nn.Module):
    def __init__(self):
        super(BaseFCC, self).__init__()

    def save(self,filename):
        torch.save(self.state_dict(), filename)

    def load(self,filename):
        self.load_state_dict(torch.load(filename))


class FCCblock(nn.Module):
    def __init__(self, dim_in, dim_out, dropout):
        super(FCCblock, self).__init__()

        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dropout = dropout

        self.block = nn.Sequential(
            nn.Linear(dim_in, dim_out),
            nn.BatchNorm1d(dim_out),
            nn.Dropout(p=self.dropout),
            nn.LeakyReLU(),
            #nn.Sigmoid()
        )

    def forward(self, x):
        return self.block(x)


class TensorSubnet(nn.Module):
    '''Fully connected network with 3 subnets one for each e,n and s.'''

    def __init__(self, dims_in, dims_out, dim_layers, dropout_perc):
        super(TensorSubnet, self).__init__()
       
        self.dims_in = dims_in
        self.dims_out = dims_out
        self.dim_layers = dim_layers
        self.dropout_perc = dropout_perc

        self.blocklist = [FCCblock(dims_in,dim_layers[0],dropout_perc[0])]
        for i in range(1,len(self.dim_layers)):
            self.blocklist.append(FCCblock(self.dim_layers[i-1],self.dim_layers[i],dropout_perc[i]))        
        self.blocklist.append(FCCblock(self.dim_layers[-1],self.dims_out,dropout_perc[-1]))        

        self.net = nn.ModuleList(self.blocklist)

    def forward(self,x):
        for l in self.net:
            x = l(x)
        return x


class FCC3(BaseFCC):
    '''Fully connected network with 3 subnets one for each e,n and s.'''
    
    def __init__(self, dims_in, dims_out, dim_layers, dropout_perc):
        super(FCC3, self).__init__()

        self.dims_in = dims_in
        self.dims_out = dims_out
        self.dim_layers = dim_layers
        self.dropout_perc = dropout_perc
        
        self.enet = TensorSubnet(self.dims_in[0],self.dims_out,self.dim_layers[0],self.dropout_perc)
        self.nnet = TensorSubnet(self.dims_in[1],self.dims_out,self.dim_layers[1],self.dropout_perc)
        self.snet = TensorSubnet(self.dims_in[2],self.dims_out,self.dim_layers[2],self.dropout_perc)

        self.modelmix = FCCblock(3*self.dims_out,self.dims_out,0.0)

    def forward(self, e_x, n_x, s_x):
        return self.modelmix(torch.cat((self.enet(e_x), self.nnet(n_x), self.snet(s_x)), 1))
        # return torch.div(torch.add(torch.add(self.enet(e_x), self.nnet(n_x)), self.snet(s_x)),3)


class FCC2(BaseFCC):
    '''Fully connected network with 2 subnets one for e,n and another one for s.'''

    def __init__(self, dims_in, dims_out, dim_layers, dropout_perc):
        super(FCC2, self).__init__()
       
        self.dims_in = dims_in
        self.dims_out = dims_out
        self.dim_layers = dim_layers
        self.dropout_perc = dropout_perc    

        self.ennet = TensorSubnet(self.dims_in[0],self.dims_out,self.dim_layers[0],self.dropout_perc)
        self.snet  = TensorSubnet(self.dims_in[1],self.dims_out,self.dim_layers[1],self.dropout_perc)

        self.modelmix = FCCblock(2*self.dims_out,self.dims_out,0.1)

    def forward(self, e_x, n_x, s_x):
        return self.modelmix(torch.cat((self.ennet(torch.cat((e_x,n_x),1)), self.snet(s_x)), 1))


class FCC1(BaseFCC):
    '''Fully connected single network for all tensors.'''

    def __init__(self, dims_in, dims_out, dim_layers, dropout_perc):
        super(FCC1, self).__init__()
       
        self.dims_in = dims_in
        self.dims_out = dims_out
        self.dim_layers = dim_layers
        self.dropout_perc = dropout_perc    

        self.ensnet = TensorSubnet(self.dims_in,self.dims_out,self.dim_layers,self.dropout_perc)
        
    def forward(self, e_x, n_x, s_x):
        return self.ensnet(torch.cat((e_x,n_x,s_x),1))

