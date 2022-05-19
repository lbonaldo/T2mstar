import torch.nn as nn
import test_config as c

class FullyConnected(nn.Module):
    '''Fully connected tranformation, not reversible, but used below.'''

    def __init__(self, dims_in, dims_out, internal_size=c.hidden_layer_sizes, dropout=c.dropout_perc, batch_norm=c.batch_norm):
        super(FullyConnected, self).__init__()
        if not internal_size:
            internal_size = 2 * dims_out

        self.batch_norm = batch_norm

        self.d1 = nn.Dropout(p=dropout)
        self.d2 = nn.Dropout(p=dropout)
        self.d2b = nn.Dropout(p=dropout)

        self.fc1 = nn.Linear(dims_in, internal_size)
        self.fc2 = nn.Linear(internal_size, internal_size)
        self.fc2b = nn.Linear(internal_size, internal_size)
        self.fc3 = nn.Linear(internal_size, dims_out)

        self.nl1 = nn.LeakyReLU()
        self.nl2 = nn.LeakyReLU()
        self.nl2b = nn.LeakyReLU()

#        self.nl1 = nn.Tanh()
#        self.nl2 = nn.Tanh()
#        self.nl2b = nn.Tanh()

        if self.batch_norm:
            self.bn1 = nn.BatchNorm1d(internal_size)
            self.bn1.weight.data.fill_(1)
            self.bn2 = nn.BatchNorm1d(internal_size)
            self.bn2.weight.data.fill_(1)
            self.bn2b = nn.BatchNorm1d(internal_size)
            self.bn2b.weight.data.fill_(1)
        

    def forward(self, x):
        if self.batch_norm:
            out = self.nl1(self.d1(self.bn1(self.fc1(x))))
            out = self.nl2(self.d2(self.bn2(self.fc2(out))))
            out = self.nl2b(self.d2b(self.bn2b(self.fc2b(out))))
            return self.fc3(out)

        else:
            out = self.nl1(self.d1(self.fc1(x)))
            out = self.nl2(self.d2(self.fc2(out)))
            out = self.nl2b(self.d2b(self.fc2b(out)))
            return self.fc3(out)


def subnet_conv(c_in, c_out):
    return nn.Sequential(nn.Conv2d(c_in, 256,   3, padding=1), nn.ReLU(),
                        nn.Conv2d(256,  c_out, 3, padding=1))