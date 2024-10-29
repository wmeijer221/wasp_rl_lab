import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_size, out_size, layer_size, act_fun=torch.relu, squash=False):
        super(MLP, self).__init__()

        self.fc_0 = nn.Linear(in_size, layer_size)
        self.fc_1 = nn.Linear(layer_size, layer_size)
        self.fc_2 = nn.Linear(layer_size, layer_size)

        self.fc_out = nn.Linear(layer_size, out_size)
        self.act_fun = act_fun
        self.squash = squash

    def forward(self, inp):
        x = self.act_fun(self.fc_0(inp))
        x = self.act_fun(self.fc_1(x))
        x = self.act_fun(self.fc_2(x))

        x = self.fc_out(x)
        if self.squash:
            x = torch.tanh(x)

        return x