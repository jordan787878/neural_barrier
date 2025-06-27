import torch
import torch.nn as nn
import torch.nn.functional as F


def load_trained_model(net, path):
    print("load model from: "+ path)
    checkpoint = torch.load(path)
    net.load_state_dict(checkpoint['model_state_dict'])
    return net


class BaseNet(nn.Module):
    def __init__(self, const, neurons=128, scale=1.0): 
        super(BaseNet, self).__init__()
        self.scale = scale
        self.constants = const
        self.hidden_layer1 = (nn.Linear(const.X_DIM, neurons))
        self.hidden_layer2 = (nn.Linear(neurons, neurons))
        self.output_layer =  (nn.Linear(neurons,1))
    def forward(self, x):
        layer1_out = F.tanh((self.hidden_layer1(x/100.0))) # divide by 100.0 to normalize inputs to [-1,1]
        layer2_out = F.tanh((self.hidden_layer2(layer1_out)))
        output = self.output_layer(layer2_out)
        return output
