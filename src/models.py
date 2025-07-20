import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def np_sigmoid(x):  
    return 1 / (1 + np.exp(-x))


def np_tanh(x):
    """
    Elementwise hyperbolic tangent using NumPy.
    """
    exp_pos = np.exp(x)
    exp_neg = np.exp(-x)
    return (exp_pos - exp_neg) / (exp_pos + exp_neg)


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
    

class IBPNet(nn.Module):
    def __init__(self, const, neurons=128, scale=1.0):
        super(IBPNet, self).__init__()
        self.scale = scale
        self.constants = const
        # Define layers in a ModuleList for flexible architecture
        self.Layers = nn.ModuleList([
            nn.Linear(self.constants.X_DIM, neurons),  # input layer
            nn.Linear(neurons, 1)                      # output layer
        ])
    def forward(self, x):
        # normalize inputs
        x_norm = x / 100.0
        # first hidden layer with tanh activation
        out = F.sigmoid(self.Layers[0](x_norm))
        # output layer (linear)
        output = self.Layers[-1](out)
        return output
    def bound_propagation_last_layer(self, x_bounds):
        """
        NOTE: bound propagation has to used the same activation fcn in the forward pass
        """
        x_bounds = x_bounds/100.0
        lowers = x_bounds[:, 0].tolist()
        uppers = x_bounds[:, 1].tolist()
        lower_bounds = [lowers]
        upper_bounds = [uppers]
        for i in range(len(self.Layers)):
            ub = []
            lb = []
            for j in range(len(self.Layers[i].weight)):
                w = self.Layers[i].weight[j].detach().numpy().copy()
                b = self.Layers[i].bias[j].item()
                if(i < len(self.Layers)-1):
                    u = np_sigmoid(b + np.sum(np.maximum(w * upper_bounds[-1], w * lower_bounds[-1])))
                    l = np_sigmoid(b + np.sum(np.minimum(w * upper_bounds[-1], w * lower_bounds[-1])))
                else:
                    u = (b + np.sum(np.maximum(w * upper_bounds[-1], w * lower_bounds[-1])))
                    l = (b + np.sum(np.minimum(w * upper_bounds[-1], w * lower_bounds[-1])))
                ub.append(u)
                lb.append(l)
            upper_bounds.append(ub)
            lower_bounds.append(lb)
        return upper_bounds[-1], lower_bounds[-1]
    def bound_propagation_hidden_layer(self, x_bounds):
        """
        NOTE: bound propagation has to used the same activation fcn in the forward pass
        """
        x_bounds = x_bounds/100.0
        lowers = x_bounds[:, 0].tolist()
        uppers = x_bounds[:, 1].tolist()
        lower_bounds = [lowers]
        upper_bounds = [uppers]
        for i in range(len(self.Layers)-1):
            ub = []
            lb = []
            for j in range(len(self.Layers[i].weight)):
                w = self.Layers[i].weight[j].detach().numpy().copy()
                b = self.Layers[i].bias[j].item()
                u = np_sigmoid(b + np.sum(np.maximum(w * upper_bounds[-1], w * lower_bounds[-1])))
                l = np_sigmoid(b + np.sum(np.minimum(w * upper_bounds[-1], w * lower_bounds[-1])))
                ub.append(u)
                lb.append(l)
            upper_bounds.append(ub)
            lower_bounds.append(lb)
        return upper_bounds[-1], lower_bounds[-1]
