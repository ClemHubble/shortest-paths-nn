import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
from torch_geometric.nn import GCNConv, GINConv, GATConv, GCN2Conv, TransformerConv
from torch.nn import ReLU, LeakyReLU, Sigmoid

# Baseline 0
def initialize_mlp(input, hidden, output, layers, batch_norm=False, activation='relu'):
    if layers == 1:
        hidden=output
    if activation == 'relu':
        func = nn.ReLU
    elif activation =='lrelu':
        func = nn.LeakyReLU
    elif activation=='sigmoid':
        func = nn.Sigmoid
    else:
        raise NameError('Not implemented')

    phi_layers= []
    phi_layers.append(nn.Linear(input, hidden))
    phi_layers.append(func())
    if batch_norm:
        phi_layers.append(nn.BatchNorm1d(input))
    for i in range(layers - 1):
        if i < layers - 2:
            phi_layers.append(nn.Linear(hidden, hidden))
            phi_layers.append(func())
            if batch_norm:
                phi_layers.append(nn.BatchNorm1d(hidden))
        else:
            phi_layers.append(nn.Linear(hidden, output))

    phi = nn.Sequential(*phi_layers)
    return phi

class MLPBaseline0(nn.Module):
    def __init__(self, siamese: nn.Module, final: nn.Module):
        super(MLPBaseline0, self).__init__()
        self.siamese = siamese
        self.final = final
    
    def init_weights(self):
        for m in self.siamese:
            if isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight)
                torch.nn.init.normal_(m.bias)
    
    def forward(self, input1, input2):
        out1 = self.siamese(input1)
        out2 = self.siamese(input2)

        sum_embd = out1 + out2
        return self.final(sum_embd)


class GINLayer(nn.Module):
    def __init__(self, input=3, output=20, eps=0.001):
        super(GINLayer, self).__init__()
        self.nn = nn.Sequential(nn.Linear(input, output), 
                                nn.ReLU(),
                                nn.Linear(output, output))
        self.layer = GINConv(self.nn, eps=0.001)

    def forward(self, x, edge_index):
        output = self.layer(x, edge_index)
        return output


class GNNModel(nn.Module):
    def __init__(self, input=3, output=20, hidden=20, layers=2, 
                 layer_type='GCNConv', activation='LeakyReLU', **kwargs):
        super(GNNModel, self).__init__()
        torch.manual_seed(1234567)
        # Initialize the first layer
        graph_layer = globals()[layer_type]
        self.initial = graph_layer(input, hidden)
        
        # Initialize the subsequent layers
        self.module_list = nn.ModuleList([graph_layer(hidden, hidden) for _ in range(layers - 1)])
        
        # Output layer
        self.output = graph_layer(hidden, output)

        # activation function
        self.activation = globals()[activation]()

    def forward(self, x, edge_index):
        x = self.initial(x, edge_index)
        x = self.activation(x)
        for layer in self.module_list:
            x = layer(x, edge_index)
            x = self.activation(x)
        x = self.output(x, edge_index)
        return x


class GCN2Model(torch.nn.Module):
    def __init__(self,alpha=0.1, theta=0.5, input=3, hidden=20, output=20,layers=2, 
                 shared_weights=True, dropout=0.0, **kwargs):
        super().__init__()

        self.lins = torch.nn.ModuleList()
        self.lins.append(nn.Linear(input, hidden))
        self.lins.append(nn.Linear(hidden, output))

        self.convs = torch.nn.ModuleList()
        for layer in range(layers):
            self.convs.append(
                GCN2Conv(hidden, alpha, theta, layer + 1,
                         shared_weights, normalize=False))

        self.dropout = dropout

    def forward(self, x, adj_t):
        x = F.dropout(x, self.dropout, training=self.training)
        x = x_0 = self.lins[0](x).relu()

        for conv in self.convs:
            x = F.dropout(x, self.dropout, training=self.training)
            x = conv(x, x_0, adj_t)
            x = x.relu()

        x = F.dropout(x, self.dropout, training=self.training)
        x = self.lins[1](x)

        return x.log_softmax(dim=-1)

class GraphTransformer(torch.nn.Module):
    def __init__(self, input, hidden, output, layers,
                 heads=2, dropout=0.3, **kwargs):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        for i in range(1, layers + 1):
            if i < layers:
                out_channels = hidden // heads
                concat = True
            else:
                out_channels = output
                concat = False
            conv = TransformerConv(input, out_channels, heads,
                                   concat=concat, beta=True, dropout=dropout)
            self.convs.append(conv)
            input = hidden

            if i < layers:
                self.norms.append(torch.nn.LayerNorm(hidden))

    def forward(self, x, edge_index):
        for conv, norm in zip(self.convs, self.norms):
            x = norm(conv(x, edge_index)).relu()
        return self.convs[-1](x, edge_index)
