import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
from torch_geometric.nn import GCNConv, GINConv, GATConv, GCN2Conv, TransformerConv, to_hetero, GeneralConv
from torch.nn import ReLU, LeakyReLU, Sigmoid
from torch_geometric.nn.conv import MessagePassing
 
# Baseline 0
def initialize_mlp(input, hidden, output, layers, batch_norm=False, activation='relu', **kwargs):
    if layers == 1:
        hidden=output
    if activation == 'relu':
        func = nn.ReLU
    elif activation =='lrelu':
        func = nn.LeakyReLU
    elif activation=='sigmoid':
        func = nn.Sigmoid
    elif activation =='softplus':
        func = nn.Softplus
    else:
        raise NameError('Not implemented')

    phi_layers= []
    phi_layers.append(nn.Linear(input, hidden))
    phi_layers.append(func())
    if batch_norm:
        phi_layers.append(nn.BatchNorm1d(input))
    for i in range(layers - 1):
        if i < layers - 2:
            phi_layers.append(nn.Dropout(p=0.30))
            phi_layers.append(nn.Linear(hidden, hidden))
            phi_layers.append(func())
            if batch_norm:
                phi_layers.append(nn.BatchNorm1d(hidden))
        else:
            phi_layers.append(nn.Linear(hidden, output))

    phi = nn.Sequential(*phi_layers)
    return phi

class MLPBaseline0(nn.Module):
    def __init__(self, siamese: nn.Module, final: nn.Module, max=False):
        super(MLPBaseline0, self).__init__()
        self.siamese = siamese
        self.final = final
        self.max = max
        print("max?", self.max)
    
    def init_weights(self):
        for m in self.siamese:
            if isinstance(m, nn.Linear):
                torch.nn.init.constant_(m.weight, 0.01)
                torch.nn.init.constant_(m.bias, 0.01)
        for m in self.final:
            if isinstance(m, nn.Linear):
                torch.nn.init.constant_(m.weight, 0.01)
                torch.nn.init.constant_(m.bias, 0.01)
    
    def forward(self, input1, input2):
        out1 = self.siamese(input1)
        out2 = self.siamese(input2)
        if not self.max:
            embd = out1 + out2
        else:
            embd = torch.max(out1, out2)
        return self.final(embd)

class MLPBaseline1(nn.Module):
    def __init__(self, mlp: nn.Module, max=True, aggr='max'):
        super(MLPBaseline1, self).__init__()
        self.mlp = mlp
        self.aggr=aggr
    
    def init_weights(self):
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                torch.nn.init.constant_(m.weight, 0.01)
                torch.nn.init.constant_(m.bias, 0.01)
    
    def forward(self, input1, input2):
        if self.aggr == 'max':
            embd = torch.max(input1, input2)
        elif self.aggr== 'sum':
            embd = input1 + input2
        elif self.aggr == 'min':
            embd = torch.min(input1, input2)
        elif self.aggr == 'combine':
            embd = torch.hstack((input1 + input2, torch.max(input1, input2), torch.min(input1, input2)))
 
        return self.mlp(embd)


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

class GeneralConvMaxAttention(nn.Module):
    def __init__(self, input=3, output=3):
        super(GeneralConvMaxAttention, self).__init__()
        self.layer = GeneralConv(in_channels=input, 
                                out_channels=output,
                                aggr='max', 
                                attention=True,
                                l2_normalize=False)
        
    def forward(self, x, edge_index):
        output = self.layer(x, edge_index)
        return output

class CNNLayer(nn.Module):
    def __init__(self, input=3, output=20, **kwargs):
        super(CNNLayer, self).__init__()
        self.layer = nn.Conv2d(in_channels=input, 
                                 out_channels=output, 
                                 kernel_size=1)
    
    def forward(self, x, edge_index):
        output = self.layer(x)
        return output

"""
Base GNN model - can take GATConv, GIN, GeneralConvMaxAttention
and CNNLayer. Technically, CNNLayer is not a graph layer but it
can work in this case because of how the terrain data is structured. 
"""
class GNNModel(nn.Module):
    def __init__(self, input=3, output=20, hidden=20, layers=2, 
                 layer_type='GATConv', activation='LeakyReLU', **kwargs):
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
        # x = data.x
        # edge_index = data.edge_index
        x = self.initial(x, edge_index)
        x = self.activation(x)
        for layer in self.module_list:
            x = layer(x, edge_index)
            x = self.activation(x)
        x = self.output(x, edge_index)
        return x

"""
GCN2 neural network as GCN2Conv does not work with the base GNN model. 
"""
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

    def forward(self, data):
        x = data.x
        adj_t = data.adj_t
        x = F.dropout(x, self.dropout, training=self.training)
        x = x_0 = self.lins[0](x).relu()

        for conv in self.convs:
            x = F.dropout(x, self.dropout, training=self.training)
            x = conv(x, x_0, adj_t)
            x = x.relu()

        x = F.dropout(x, self.dropout, training=self.training)
        x = self.lins[1](x)

        return x.log_softmax(dim=-1)

"""
Heterogeneous GNN with virtual nodes. 
TODO: This implementation may need to be changed as currently
the memory requirement for even a single virtual node is high for
a graph with 250,000 nodes. 
"""
class VNModel(torch.nn.Module):
    def __init__(self, metadata, **kwargs):
        super().__init__()
        self.gnn = GNNModel(**kwargs)
        self.gnn = to_hetero(self.gnn, metadata)
        
    def forward(self, data):
        x_dict = data.x_dict
        edge_index_dict = data.edge_index_dict
        output_x_dict = self.gnn(x_dict, edge_index_dict)
        return output_x_dict['real']



class GNN_VN_Model(torch.nn.Module):
    def __init__(self, input=3, output=20, hidden=20, layers=2, 
                 layer_type='GATConv', activation='LeakyReLU', **kwargs):
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

        ## List of MLPS to transform virtual node at every layer
        self.mlp_virtualnode_list = torch.nn.ModuleList()
        for layer in range(layers - 1):
            self.mlp_virtualnode_list.append(
                    torch.nn.Sequential(
                        torch.nn.Linear(input, hidden), 
                        torch.nn.LeakyReLU(negative_slope = 0.1),
                        torch.nn.Linear(hidden, output), 
                        torch.nn.LeakyReLU(negative_slope = 0.1)
                    )
            )
    def forward(self, data):
        self.initial()

"""
Graph transformer model. 
"""
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
