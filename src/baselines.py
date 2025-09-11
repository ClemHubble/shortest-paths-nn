import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
from torch_geometric.nn import GCNConv, GINConv, GATConv, GCN2Conv, TransformerConv, to_hetero, GeneralConv, GATv2Conv,GCN
from torch.nn import ReLU, LeakyReLU, Sigmoid, SiLU
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn import aggr 
from torch_geometric.nn.norm import LayerNorm

# import torch_geometric.graphgym.models.head  # noqa, register module
# import torch_geometric.graphgym.register as register
# from torch_geometric.graphgym.config import cfg
# from torch_geometric.graphgym.models.gnn import FeatureEncoder, GNNPreMP
# from torch_geometric.graphgym.register import register_network
from torch_geometric.nn import global_add_pool
import torch.nn.functional as F

from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, softmax
from torch_geometric.nn.inits import glorot, zeros

def get_coords(batch):
    graph = batch[3]
    source_coord = graph.pos[batch[0]]
    target_coord = graph.pos[batch[1]]
    return source_coord, target_coord

class LinearLayer(nn.Module):
    def __init__(self, input_dim, out_dim, add_norm=True):
        super().__init__()
        self.linear = nn.Linear(input_dim, out_dim)
        if add_norm:
            self.norm = nn.LayerNorm(out_dim)
    
    def forward(self, x):
        x = self.linear(x)
        if hasattr(self, "norm"):
            x = self.norm(x)
        return F.silu(x)

class NewMLP(nn.Module):
    def __init__(self, input, output, hidden, layers, add_norm=True, edge_dim=None):
        input_dim = input 
        out_dim = output
        hid_dim = hidden
        n_layers = layers
        super().__init__()
        self.layers = nn.ModuleList(
            [LinearLayer(input_dim, hid_dim, add_norm)] +
            [LinearLayer(hid_dim, hid_dim, add_norm) for _ in range(n_layers - 2)]
        )
        self.out_proj = nn.Linear(hid_dim, out_dim)
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.out_proj(x)

 
# Baseline 0
def initialize_mlp(input, 
                   hidden, 
                   output, 
                   layers, 
                   dropout=True, 
                   layer_norm=True, 
                   activation='relu', 
                   **kwargs):
    if layers == 1:
        hidden=output
    if activation == 'relu' or activation == 'ReLU':
        func = nn.ReLU
    elif activation =='lrelu' or activation == 'LeakyReLU':
        func = nn.LeakyReLU
    elif activation=='sigmoid':
        func = nn.Sigmoid
    elif activation =='softplus':
        func = nn.Softplus
    elif activation == 'silu' or activation == 'SiLU':
        func = nn.SiLU
    else:
        raise NameError('Not implemented')

    phi_layers= []
    phi_layers.append(nn.Linear(input, hidden))
    if layer_norm:
        phi_layers.append(nn.LayerNorm(hidden))
    if dropout:
        phi_layers.append(nn.Dropout(p=0.30))
    phi_layers.append(func())
    for i in range(layers - 1):
        if i < layers - 2:
            phi_layers.append(nn.Linear(hidden, hidden))
            if layer_norm:
                phi_layers.append(nn.LayerNorm(hidden))
            if dropout:
                phi_layers.append(nn.Dropout(p=0.30))
            phi_layers.append(func())
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
                # torch.nn.init.constant_(m.weight, 0.01)
                # torch.nn.init.constant_(m.bias, 0.01)
                torch.nn.init.normal_(m.weight, std=0.01)
                torch.nn.init.normal_(m.bias, std=0.01)
    
    def forward(self, input1, input2, vn_emb=None, batch=False):
        if self.aggr == 'max':
            embd = torch.max(input1, input2)
        elif self.aggr== 'sum':
            embd = input1 + input2
        elif self.aggr == 'min':
            embd = torch.min(input1, input2)
        elif self.aggr == 'combine':
            embd = torch.hstack((input1 + input2, torch.max(input1, input2), torch.min(input1, input2)))
        elif self.aggr == 'concat':
            embd = torch.hstack((input1, input2))
        elif self.aggr == 'sum+diff':
            embd = torch.hstack((input1 + input2, torch.abs(input1 - input2)))
        elif self.aggr == 'sum+diff+vn' and batch:
            embd = torch.hstack((input1 + input2, torch.abs(input1 - input2), vn_emb))
        elif self.aggr == 'sum+diff+vn':
            embd = torch.hstack((input1 + input2, torch.abs(input1 - input2), vn_emb.repeat(input1.size()[0], 1)))
        elif self.aggr == 'diff':
            embd = torch.abs(input1- input2)
        elif self.aggr == 'diff-no-abs':
            embd = input1 - input2
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


class GCNLayer(MessagePassing):
    def __init__(self, input, output, edge_dim, aggr='add'):
        super(GCNLayer, self).__init__(aggr=aggr)  # "add" aggregation
        self.layer = GCN(input, output, num_layers=1)
    
    def forward(self, x, edge_index, edge_attr):
        return self.layer(x, edge_index, edge_attr)

class GeneralConvMaxAttention(nn.Module):
    def __init__(self, input=3, output=3, edge_dim=None):
        super(GeneralConvMaxAttention, self).__init__()
        self.layer = GeneralConv(in_channels=input, 
                                out_channels=output,
                                aggr='max',
                                in_edge_channels=edge_dim,
                                attention=True,
                                l2_normalize=False)
        
        
    def forward(self, x, edge_index, edge_attr=None):
        general_conv_edge_attr = None
        if edge_attr != None:
            general_conv_edge_attr = edge_attr.unsqueeze(-1)
        output = self.layer(x, edge_index, edge_attr=general_conv_edge_attr)
        return output

class GeneralConvMinAttention(nn.Module):
    def __init__(self, input=3, output=3, edge_dim=None):
        super(GeneralConvMinAttention, self).__init__()
        self.aggregation = aggr.MultiAggregation([ 'min'])
        self.layer = GeneralConv(in_channels=input, 
                                 out_channels=output, 
                                 aggr='min', 
                                 in_edge_channels=edge_dim,
                                 attention=True, 
                                 l2_normalize=False)
    
    def forward(self, x, edge_index, edge_attr=None):
        general_conv_edge_attr = None
        if edge_attr != None:
            general_conv_edge_attr = edge_attr.unsqueeze(-1)
        output = self.layer(x, edge_index, edge_attr=general_conv_edge_attr)
        return output

class GeneralConvMultiAttention(nn.Module):
    def __init__(self, input=3, output=3, edge_dim=None):
        super(GeneralConvMultiAttention, self).__init__()
        self.aggregation = aggr.MultiAggregation(['mean', 'max', 'sum', 'min'])
        self.layer = GeneralConv(in_channels=input, 
                                 out_channels=output, 
                                 aggr='min', 
                                 in_edge_channels=edge_dim,
                                 attention=True, 
                                 l2_normalize=False)
    
    def forward(self, x, edge_index, edge_attr=None):
        general_conv_edge_attr = None
        if edge_attr != None:
            general_conv_edge_attr = edge_attr.unsqueeze(-1)
        output = self.layer(x, edge_index, edge_attr=general_conv_edge_attr)
        return output
    
class CNNLayer(nn.Module):
    def __init__(self, input=3, output=20, sz=25, **kwargs):
        super(CNNLayer, self).__init__()
        self.layer = nn.Conv2d(in_channels=input, 
                                 out_channels=output, 
                                 kernel_size=1)
        self.size = sz
        self.in_channels = input

        self.out_channels = output
    
    def forward(self, x, edge_index, edge_attr=None, batch=None):
        output = self.layer(x)
        return output

'''

New and improved version of graph neural network with a linear output layer. 

'''  
class GNNModel(nn.Module):
    def __init__(self, input=3, output=20, hidden=20, layers=2, 
                 layer_type='GATConv', activation='LeakyReLU',
                 edge_dim=None, layer_norm=False, **kwargs):
        super(GNNModel, self).__init__()

        # Initialize the first layer
        graph_layer = globals()[layer_type]
        self.initial = graph_layer(input, hidden, edge_dim=edge_dim)
        
        # Initialize the subsequent layers
        self.module_list = nn.ModuleList([graph_layer(hidden, hidden, edge_dim=edge_dim) for _ in range(layers - 1)])
        
        # Output layer
        self.output = nn.Linear(hidden, output)
        #self.output = graph_layer(hidden, output)

        # activation function
        self.activation = globals()[activation]()

        self.layer_type = layer_type
        self.hidden_channels = hidden
        if layer_norm:
            self.norm = LayerNorm(hidden)

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        # x = data.x
        # edge_index = data.edge_index

        x = self.initial(x, edge_index, edge_attr=edge_attr)
        x = self.activation(x)
        for layer in self.module_list:
            x = layer(x, edge_index, edge_attr=edge_attr)
            x = self.activation(x)
            if hasattr(self, 'norm'):
                x = self.norm(x)
        if self.layer_type=='CNNLayer':
            num_nodes = x.size()[2]**2 if batch == None else x.size()[0] * x.size()[2] * x.size()[3]
            cnn_bsz = 1 if batch == None else batch
            #x = x.reshape(cnn_bsz, self.hidden_channels, num_nodes).squeeze().T
            x = x.flatten(start_dim=2, end_dim=3).mT.flatten(start_dim=0, end_dim=1)
        x = self.output(x)
        return x
    

class CNN_Final_VN_Model(torch.nn.Module):
    def __init__(self, input=3, output=20, hidden=20, layers=2, 
                    layer_type='GATConv', activation='LeakyReLU', batches=False, 
                    edge_dim=None, **kwargs):
        super(CNN_Final_VN_Model, self).__init__()
        # Initialize the first layer
        graph_layer = globals()[layer_type]
        self.initial = graph_layer(input, hidden, edge_dim=edge_dim)
        
        # Initialize the subsequent layers
        self.module_list = nn.ModuleList([graph_layer(hidden, hidden, edge_dim=edge_dim) for _ in range(layers - 1)])
        
        # Output layer
        self.output = nn.Linear(hidden, output)
        #self.output = graph_layer(hidden, output)

        # activation function
        self.activation = globals()[activation]()

        self.layer_type = layer_type
        self.hidden_channels = hidden

        self.virtualnode_embedding = torch.nn.Embedding(1, hidden)
        self.mlp_virtualnode_list = torch.nn.ModuleList()
        torch.nn.init.constant_(self.virtualnode_embedding.weight.data, 0)
        for layer in range(layers  - 2):
            if batches:
                self.mlp_virtualnode_list.append(
                    torch.nn.Sequential(torch.nn.Linear(hidden, hidden), torch.nn.BatchNorm1d(hidden), torch.nn.LeakyReLU(), \
                                        torch.nn.Linear(hidden, hidden), torch.nn.BatchNorm1d(hidden), torch.nn.LeakyReLU()))
            else:
                self.mlp_virtualnode_list.append(
                    torch.nn.Sequential(torch.nn.Linear(hidden, hidden), torch.nn.ReLU(), \
                                        torch.nn.Linear(hidden, hidden), torch.nn.ReLU()))
        self.mlp_virtualnode_list.append(torch.nn.Sequential(torch.nn.Linear(hidden, hidden), torch.nn.ReLU(), \
                                        torch.nn.Linear(hidden, output), torch.nn.ReLU()))
                

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        # x = data.x
        # edge_index = data.edge_index

        x = self.initial(x, edge_index, edge_attr=edge_attr)
        x = self.activation(x)
        for layer in self.module_list:
            x = layer(x, edge_index, edge_attr=edge_attr)
            x = self.activation(x)
        
        if batch == None:
            num_nodes = x.size()[2]**2
            out = x.reshape(1, self.hidden_channels, num_nodes).squeeze().T
            vn_emb = torch.sum(x, dim = 0)
            # vn_emb = self.virtualnode_embedding(torch.zeros(1).to(edge_index.dtype).to(edge_index.device))
            # vn_emb  = global_add_pool(x, None, size=1) + vn_emb
        else: 
            out = x.flatten(start_dim=2, end_dim=3).mT.flatten(start_dim=0, end_dim=1)
            vn_emb = self.virtualnode_embedding(torch.zeros(batch.num_graphs).to(edge_index.dtype).to(edge_index.device))
            vn_emb = global_add_pool(out, batch.batch) + vn_emb
            
        for mlp_layer in self.mlp_virtualnode_list:
            vn_emb = mlp_layer(vn_emb)
        x = self.output(out)
        return x, vn_emb

class GNN_Final_VN_Model(torch.nn.Module):
    """
    GNN model that customizes the torch_geometric.graphgym.models.gnn.GNN
    to support specific handling of new conv layers.
    """
    def __init__(self, input=3, output=20, hidden=20, layers=2, 
                    layer_type='GATConv', activation='LeakyReLU', batches=False, 
                    edge_dim=None, **kwargs):
        super(GNN_Final_VN_Model, self).__init__()
        # Initialize the first layer
        graph_layer = globals()[layer_type]
        self.initial = graph_layer(input, hidden, edge_dim=edge_dim)
        
        # Initialize the subsequent layers
        self.module_list = nn.ModuleList([graph_layer(hidden, hidden, edge_dim=edge_dim) for _ in range(layers - 1)])
        
        # Output layer
        self.output = torch.nn.Linear(hidden, output)

        # activation function
        self.activation = globals()[activation]()

        # code from Chen Cai
        self.virtualnode_embedding = torch.nn.Embedding(1, hidden)
        self.mlp_virtualnode_list = torch.nn.ModuleList()
        torch.nn.init.constant_(self.virtualnode_embedding.weight.data, 0)
        for layer in range(layers  - 2):
            # if batches:
            #     self.mlp_virtualnode_list.append(
            #         torch.nn.Sequential(torch.nn.Linear(hidden, hidden), torch.nn.BatchNorm1d(hidden), torch.nn.LeakyReLU(), \
            #                             torch.nn.Linear(hidden, hidden), torch.nn.BatchNorm1d(hidden), torch.nn.LeakyReLU()))
            # else:
            self.mlp_virtualnode_list.append(
                    torch.nn.Sequential(torch.nn.Linear(hidden, hidden), torch.nn.ReLU(), \
                                        torch.nn.Linear(hidden, hidden), torch.nn.ReLU()))
        self.mlp_virtualnode_list.append(torch.nn.Sequential(torch.nn.Linear(hidden, hidden), torch.nn.ReLU(), \
                                        torch.nn.Linear(hidden, output), torch.nn.ReLU()))
                
    def forward(self, x, edge_index, edge_attr=None, batch=None):
        out = self.initial(x, edge_index, edge_attr=edge_attr)
        
        for layer in self.module_list:            
            out = layer(out, edge_index, edge_attr=edge_attr)
            out = self.activation(out)
        
        #vn_emb = self.virtualnode_embedding(torch.zeros(1).to(edge_index.dtype).to(edge_index.device))
        if batch == None:
            vn_emb = self.virtualnode_embedding(torch.zeros(1).to(edge_index.dtype).to(edge_index.device))
            vn_emb  = global_add_pool(out, None, size=1) + vn_emb
        else: 
            vn_emb = self.virtualnode_embedding(torch.zeros(batch.num_graphs).to(edge_index.dtype).to(edge_index.device))
            vn_emb = global_add_pool(out, batch.batch) + vn_emb
        
        for mlp_layer in self.mlp_virtualnode_list:
            vn_emb = mlp_layer(vn_emb)
        out = self.output(out)
        return out, vn_emb

class GNN_VN_Model(torch.nn.Module):
    """
    GNN model that customizes the torch_geometric.graphgym.models.gnn.GNN
    to support specific handling of new conv layers.
    """
    def __init__(self, input=3, output=20, hidden=20, layers=2, 
                 layer_type='GATConv', activation='LeakyReLU', batches=False, 
                 edge_dim=None, **kwargs):
        super(GNN_VN_Model, self).__init__()

        torch.manual_seed(1234567)
        # Initialize the first layer
        graph_layer = globals()[layer_type]
        self.initial = graph_layer(input, hidden, edge_dim=edge_dim)
        
        # Initialize the subsequent layers
        self.module_list = nn.ModuleList([graph_layer(hidden, hidden, edge_dim=edge_dim) for _ in range(layers - 1)])
        
        # Output layer
        self.output = torch.nn.Linear(hidden, output)

        # activation function
        self.activation = globals()[activation]()

        # code from Chen Cai
        self.virtualnode_embedding = torch.nn.Embedding(1, hidden)
        self.mlp_virtualnode_list = torch.nn.ModuleList()
        torch.nn.init.constant_(self.virtualnode_embedding.weight.data, 0)
        for layer in range(layers  - 1):
            if batches:
                self.mlp_virtualnode_list.append(
                    torch.nn.Sequential(torch.nn.Linear(hidden, hidden), torch.nn.BatchNorm1d(hidden), torch.nn.LeakyReLU(), \
                                        torch.nn.Linear(hidden, hidden), torch.nn.BatchNorm1d(hidden), torch.nn.LeakyReLU()))
            else:
                self.mlp_virtualnode_list.append(
                    torch.nn.Sequential(torch.nn.Linear(hidden, hidden), torch.nn.ReLU(), \
                                        torch.nn.Linear(hidden, hidden), torch.nn.ReLU()))
                
    def forward(self, x, edge_index, edge_attr=None, batch=None):
        out = self.initial(x, edge_index, edge_attr=edge_attr)
        
        if batch == None:
            vn_emb = self.virtualnode_embedding(torch.zeros(1).to(edge_index.dtype).to(edge_index.device))
        else: 
            vn_emb = self.virtualnode_embedding(torch.zeros(batch.num_graphs).to(edge_index.dtype).to(edge_index.device))
        
        for layer in self.module_list:

            out = out + vn_emb[batch.batch] if batch != None else out + vn_emb
            
            out = layer(out, edge_index, edge_attr=edge_attr)
            if batch == None:
                vn_emb  = global_add_pool(out, None, size=1) + vn_emb
            else:
                vn_emb = global_add_pool(out, batch.batch) + vn_emb
            for mlp_layer in self.mlp_virtualnode_list:
                vn_emb = mlp_layer(vn_emb)
        out = self.output(out)
        return out

class GNN_VN_Hierarchical(torch.nn.Module):
    def __init__(self, input=3, output=20, hidden=20, layers=2, 
                 layer_type='GATConv', activation='LeakyReLU', batches=False, num_vn=1, 
                 n=100, m = 100, **kwargs):
        super(GNN_VN_Hierarchical, self).__init__()

        torch.manual_seed(1234567)
        # Initialize the first layer
        graph_layer = globals()[layer_type]
        self.initial = graph_layer(input, hidden)
        
        # Initialize the subsequent layers
        self.module_list = nn.ModuleList([graph_layer(hidden, hidden) for _ in range(layers - 1)])
        
        # Output layer
        self.output = torch.nn.Linear(hidden, output)

        # activation function
        self.activation = globals()[activation]()

        # number of virtual nodes
        self.num_vn = num_vn
        

        # code from Chen Cai
        self.virtualnode_embedding = torch.nn.Embedding(1, hidden)
        self.mlp_virtualnode_list = torch.nn.ModuleList()
        torch.nn.init.constant_(self.virtualnode_embedding.weight.data, 0)
        for layer in range(layers  - 1):
            if batches:
                self.mlp_virtualnode_list.append(
                    torch.nn.Sequential(torch.nn.Linear(hidden, hidden), torch.nn.BatchNorm1d(hidden), torch.nn.ReLU(), \
                                        torch.nn.Linear(hidden, hidden), torch.nn.BatchNorm1d(hidden), torch.nn.ReLU()))
            else:
                self.mlp_virtualnode_list.append(
                    torch.nn.Sequential(torch.nn.Linear(hidden, hidden), torch.nn.ReLU(), \
                                        torch.nn.Linear(hidden, hidden), torch.nn.ReLU()))
                
    def forward(self, x, edge_index, h_blocks, h_levels, h_num):
        out = self.initial(x, edge_index)
        vn_direct = self.virtualnode_embedding(torch.zeros(h_num).to(edge_index.dtype).to(edge_index.device))
        vn_root = self.virtualnode_embedding(torch.zeros(1).to(edge_index.dtype).to(edge_index.device))

        for layer in self.module_list:
            # Get information from virtual nodes
            out = out + vn_direct[h_blocks]
            out = layer(out, edge_index)

            # Get information from real nodes + root virtual node
            vn_direct = global_add_pool(out, h_blocks) + vn_direct
            vn_direct = vn_direct + vn_root

            # Root VN gets information from vn_direct
            vn_root = global_add_pool(vn_direct, None, size=1) + vn_root
            
            for mlp_layer in self.mlp_virtualnode_list:
                vn_direct = mlp_layer(vn_direct)
                vn_root = mlp_layer(vn_root)
            
            
        return out
        

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

class Transformer(torch.nn.Module):
    def __init__(self, input, hidden, output, layers, heads=2, dropout=0.3, **kwargs):
        super(Transformer, self).__init__()
        # project to high dimensions
        self.projection_layer = nn.Linear(in_features=input, out_features=hidden)
        #self.module_list = nn.ModuleList([graph_layer(hidden, hidden, edge_dim=edge_dim) for _ in range(layers)])
        self.layer = nn.TransformerEncoderLayer(d_model = input, nhead=heads, batch_first=True)
        self.final_layer = nn.Linear(in_features=input, out_features=output)
        self.activation = LeakyReLU()

    def forward(self, x, edge_index):
        out = self.activation(self.layer(x)).squeeze()
        out = self.final_layer(out)
        return out