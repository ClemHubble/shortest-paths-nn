from typing import Any, Optional

import numpy as np
import torch

from torch_geometric.data import Data, HeteroData
from torch_geometric.transforms import BaseTransform, ToUndirected
from torch_geometric.utils import (
    get_laplacian,
    to_scipy_sparse_matrix,
    is_undirected
)
from scipy.sparse.linalg import eigs, eigsh
from torch_geometric.transforms import ToUndirected

from .baselines import *

class TerrainHeteroData(HeteroData):
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'src':
            return self['real'].x.size(0)
        if key == 'tar':
            return self['real'].x.size(0)
        return super().__inc__(key, value, *args, **kwargs)

def add_node_attr(data: Data, value: Any,
                  attr_name: Optional[str] = None) -> Data:
    # TODO Move to `BaseTransform`.
    if attr_name is None:
        if 'x' in data:
            x = data.x.view(-1, 1) if data.x.dim() == 1 else data.x
            data.x = torch.cat([x, value.to(x.device, x.dtype)], dim=-1)
        else:
            data.x = value
    else:
        data[attr_name] = value

    return data

def add_laplace_positional_encoding(data, k=10):
    eig_fn = eigsh

    num_nodes = data.num_nodes
    edge_index, edge_weight = get_laplacian(
        data.edge_index,
        data.edge_weight,
        normalization='sym',
        num_nodes=num_nodes,
    )

    L = to_scipy_sparse_matrix(edge_index, edge_weight, num_nodes)

    eig_vals, eig_vecs = eig_fn(
        L,
        k=k + 1,
        which='SA',
        return_eigenvectors=True
    )

    eig_vecs = np.real(eig_vecs[:, eig_vals.argsort()])
    print(eig_vals.argsort(), eig_vals[eig_vals.argsort()])
    pe = torch.from_numpy(eig_vecs[:, 1:k + 1])
    sign = -1 + 2 * torch.randint(0, 2, (k, ))
    pe *= sign

    data = add_node_attr(data, pe)
    return data


def add_virtual_node(data):
    hetero_data = HeteroData()
    sz_features = data.x.size()[1]
    hetero_data['real'].x = data.x.double()
    hetero_data['real', 'e1', 'real'].edge_index = data.edge_index

    vn = torch.zeros(size = (1, sz_features), dtype=torch.double )
    hetero_data['vn'].x = vn
    vn_edge_index = [[], []]
    for i in range(data.x.size()[0]):
        vn_edge_index[0].append(0)
        vn_edge_index[1].append(i)
    hetero_data['vn', 'e2', 'real'].edge_index = torch.tensor(vn_edge_index, dtype=torch.long)
    
    return hetero_data

def add_virtual_node_patch(data):
    hetero_data = TerrainHeteroData()
    sz_features = data.x.size()[1]
    hetero_data.src = data.src
    hetero_data.tar = data.tar
    hetero_data.length = data.length
    hetero_data['real'].x = data.x.double()
    hetero_data['real', 'e1', 'real'].edge_index = data.edge_index

    vn = torch.zeros(size = (1, sz_features), dtype=torch.double )
    hetero_data['vn'].x = vn
    vn_edge_index = [[], []]
    for i in range(data.x.size()[0]):
        vn_edge_index[0].append(0)
        vn_edge_index[1].append(i)
    hetero_data['vn', 'e2', 'real'].edge_index = torch.tensor(vn_edge_index, dtype=torch.long)
    hetero_data = ToUndirected()(hetero_data)
    return hetero_data

def test():
    # simple 3-cycle
    x_feat = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=torch.float)
    edge_index = torch.tensor([[0, 0, 1, 1, 2, 2], [1, 2, 0, 2, 1, 0]])
    #edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
    graph = Data(x = x_feat, edge_index = edge_index)
    vn_graph = add_virtual_node(graph)
    vn_graph = ToUndirected()(vn_graph)
    print(vn_graph)

    gnn_model = VNModel(vn_graph.metadata(), layer_type='GATConv' )
    print("trying gnn")
    val = gnn_model(vn_graph)
    print(val)
    

if __name__ == "__main__":
    test()