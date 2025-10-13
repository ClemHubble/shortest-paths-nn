import torch, time, itertools
import numpy as np
import networkx as nx

from scipy.spatial import Delaunay

from torch_geometric.utils import erdos_renyi_graph
from torch_geometric.nn import knn_graph, radius_graph
from torch_geometric.utils import to_networkx
from torch_geometric.data import Data

import os

from .baselines import *

import copy

from tqdm import tqdm, trange

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def npz_to_dataset(data):
    
    edge_index = torch.tensor(data['edge_index'], dtype=torch.long)

    srcs = torch.tensor(data['srcs'])
    tars = torch.tensor(data['tars'])
    lengths = torch.tensor(data['lengths'])
    node_features = torch.tensor(data['node_features'], dtype=torch.double)
    edge_weights = torch.tensor(data['distances'])

    return srcs, tars, lengths, node_features, edge_index, edge_weights

def generate_train_data(train_file_name):
    data = np.load(train_file_name, allow_pickle=True)
    train_srcs, train_tars, train_lengths, node_features, edge_index, edge_weights = npz_to_dataset(data)
    
    graph_data = Data(x =node_features, edge_index = edge_index, edge_attr = edge_weights)
    G = to_networkx(graph_data)
    for i in trange(len(edge_index[0])):
        v1 = edge_index[0][i].item()
        v2 = edge_index[1][i].item()
        G[v1][v2]['weight'] = graph_data.edge_attr[i].item()

    test_info = {'graph': graph_data, 
                 'nx_graph': G,
                 'node_features': np.copy(node_features),
                 'train_srcs': train_srcs.numpy(), 
                 'train_tars': train_tars.numpy(), 
                 'train_lengths': train_lengths.numpy()}
    return test_info

def load_top_lvl_directory_from_config(dataset_name, config):
    layer_type = config['layer_type']
    vn = config['vn']
    mlp=config['mlp']
    loss_func = config['loss']
    if mlp:
        aggr = config['aggr']
    p = config['p']
    
    vn_ = 'vn' if vn else 'no-vn'
    mlp_ = f'mlp/{aggr}' if mlp else 'siamese'
    if mlp:
        directory = f'/data/sam/terrain/models/single_dataset/{dataset_name}/{layer_type}/{vn_}/mlp/p-{p}/{aggr}/{loss_func}'
    else:
        directory = f'/data/sam/terrain/models/single_dataset/{dataset_name}/{layer_type}/{vn_}/siamese/p-{p}/{loss_func}'
    return directory

def load_models(model_dictionary, file, trials = ['1', '2', '3', '4', '5'], finetune=False):
    best_models = {}

    for model in model_dictionary:
        directory = load_top_lvl_directory_from_config(file, model_dictionary[model])

        mlp = model_dictionary[model]['mlp']
        vn = model_dictionary[model]['vn']
        layer_type = model_dictionary[model]['layer_type']
        aggr = model_dictionary[model]['aggr']
        
        model_configs = model_dictionary[model]['model_config']
        best_models[model] = {}

        
        for name in model_configs:
            best_models[model][name] = {}
            best_models[model][name]['gnn_model'] = []
            best_models[model][name]['mlp_model'] = []
            for t in trials:
                cfg = model_configs[name]
                if aggr == 'combine' or aggr == 'sum+diff+vn':
                    cfg = copy.deepcopy(model_configs[name])
                    cfg['mlp']['input'] = cfg['mlp']['input'] * 3
                elif aggr == 'concat' or aggr == 'sum+diff':
                    cfg = copy.deepcopy(model_configs[name])
                    cfg['mlp']['input'] = cfg['mlp']['input'] * 2 
                
                if mlp:    
                    mlp_model = MLPBaseline1(initialize_mlp(**cfg['mlp']), max=True, aggr=aggr)
                else:
                    mlp_model=None
                if layer_type=='MLP':
                    cfg_mlp = model_configs[name]['gnn']
                    gnn_model = initialize_mlp(**cfg_mlp)
                elif layer_type == 'Transformer':
                    gnn_model=Transformer(**cfg['gnn'])
                elif aggr == 'sum+diff+vn' and layer_type=='CNNLayer':
                    gnn_model = CNN_Final_VN_Model(batches=None, layer_type=layer_type, edge_dim=1, **cfg['gnn'])
                elif aggr == 'sum+diff+vn':
                    gnn_model = GNN_Final_VN_Model(batches=None, layer_type=layer_type,edge_dim=1, **cfg['gnn'])

                else:
                    gnn_model = GNN_VN_Model(batches=None, layer_type=layer_type,edge_dim=1, **cfg['gnn']) if vn else GNNModel(layer_type=layer_type, edge_dim=1, **cfg['gnn'])
                
                if finetune:
                    model_pth = os.path.join(directory, name, t, 'finetune', 'final_model.pt')
                    print(model_pth)
                else:
                    model_pth = os.path.join(directory, name, t, 'final_model.pt')

                if not os.path.exists(model_pth):
                    print(model, "not trained yet or not found at", model_pth)
                    continue
                model_info = torch.load(model_pth, map_location='cpu')
                if mlp:
                    gnn_info = model_info['gnn_state_dict']
                    mlp_info = model_info['mlp_state_dict']
                    gnn_model.load_state_dict(gnn_info)
                    mlp_model.load_state_dict(mlp_info)
                    gnn_model.to(torch.double)
                    mlp_model.to(torch.double)
                else:
                    gnn_model.load_state_dict(model_info)
                    gnn_model.to(torch.double)
                best_models[model][name]['gnn_model'].append(gnn_model)
                best_models[model][name]['mlp_model'].append(mlp_model)
    return best_models



def compute_embeddings(gnn, data):
    vn_emb = None
    if isinstance(gnn, nn.Sequential):
        embedding_vecs = gnn(data.x)
    elif isinstance(gnn, Transformer):
        embedding_vecs = gnn(data.x.unsqueeze(dim=0), edge_index=None)
    elif isinstance(gnn, CNN_Final_VN_Model):
        embedding_vecs, vn_emb = gnn(data, edge_index=None)
    elif isinstance(gnn, GNN_Final_VN_Model):
        embedding_vecs, vn_emb = gnn(data.x, data.edge_index, edge_attr = data.edge_attr)
    elif isinstance(gnn.initial, CNNLayer):
        embedding_vecs = gnn(data, edge_index = None)
    else:
        embedding_vecs = gnn(data.x, data.edge_index, edge_attr = data.edge_attr)
    return embedding_vecs.detach().numpy(), vn_emb

def test_model_error(embeddings, srcs, tars, lengths, mlp_model=None, p=1, vn_emb=None):
    if vn_emb is not None:
        vn_emb = torch.tensor(vn_emb)
    if mlp_model:
        print("mlp predictions.....")
        preds = mlp_model(torch.tensor(embeddings[srcs]), 
                          torch.tensor(embeddings[tars]), vn_emb=vn_emb)
        print("finished mlp predictions")
        preds = preds.squeeze(1).detach().numpy()
    else:
        preds = np.linalg.norm(embeddings[srcs]- embeddings[tars], ord=p, axis=1)
    nz = np.nonzero(lengths)[0]
    total_relative_error = np.abs(preds[nz] - lengths[nz])/lengths[nz]
    mean_absolute_error = np.abs(preds[nz]-lengths[nz])
    squared_error = np.square(preds[nz] - lengths[nz])
    return total_relative_error, mean_absolute_error, squared_error, preds


def generate_test_dataset(G, s=100, keep_path=False, cutoff=50):
    num_nodes = len(G.nodes)
    lengths = []
    srcs = []
    tars = []
    paths = []
    if isinstance(s, list):
        src_lst = s
    else:
        src_lst = np.random.choice(num_nodes, size=s)
    for i in trange(len(src_lst)):
        src = src_lst[i]
        if keep_path:
            all_pairs_shortest_paths = nx.single_source_dijkstra_path(G, src, weight='weight')
            for tar in all_pairs_shortest_paths:
                if all_pairs_shortest_paths[tar] == 0:
                    continue
                srcs.append(src)
                tars.append(tar)
                paths.append(all_pairs_shortest_paths[tar])
        else:
            all_pairs_shortest_paths = nx.single_source_dijkstra_path_length(G, src,  cutoff=cutoff, weight='weight')
            for tar in all_pairs_shortest_paths:
                if all_pairs_shortest_paths[tar] == 0:
                    continue
                srcs.append(src)
                tars.append(tar)
                lengths.append(all_pairs_shortest_paths[tar])
    if keep_path:
        return srcs, tars, paths
    else:
        return srcs, tars, lengths

def get_highest_points_per_patch(patch_size, elevation_array, npp = 1):
    rows = elevation_array.shape[0]
    cols = elevation_array.shape[1]
    counts = np.reshape(np.arange(0, rows * cols), (rows, cols))
    srcs = []
    for i in range(0, elevation_array.shape[0], patch_size):
        for j in range(0, elevation_array.shape[1], patch_size):
            patch = elevation_array[i:i + patch_size, j:j + patch_size]
            ct_patch = counts[i:i+patch_size, j:j + patch_size]
            ind = np.unravel_index(np.argmax(patch, axis=None), patch.shape)
            src = ct_patch[ind]
            srcs.append(src)
    return srcs

def dimensionality_reduction(X, n_components=2, method='PCA'):
    if method == 'PCA':
        dim_reduce = PCA(n_components=n_components)
    elif method == 'tSNE':
        dim_reduce =TSNE(n_components=n_components)
    else:
        raise NotImplementedError("Other methods not supported")

    X_new = dim_reduce.fit_transform(X)
    return X_new

def get_contour(x, y, z, tar_idx, n):
    height_array = np.zeros((n, n))
    x_array = np.zeros((n, n))
    y_array = np.zeros((n, n))
    for i in range(len(tar_idx)):
        t_idx = tar_idx[i]
        t_x = t_idx // n
        t_y = t_idx % n
        height_array[t_x, t_y] = z[i]
        x_array[t_x, t_y] = x[i]
        y_array[t_x, t_y ] = y[i]
    return height_array, x_array, y_array