import numpy as np
import torch, queue
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
import multiprocessing as mp
import time
import os

import argparse

NSRCS = 100

DATASET_INFO = {'norway': [10, False], 'phil': [3, True], 'holland': [1.524, True]}

def single_source_sample(G, num_per_src, num_srcs):
    number_of_nodes = G.number_of_nodes()
    src_nodes = np.random.choice(number_of_nodes, size=NSRCS)
    shortest_pth_dataset = {'srcs': [], 'tars': [], 'lengths': []}
    for src in src_nodes:
        shortest_paths = nx.single_source_dijkstra_path_length(G, s, weight='weight')
        for i in trange(num_per_src):
            t = np.random.choice(number_of_nodes)
            shortest_pth_dataset['srcs'].append(s)
            shortest_pth_dataset['tars'].append(t)
            shortest_pth_dataset['lengths'].append(shortest_paths[t])
    
    return shortest_pth_dataset

def random_sample(G, num_sample):
    number_of_nodes = G.number_of_nodes()
    shortest_pth_dataset = {'srcs': [], 'tars': [], 'lengths': []}
    for _ in trange(num_sample):
        src, tar = np.random.choice(number_of_nodes, [2, ], replace=False)
        length = nx.shortest_path_length(G, src, tar, weight='weight')
        shortest_pth_dataset['srcs'].append(src)
        shortest_pth_dataset['tars'].append(tar)
        shortest_pth_dataset['lengths'].append(length)
    return shortest_pth_dataset

def construct_cross_terrains_dataset(nx_graphs, pyg_graphs, num_per_graph, sampling_technique='single_source_sample'):
    dataset = {'graphs': pyg_graphs, 'datasets': []}
    num_graphs = len(nx_graphs)

    for i in range(num_graphs):
        print("Processing graph:", i)
        nx_graph = nx_graphs[i]
        if sampling_technique == 'single_source_sample':
            dataset['datasets'] = single_source_sample(nx_graph, num_per_graph//NSRCS, NSRCS)
        elif sampling_technique == 'random_sample':
            dataset['datasets'] = random_sample(nx_graph, num_per_graph)
        else:
            raise NotImplementedError('Other sampling techniques not implemented')
    return dataset

def retrieve_artificial_dataset(file_pth):
    # pth = f'/data/sam/terrain/data/artificial/change-heights/amp-{a}-res-{RES}-train-50k.npz'
    # test_info = generate_train_data(pth, cnn_sz=100)
    amps = []
    pyg_graphs = []
    nx_graphs = []
    for a in amps:
        fname = os.path.join(file_pth, f'amp-{a}-res-2.0-train-50k.npz')
        _, _, _, node_features, edge_index, edge_weights = npz_to_dataset(data)
        py_graph = Data(x =node_features, edge_index = edge_index, edge_attr = edge_weights)
        nx_graph = to_networkx(py_graph)
        for i in trange(len(edge_index[0])):
            v1 = edge_index[0][i].item()
            v2 = edge_index[1][i].item()
            nx_graph[v1][v2]['weight'] = pyg_graph.edge_attr[i].item()
        nx_graphs.append(nx_graph)
        pyg_graphs.append(pyg_graphs)
    raise nx_graphs, pyg_graphs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-name', type=str)
    parser.add_argument('--raw-data', type=str)
    parser.add_argument('--filename', type=str) # saves should be named `gr-{graph-resolution}-ps-{patch-size}-ol-{overlap}`
    parser.add_argument('--graph-resolution', type=int)
    parser.add_argument('--per-graph', type=int)
    parser.add_argument('--patch-size', type=int)
    parser.add_argument('--overlap', type=int)
    parser_add_argument('--sampling-technique', type=str)

    args = parser.parse_args()
    if args.dataset_name != 'artificial':
        raise NotImplementedError('Other datasets not implemented yet')
    nx_graphs, pyg_graphs = retrieve_artificial_dataset(file_pth)
    dataset = construct_cross_terrains_dataset(nx_graphs, 
                                               pyg_graphs,
                                               args.per_graph,
                                               sampling_technique=args.sampling_technique)
    torch.save(dataset, args.filename)

if __name__ == '__main__':
    main()