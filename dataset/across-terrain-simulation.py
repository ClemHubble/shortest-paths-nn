import numpy as np
import torch, queue
from torch_geometric.data import Data
from torch.utils.data import Dataset
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

class SingleGraphShortestPathDataset(Dataset):
    def __init__(self, sources, targets, lengths):
        self.sources = sources
        self.targets = targets
        self.lengths = lengths

    def __len__(self):
        return len(self.sources)

    def __getitem__(self, idx):
        return self.sources[idx], self.targets[idx], self.lengths[idx]

def single_source_sample(G, num_per_src, num_srcs):
    number_of_nodes = G.number_of_nodes()
    src_nodes = np.random.choice(number_of_nodes, size=NSRCS)
    srcs = []
    tars = []
    lengths = []
    for src in src_nodes:
        shortest_paths = nx.single_source_dijkstra_path_length(G, s, weight='weight')
        for i in trange(num_per_src):
            t = np.random.choice(number_of_nodes)
            srcs.append(s)
            tars.append(t)
            lengths.append(shortest_paths[t])
    dataset = SingleGraphShortestPathDataset(sources = srcs, targets = tars, lengths = lengths)
    return dataset

def random_sample(G, num_sample):
    number_of_nodes = G.number_of_nodes()
    srcs = []
    tars = []
    lengths = []
    for _ in trange(num_sample):
        src, tar = np.random.choice(number_of_nodes, [2, ], replace=False)
        length = nx.shortest_path_length(G, src, tar, weight='weight')
        srcs.append(src)
        tars.append(tar)
        lengths.append(length)
    dataset = SingleGraphShortestPathDataset(sources = srcs, targets = tars, lengths = lengths)
    return dataset

def construct_cross_terrains_dataset(nx_graphs, pyg_graphs, num_per_graph, sampling_technique='single_source_sample'):
    dataset = {'graphs': pyg_graphs, 'datasets': []}
    num_graphs = len(nx_graphs)

    for i in range(num_graphs):
        print("Processing graph:", i)
        nx_graph = nx_graphs[i]
        if sampling_technique == 'single_source_sample':
            dataset['datasets'].append(single_source_sample(nx_graph, num_per_graph//NSRCS, NSRCS))
        elif sampling_technique == 'random_sample':
            dataset['datasets'].append(random_sample(nx_graph, num_per_graph))
        else:
            raise NotImplementedError('Other sampling techniques not implemented')
    return dataset

def npz_to_dataset(data):
    
    edge_index = torch.tensor(data['edge_index'], dtype=torch.long)

    srcs = torch.tensor(data['srcs'])
    tars = torch.tensor(data['tars'])
    lengths = torch.tensor(data['lengths'])
    node_features = torch.tensor(data['node_features'], dtype=torch.double)
    edge_weights = torch.tensor(data['distances'])

    return srcs, tars, lengths, node_features, edge_index, edge_weights

def retrieve_artificial_dataset(file_pth):
    # pth = f'/data/sam/terrain/data/artificial/change-heights/amp-{a}-res-{RES}-train-50k.npz'
    # test_info = generate_train_data(pth, cnn_sz=100)
    #amps = [1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0]
    amps = [2.0, 6.0,  10.0, 14.0,  18.0]
    pyg_graphs = []
    nx_graphs = []
    for a in amps:
        fname = os.path.join(file_pth, f'amp-{a}-res-2-train-50k.npz')
        np_data = np.load(fname, allow_pickle=True)
        _, _, _, node_features, edge_index, edge_weights = npz_to_dataset(np_data)
        pyg_graph = Data(x =node_features, edge_index = edge_index, edge_attr = edge_weights)
        nx_graph = to_networkx(pyg_graph)
        for i in range(len(edge_index[0])):
            v1 = edge_index[0][i].item()
            v2 = edge_index[1][i].item()
            nx_graph[v1][v2]['weight'] = pyg_graph.edge_attr[i].item()
        nx_graphs.append(nx_graph)
        pyg_graphs.append(pyg_graph)
    return nx_graphs, pyg_graphs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-name', type=str)
    parser.add_argument('--raw-data', type=str)
    parser.add_argument('--filename', type=str) # saves should be named `gr-{graph-resolution}-ps-{patch-size}-ol-{overlap}`
    parser.add_argument('--graph-resolution', type=int)
    parser.add_argument('--per-graph', type=int)
    parser.add_argument('--patch-size', type=int)
    parser.add_argument('--overlap', type=int)
    parser.add_argument('--sampling-technique', type=str)

    args = parser.parse_args()
    if args.dataset_name != 'artificial':
        raise NotImplementedError('Other datasets not implemented yet')
    file_pth = '/data/sam/terrain/data/artificial/change-heights'
    nx_graphs, pyg_graphs = retrieve_artificial_dataset(file_pth)
    dataset = construct_cross_terrains_dataset(nx_graphs, 
                                               pyg_graphs,
                                               args.per_graph,
                                               sampling_technique=args.sampling_technique)
    torch.save(dataset, args.filename)

if __name__ == '__main__':
    main()