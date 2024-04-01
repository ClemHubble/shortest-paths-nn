import numpy as np
import torch, queue
from torch_geometric.data import Data
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
import multiprocessing as mp
import time
import itertools

import argparse

from torch_geometric.utils import to_networkx

def npz_to_dataset(data):
    
    edge_index = torch.tensor(data['edge_index'], dtype=torch.long)

    srcs = data['srcs']
    tars = data['tars']
    lengths = data['lengths']
    node_features = torch.tensor(data['node_features'], dtype=torch.double)

    return srcs, tars, lengths, node_features, edge_index

def triangle_graph_to_wavefront_obj(vertices, edge_index, n, m, filename, triangle=False):
    f = open(filename, "w")

    ids = np.reshape(np.arange(1, n * m  + 1), (n, m))

    for v in vertices:
        string = f'v {v[0]} {v[1]} {v[2]}\n'
        f.write(string)
    graph_data = Data(x =vertices, edge_index = edge_index)
    G = to_networkx(graph_data)
    
    for i in range(n - 1):
        for j in range(m - 1):
            # cell_idx = ids[i, j]
            if triangle:
                if G.has_edge(ids[i, j + 1], ids[i + 1, j]):
                    # diagonal = (ids[i, j + 1], ids[i + 1, j])
                    f1 = f'f {ids[i, j]} {ids[i + 1, j]} {ids[i, j + 1]} \n'
                    f2 = f'f {ids[i, j + 1]} {ids[i + 1, j]} {ids[i + 1, j + 1]}\n'
                    f.write(f1)
                    f.write(f2)
                else:
                    # diagonal = (ids[i, j], ids[i + 1, j + 1])
                    f1 = f'f {ids[i, j]} {ids[i + 1, j + 1]} {ids[i, j + 1]}\n'
                    f2 = f'f {ids[i, j]} {ids[i + 1, j]} {ids[i + 1, j + 1]} \n'
                    f.write(f1)
                    f.write(f2)
            else:
                face = f'f {ids[i, j]}  {ids[i + 1, j]} {ids[i + 1, j + 1]} {ids[i, j + 1]} \n'
                f.write(face)
            
    f.close()
    print("Saved wavefront obj to:", filename)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw-data', type=str)
    parser.add_argument('--filename', type=str)
    parser.add_argument('--n', type=int)
    parser.add_argument('--m', type=int)
    parser.add_argument('--triangle', action='store_true')

    args = parser.parse_args()

    data = np.load(args.raw_data, allow_pickle=True)
    _, _, _, vertices, edge_index = npz_to_dataset(data)
    print(len(vertices))
    triangle_graph_to_wavefront_obj(vertices, edge_index, args.n, args.m, args.filename, triangle=args.triangle)

if __name__ =='__main__':
    main()