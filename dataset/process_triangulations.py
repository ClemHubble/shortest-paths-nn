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
import os
from point_sampler import * 
from dataset import * 
import csv


def process_node_features(filename):

    node_idx_dict = {}
    node_feature_lst = []
    counter = 0
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            if counter == 0:
                counter += 1
                continue
            idx = int(row[0])
            x = float(row[1])/1000
            y = float(row[2])/1000
            z = float(row[3])/1000
            node_idx_dict[idx] = [x, y, z]
            node_feature_lst.append([x, y, z])
            counter += 1
    return node_idx_dict, node_feature_lst

def process_edges(filename):
    edges = []
    counter = 0
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            if counter == 0:
                counter += 1
                continue
            edges.append([int(row[0]), int(row[1])])
    return edges

def construct_nx_graph(edges, node_mappings, scale=False):
    G = nx.Graph()
    for edge in tqdm(edges):
        idx1 = int(edge[0])
        idx2 = int(edge[1])
        p1 = np.array(node_mappings[idx1])
        p2 = np.array(node_mappings[idx2])
        if scale:
            slope = (abs(p1[2] - p2[2]))/(abs(p1[0] - p2[0]) + abs(p1[1] - p2[1]))
            angle_of_elevation = np.abs(np.arctan(p1[2] - p2[2])/np.linalg.norm(p2[:2] - p1[:2], ord=2))
            val = angle_of_elevation
            w = (1 + val) * np.linalg.norm(p1 - p2, ord=2)
        else: 
            w = np.linalg.norm(p1 - p2, ord=2)
        G.add_edge(idx1, idx2, weight=w)
    print("Number of nodes:", G.number_of_nodes())
    print("Number of edges:", G.number_of_edges())
    return G


def construct_dataset(G, 
                      node_features, 
                      filename, 
                      num_srcs, 
                      samples_per_source,
                      sampling_method='random-sampling'):
    edges, distances = to_pyg_graph(G)
    if sampling_method == 'random-sampling':
        src_nodes = np.random.choice(len(node_features), size=num_srcs)
        src_sampling_fn_pairs = [(src_nodes, random_sampling)]
    else:
        raise NotImplementedError("Sampling technique not implemented yet")
    srcs = []
    tars = []
    lengths = []
    print("Number of source nodes:", num_srcs)
    print("Generating shortest path distances.....")
    for pair in src_sampling_fn_pairs:
        src_nodes = pair[0]
        sampling_fn = pair[1]
        for src in tqdm(src_nodes):
            source, target, length = sampling_fn(G, samples_per_source, src=src)
            srcs += source
            tars += target
            lengths += length

    print("Number of lengths in dataset:", len(lengths))
    print("Saved dataset in:", filename)
    np.savez(filename, 
         edge_index = edges, 
         distances=distances, 
         srcs = srcs,
         tars = tars,
         lengths = lengths,
         node_features=node_features)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', type=str)
    parser.add_argument('--edge-input-data', type=str)
    parser.add_argument('--node-feature-data', type=str)
    parser.add_argument('--num-sources', type=int)
    parser.add_argument('--dataset-size', type=int)

    args = parser.parse_args()
    node_idx_dict, node_features = process_node_features(args.node_feature_data)
    edges = process_edges(args.edge_input_data)

    G = construct_nx_graph(edges, node_idx_dict)
    construct_dataset(G=G, 
                       node_features=node_features, 
                       filename=args.filename,
                       num_srcs=args.num_sources, 
                       samples_per_source = args.dataset_size//args.num_sources)
    return 

if __name__ == '__main__':
    main()