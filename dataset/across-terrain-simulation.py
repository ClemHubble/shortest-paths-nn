import numpy as np
import torch, queue
from torch_geometric.data import Data
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
import multiprocessing as mp
import time
import os

import argparse

DATASET_INFO = {'norway': [10, False], 'phil': [3, True], 'holland': [1.524, True]}

class TerrainPatchesData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'src':
            return self.x.size(0)
        if key == 'tar':
            return self.x.size(0)
        return super().__inc__(key, value, *args, **kwargs)

# Load DEM data from file
def load_dem_data_(filename, imperial=False):
    f = open(filename)

    lines = f.readlines()
    arr = []
    print("Elevation given in imperial units:", imperial)
    c = 1
    if imperial:
        c = 3.28084
    for i in range(1, len(lines)):
        vals = lines[i].split()
        a = []
        for j in range(len(vals)):
            a.append(float(vals[j])/c)
        arr.append(a)
    arr = np.array(arr)
    print("loaded DEM array with shape:", arr.shape)
    return arr

# Get DEM array xloc and yloc
def get_dem_xv_yv_(arr, resolution, visualize=True):
    sz = arr.shape[0]
    total_width = resolution * sz
    x = np.linspace(0, total_width, sz)
    y = np.linspace(0, total_width, sz)
    xv, yv = np.meshgrid(x, y)
    if visualize == True:
        plt.contourf(xv/1000, yv/1000, arr/1000)
        print("minimal elevation:", np.min(arr/1000), "maximum elevation:", np.max(arr/1000))
        plt.axis("scaled")
        plt.colorbar()
        plt.show()
    return xv/1000, yv/1000, arr/1000

    
# Construct grid
def get_array_neighbors_(x, y, left=0, right=500, top=0, bottom=500, radius=1):
    temp = [(x - radius, y), (x + radius, y), (x, y - radius), (x, y + radius)]
    neighbors = temp.copy()

    for val in temp:
        if val[0] < left or val[0] >= right:
            neighbors.remove(val)
        elif val[1] < top or val[1] >= bottom:
            neighbors.remove(val)

    return neighbors


# External use ok
def construct_nx_graph(xv, yv, elevation):
    sz = elevation.shape[0]
    counts = np.reshape(np.arange(0, elevation.shape[0]*elevation.shape[1]), (elevation.shape[0], elevation.shape[1]))
    G = nx.Graph()

    node_features = []

    for i in range(0, elevation.shape[0]):
        for j in range(0, elevation.shape[1]):
            idx1 = counts[i, j]
            G.add_node(idx1)
            node_features.append(np.array([xv[i, j], yv[i, j], elevation[i, j]]))

    for i in range(0, elevation.shape[0]):
        for j in range(0, elevation.shape[1]):
            neighbors = get_array_neighbors_(i, j, right=elevation.shape[0], bottom=elevation.shape[1])
            for n in neighbors:
                p1 = np.array([xv[i, j], yv[i, j], elevation[i, j]])
                p2 = np.array([xv[n[0], n[1]], yv[n[0], n[1]], elevation[n[0], n[1]]])
                w = np.linalg.norm(p1 - p2, ord=1)
                idx2 = counts[n[0], n[1]]
                idx1 = counts[i, j]
                G.add_edge(idx1, idx2, weight=w)
    return G, node_features

def get_patches_(xv, yv, dem_array, patch_size, overlap):
    patches = []
    patch_graphs = []
    patch_features = []
    for i in trange(0, dem_array.shape[0] , patch_size - overlap):
        for j in range(0, dem_array.shape[1], patch_size- overlap):
            xv_patch = xv[i:i + patch_size, j:j+patch_size]
            yv_patch = yv[i:i+patch_size, j : j+patch_size]
            patch = dem_array[i : i + patch_size, j : j + patch_size].copy()
            graph, node_features = construct_nx_graph(xv_patch, yv_patch, patch)
            # print(patch.shape)
            # print(list(nx.selfloop_edges(graph)))
            patches.append(patch)
            patch_graphs.append(graph)
            patch_features.append(node_features)

    return patches, patch_graphs, patch_features

def get_edge_index(G):
    weights = []

    edges = [[], []]

    print("Formatting edge index.......")
    for e in tqdm(G.edges(data=True)):
        edges[0].append(e[0])
        edges[1].append(e[1])
        edges[0].append(e[1])
        edges[1].append(e[0])

        weights.append(e[2]['weight'])
        weights.append(e[2]['weight'])
    return edges, weights

# For each graph, sample n number of 
def construct_patch_dataset(patches, patch_graphs, patch_features, per_graph=10):
    all_data = []
    print(len(patches))
    for patch in range(len(patches)):
        nx_graph = patch_graphs[patch]
        edge_index, weights = get_edge_index(patch_graphs[patch])
        pf = torch.tensor(patch_features[patch])
        edge_index = torch.tensor(edge_index)
        weights = torch.tensor(weights)
        for i in trange(per_graph):
            src, tar = np.random.choice(len(patch_features[patch]), [2,], replace=False)
            shortest_path = nx.shortest_path_length(nx_graph, src, tar, weight='weight')
            data=TerrainPatchesData(x=pf, edge_index = edge_index, edge_attr=weights, src=src, tar=tar, length=shortest_path)
            all_data.append(data)
        
    return all_data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw-data', type=str)
    parser.add_argument('--filename', type=str) # saves should be named `gr-{graph-resolution}-ps-{patch-size}-ol-{overlap}`
    parser.add_argument('--graph-resolution', type=int)
    parser.add_argument('--per-graph', type=int)
    parser.add_argument('--patch-size', type=int)
    parser.add_argument('--overlap', type=int)

    args = parser.parse_args()

    if 'norway' in args.raw_data:
        dem_res = DATASET_INFO['norway'][0]
        imperial = DATASET_INFO['norway'][1]
    elif 'holland' in args.raw_data:
        dem_res = DATASET_INFO['holland'][0]
        imperial = DATASET_INFO['holland'][1]
    elif 'phil' in args.raw_data:
        dem_res = DATASET_INFO['phil'][0]
        imperial = DATASET_INFO['phil'][1]
    
    dem_array = load_dem_data_(args.raw_data, imperial)
    xv, yv, elevations = get_dem_xv_yv_(dem_array, dem_res)
    xv = xv[::args.graph_resolution, ::args.graph_resolution]
    yv = yv[::args.graph_resolution, ::args.graph_resolution]
    elevations = elevations[::args.graph_resolution, ::args.graph_resolution]
    print("DEM array shape", elevations.shape)

    patches, patch_graphs, patch_features = get_patches_(xv, yv, elevations,args.patch_size, args.overlap)
    all_data = construct_patch_dataset(patches, patch_graphs, patch_features, args.per_graph)
    torch.save(all_data, args.filename+'.pt')
    print("Number of patches:", len(patch_graphs))
    print("Number of samples:", args.per_graph * len(patch_graphs))

if __name__ == '__main__':
    main()