import numpy as np
import torch, queue
from torch_geometric.data import Data
from utils import gen_edge_index
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm, trange

import argparse

DATASET_INFO = {'norway': [10, False], 'phil': [3, True], 'holland': [1.524, True]}

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
        print(np.min(arr/1000))
        plt.axis("scaled")
        plt.colorbar()
        plt.show()
    return xv/1000, yv/1000, arr/1000

    
# Construct triangulation
def get_array_neighbors_(x, y, left=0, right=500, radius=1):
    temp = [(x - radius, y), (x + radius, y), (x, y - radius), (x, y + radius)]
    neighbors = temp.copy()

    for val in temp:
        if val[0] < left or val[0] >= right:
            neighbors.remove(val)
        elif val[1] < left or val[1] >= right:
            neighbors.remove(val)

    return neighbors

# External use ok
def construct_nx_graph(xv, yv, elevation, res):
    sz = elevation.shape[0]//res
    counts = np.reshape(np.arange(0, sz*sz), (sz, sz))
    G = nx.Graph()

    node_features = []

    for i in trange(0, len(elevation),res):
        for j in range(0, len(elevation), res):
            idx1 = counts[i//res, j//res]
            G.add_node(idx1)
            node_features.append(np.array([xv[i, j], yv[i, j], elevation[i, j]]))
            neighbors = get_array_neighbors_(i, j, right=elevation.shape[0], radius=res)
            for n in neighbors:
                p1 = np.array([xv[i, j], yv[i, j], elevation[i, j]])
                p2 = np.array([xv[n[0], n[1]], yv[n[0], n[1]], elevation[n[0], n[1]]])
                w = np.linalg.norm(p1 - p2)
                idx2 = counts[n[0]//res, n[1]//res]
                G.add_edge(idx1, idx2, weight=w)
    print("Size of graph:", len(node_features))
    return G, node_features

def construct_pyg_dataset(G, node_features, filename, size=100):
    Nodes = np.sort(list(G.nodes()))

    distances = []

    edges = [[], []]

    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(projection='3d')
    print("Formatting edge index.......")
    for e in tqdm(G.edges(data=True)):
        edges[0].append(e[0])
        edges[1].append(e[1])
        edges[0].append(e[1])
        edges[1].append(e[0])
        p1 = node_features[e[0]]
        p2= node_features[e[1]]

        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color='black')
        distances.append(e[2]['weight'])
        distances.append(e[2]['weight'])
    
    srcs = []
    tars = []
    lengths = []
    print("Generating shortest paths......")
    #size = len(Nodes)
    # for i in trange(size):
    #     for j in range(i+1, size):
    #         #src, tar = np.random.choice(len(node_features), [2,], replace=False)
    #         src = i
    #         tar = j
    #         srcs.append(src)
    #         tars.append(tar)
    #         length = nx.shortest_path_length(G, src, tar, weight='weight')
    #         lengths.append(length)
    for i in trange(size):
        src, tar = np.random.choice(len(node_features), [2,], replace=False)
        srcs.append(src)
        tars.append(tar)
        length = nx.shortest_path_length(G, src, tar, weight='weight')
        lengths.append(length)
    print("Saved dataset in:", filename)
    np.savez(filename, 
         edge_index = edges, 
         distances=distances, 
         nodes=Nodes,
         srcs = srcs,
         tars = tars,
         lengths = lengths,
         node_features=node_features)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw-data', type=str)
    parser.add_argument('--filename', type=str)
    parser.add_argument('--graph-resolution', type=int)
    parser.add_argument('--dataset-size', type=int)

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

    G, node_features = construct_nx_graph(xv, yv, elevations, args.graph_resolution)

    construct_pyg_dataset(G, node_features, args.filename, size=args.dataset_size)


if __name__ == '__main__':
    main()