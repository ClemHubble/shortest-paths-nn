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

DATASET_INFO = {'norway': [10, False], 'phil': [3, True], 'holland': [1.524, True], 'la': [28.34, False]}

# Load DEM data from file, 
# outputs elevations in meters
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
# outputs all relevant values in km
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
    return xv, yv, arr
    #return xv/1000, yv/1000, arr/1000

    
# Construct grid
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
def construct_nx_graph(xv, yv, elevation, triangles=False, p=2):
    
    n = elevation.shape[0]
    m = elevation.shape[1]
    print("shape", n, m)
    counts = np.reshape(np.arange(0, n*m), (n, m))
    G = nx.Graph()

    node_features = []
    #fig = plt.figure()
    #ax = fig.add_subplot(projection='3d')
    for i in trange(0, n):
        for j in range(0, m):
            idx1 = counts[i, j]
            G.add_node(idx1)
            node_features.append(np.array([xv[i, j], yv[i, j], elevation[i, j]]))
            neighbors = get_array_neighbors_(i, j, right=elevation.shape[0], radius=1)
            for neighbor in neighbors:
                p1 = np.array([xv[i, j], yv[i, j], elevation[i, j]])
                p2 = np.array([xv[neighbor[0], neighbor[1]], yv[neighbor[0], neighbor[1]], elevation[neighbor[0], neighbor[1]]])
                w = np.linalg.norm(p1 - p2, ord=p)
                #ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color='black')
                idx2 = counts[neighbor[0], neighbor[1]]
                G.add_edge(idx1, idx2, weight=w)
    print("Size of graph:", len(node_features))
    if triangles:
        for i in trange(0, n - 1):
            for j in range(0, m - 1):
                # index cell by top left coordinate
                triangle_edge = [(counts[i, j], counts[i + 1, j + 1]), (counts[i + 1, j], counts[i, j + 1])]
                edge = triangle_edge[np.random.choice(2)]
                p1 = node_features[edge[0]]
                p2 = node_features[edge[1]]
                w = np.linalg.norm(p1 - p2, ord = p)
                #ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color='black')
                G.add_edge(edge[0], edge[1], weight=w)
    #fig.savefig('../images/norway-250.png')
    return G, node_features

def to_pyg_graph(G):
    distances = []

    edges = [[], []]

    print("Formatting edge index.......")
    for e in tqdm(G.edges(data=True)):
        edges[0].append(e[0])
        edges[1].append(e[1])
        edges[0].append(e[1])
        edges[1].append(e[0])

        distances.append(e[2]['weight'])
        distances.append(e[2]['weight'])
    return edges, distances
    
def generate_probabilities(N, m):
    all_pairs = list(itertools.combinations(range(N), 2))
    probabilities = []
    for src, tar in tqdm(all_pairs):
        hops = abs(src//m - tar//m) + abs(src % m - tar % m )
        probabilities.append(1/(hops**2) if hops > 0 else 1)
    return all_pairs, probabilities

def construct_pyg_dataset(G, node_features, filename, size=100, distance_based=False, m=10):
    Nodes = np.sort(list(G.nodes()))

    edges, distances = to_pyg_graph(G)
    
    srcs = []
    tars = []
    lengths = []
    print("Generating shortest paths......")
    #jobs = []
    #pool = mp.Pool(processes=20)
    
    lst= np.arange(len(node_features))
    for i in trange(size):
        if distance_based:
            src = np.random.choice(len(node_features))
            hops = abs(src//m - lst//m) + abs(src % m - lst % m)+ 1
            probs = 1/hops
            probs = probs/np.linalg.norm(probs, ord=1)
            tar = np.random.choice(len(node_features), p = probs)
        else:
            src, tar = np.random.choice(len(node_features), [2, ], replace=False)
        length = nx.shortest_path_length(G, src, tar, weight='weight')
        srcs.append(src)
        tars.append(tar)
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
    parser.add_argument('--name', type=str)
    parser.add_argument('--raw-data', type=str)
    parser.add_argument('--filename', type=str)
    parser.add_argument('--graph-resolution', type=int)
    parser.add_argument('--dataset-size', type=int)
    parser.add_argument('--distance-based-sampling', action='store_true')
    parser.add_argument('--triangles', action='store_true')

    args = parser.parse_args()
    dem_res = DATASET_INFO[args.name][0]
    imperial = DATASET_INFO[args.name][1]
    if args.name == 'la':
        dem_array = np.load(args.raw_data)
    else:
        dem_array = load_dem_data_(args.raw_data, imperial)
    xv, yv, elevations = get_dem_xv_yv_(dem_array, dem_res)
    #row,col = np.random.choice(elevations.shape[0], size=[2,])
    # row = 122 
    # col = 1647
    # xv = xv[row:row+ 100, col:col+100]
    # yv = yv[row:row+100, col:col+100]
    # elevations = elevations[row:row+100, col:col+100]
    xv = xv[::args.graph_resolution, ::args.graph_resolution]
    yv = yv[::args.graph_resolution, ::args.graph_resolution]
    elevations = elevations[::args.graph_resolution, ::args.graph_resolution]
    print('terrain shape:', elevations.shape)

    G, node_features = construct_nx_graph(xv, yv, elevations, triangles=args.triangles)

    construct_pyg_dataset(G, 
                          node_features, 
                          args.filename, 
                          size=args.dataset_size, 
                          distance_based=args.distance_based_sampling,
                          m = elevations.shape[1])


if __name__ == '__main__':
    main()