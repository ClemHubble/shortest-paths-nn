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

DATASET_INFO = {'norway': [10, False], 'phil': [3, True], 'holland': [1.524, True], 'la': [28.34, False], 'artificial': [10/50, False]}

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

def mesh_to_graph(edge_filename, vertex_filename):
    f = open(edge_filename)
    all_vertices = []
    lines = f.readlines()
    edges = []
    for i in range(len(lines)):

        vals = lines[i].split()
        edges.append((int(vals[0]), int(vals[1])))
        all_vertices.append(int(vals[0]))
        all_vertices.append(int(vals[1]))
    
    unique_vertices = np.sort(np.unique(all_vertices))

    nx_graph = nx.Graph()

    temp = {}
    for i in range(len(unique_vertices)):
        temp[unique_vertices[i]] = i    
    
    f = open(vertex_filename)

    lines = f.readlines()
    vertices = np.zeros((len(unique_vertices), 3))
    for i in range( len(lines)):
        vals = lines[i].split()
        vertices[i] = [ float(vals[0])/1000, float(vals[1])/1000, float(vals[2])/1000]

    for i in range(len(edges)):
        v1 = temp[edges[i][0]]
        v2 = temp[edges[i][1]]
        weight = np.linalg.norm(vertices[v1] - vertices[v2], ord=2)
        nx_graph.add_edge(v1, v2, weight=weight)

    return nx_graph, vertices

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
        print("minimal elevation:", np.min(arr), "maximum elevation:", np.max(arr))
        plt.axis("scaled")
        plt.colorbar()
        plt.show()
    #return xv, yv, arr
    return xv/1000, yv/1000, arr/1000

    
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
def construct_nx_graph(xv, yv, elevation, triangles=False, p=2, scale=False):
    
    n = elevation.shape[0]
    m = elevation.shape[1]
    print("shape", n, m)
    counts = np.reshape(np.arange(0, n*m), (n, m))
    G = nx.Graph()

    print(triangles)

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
                if scale:
                    val = abs(np.random.normal(1.0, 1.0))
                    slope = (abs(p1[2] - p2[2]))/(abs(p1[0] - p2[0]) + abs(p1[1] - p2[1]))
                    angle_of_elevation = np.abs(np.arctan(p1[2] - p2[2])/np.linalg.norm(p2[:2] - p1[:2], ord=2))
                    val = angle_of_elevation
                    w = (1 + val) * np.linalg.norm(p1 - p2, ord=p)
                else:
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
                for edge in triangle_edge:
                    p1 = node_features[edge[0]]
                    p2 = node_features[edge[1]]
                    if scale:
                        angle_of_elevation = np.abs(np.arctan(p1[2] - p2[2])/np.linalg.norm(p2[:2] - p1[:2], ord=2))
                        slope = (abs(p1[2] - p2[2]))/(abs(p1[0] - p2[0]) + abs(p1[1] - p2[1]))
                        val = angle_of_elevation
                        w = (1 + val) * np.linalg.norm(p1 - p2, ord=p)
                    else:
                        w = np.linalg.norm(p1 - p2, ord = p)
                    G.add_edge(edge[0], edge[1], weight=w)
    #fig.savefig('../images/norway-250.png')
    print(G.edges(0))
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

def construct_pyg_dataset(G, node_features, filename, size=100, sampling_technique='distance-based', m=10, p=0.10):
    Nodes = np.sort(list(G.nodes()))

    edges, distances = to_pyg_graph(G)
    
    srcs = []
    tars = []
    lengths = []
    print("Generating shortest paths......")
    #jobs = []
    #pool = mp.Pool(processes=20)
    node_idxs = np.reshape(np.arange(m * m), (m, m))
    lst= np.arange(len(node_features))
    print(sampling_technique)
    for i in trange(size):
        if sampling_technique == 'distance-based':
            src = np.random.choice(len(node_features))
            hops = abs(src//m - lst//m) + abs(src % m - lst % m)+ 1
            probs = 1/hops
            probs = probs/np.linalg.norm(probs, ord=1)
            tar = np.random.choice(len(node_features), p = probs)
        elif sampling_technique == 'constrained-125x125':
            src = np.random.choice(len(node_features))
            src_row = src//m
            src_col = src %m 
            if np.random.uniform(low=0.0, high=1.0) <= p:
                tar = np.random.choice(len(node_features))
            else:
                b1 = 0 if src_row -125  < 0 else src_row - 125
                b2 = 0 if src_col - 125 < 0 else src_col - 125
                #print(b1, src_row + 25, b2, src_col + 25, node_idxs[b1 : src_row + 25, b2: src_col+25].flatten())
                tar = np.random.choice(node_idxs[b1 : src_row + 125, b2: src_col+125].flatten())
        elif sampling_technique == 'constrained-25x25':
            src = np.random.choice(len(node_features))
            src_row = src//m
            src_col = src %m 
            p = np.random.uniform(low=0.0, high=1.0)
            if p > 1.0:
                tar = np.random.choice(len(node_features))
            else:
                b1 = 0 if src_row -25  < 0 else src_row - 25
                b2 = 0 if src_col - 25 < 0 else src_col - 25
                tar = np.random.choice(node_idxs[b1 : src_row + 25, b2: src_col+25].flatten())
        elif sampling_technique == 'ss-random':
            n_srcs = 1000
            num_tars = size // n_srcs
            if 'test' in filename:
                src_nodes = np.random.choice(len(node_features), size=1000)
                tars = []
                srcs = []
                lengths = []
                for s in tqdm(src_nodes):
                    shortest_paths = nx.single_source_dijkstra_path_length(G, s, weight='weight', cutoff=20)
                    for tar in shortest_paths:
                        srcs.append(s)
                        tars.append(tar)
                        lengths.append(shortest_paths[tar])
            else:
                src_nodes = np.random.choice(len(node_features), size=n_srcs)
                num_nodes = len(node_features)
                print("number of nodes", num_nodes)
                tars = []
                srcs = []
                lengths = []

                for s in tqdm(src_nodes):
                    shortest_paths = nx.single_source_dijkstra_path_length(G, s, weight='weight')
                    for i in range(num_tars):
                        t = np.random.choice(num_nodes)
                        srcs.append(s)
                        tars.append(t)
                        lengths.append(shortest_paths[t])

            break
        else:
            src, tar = np.random.choice(len(node_features), [2, ], replace=False)
        if sampling_technique != 'ss-random':
            length = nx.shortest_path_length(G, src, tar, weight='weight')
            srcs.append(src)
            tars.append(tar)
            lengths.append(length)
    # rotation = np.array([[np.cos(np.pi/9), -np.sin(np.pi/9)], [np.sin(np.pi/9), np.cos(np.pi/9)]])
    # node_features = np.array(node_features)
    # rotated_pts_x_y = (rotation @ node_features[:, :2].T).T
    # node_features[:, :2] = rotated_pts_x_y
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
    parser.add_argument('--sampling-technique', type=str, default='random')
    parser.add_argument('--triangles', action='store_true')
    parser.add_argument('--edge-weight', action='store_true')
    parser.add_argument('--change-heights', action='store_true')

    args = parser.parse_args()
    dem_res = DATASET_INFO[args.name][0]
    imperial = DATASET_INFO[args.name][1]
    if 'meshes' in args.raw_data:
        edge_filename = os.path.join(args.raw_data, 'percent_edges')
        vertex_filename = os.path.join(args.raw_data, 'percent_vertices')
        G, node_features = mesh_to_graph(edge_filename, vertex_filename)
        m = 10
    else:
        if args.name == 'la' or args.name == 'artificial':
            dem_array = np.load(args.raw_data)
        else:
            dem_array = load_dem_data_(args.raw_data, imperial)
        xv, yv, elevations = get_dem_xv_yv_(dem_array, dem_res)
        row,col = np.random.choice(elevations.shape[0], size=[2,])
        res = args.graph_resolution
        
        filename = args.filename
        
        xv_n = xv[::res, ::res]
        yv_n = yv[::res, ::res]
        elevations_n = elevations[::res, ::res]
        print(np.min(elevations_n))
        m = elevations_n.shape[1]
        print('terrain shape:', elevations_n.shape)
        print('resolution:', res)
        if args.change_heights:
            for k in range(1, 60):
                filename = f'/data/sam/terrain/data/{args.name}/uncertainty/50/50k-{k}.npz'
                uncertainty = np.random.uniform(-0.050, 0.050, size=elevations_n.shape)
                elevations_n = uncertainty + elevations_n
                
                G, node_features = construct_nx_graph(xv_n, yv_n, elevations_n, triangles=args.triangles, scale=args.edge_weight)
                sz = args.dataset_size
                sampling_technique = args.sampling_technique
                
                construct_pyg_dataset(G, 
                                    node_features, 
                                    filename, 
                                    size=args.dataset_size, 
                                    sampling_technique=sampling_technique,
                                    m = 10)
        else:
            G, node_features = construct_nx_graph(xv_n, yv_n, elevations_n, triangles=args.triangles, scale=args.edge_weight)
            filename = args.filename
            sz = args.dataset_size
            sampling_technique = args.sampling_technique
            
            construct_pyg_dataset(G, 
                                node_features, 
                                filename, 
                                size=args.dataset_size, 
                                sampling_technique=sampling_technique,
                                m = 10)


if __name__ == '__main__':
    main()