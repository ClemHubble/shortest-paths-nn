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

DATASET_INFO = {'norway': [10, False], 
                'phil': [3, True], 
                'holland': [1.524, True], 
                'la': [28.34, False], 
                'artificial': [10/50, False]}

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

    
# Construct grid with both cross edges
def get_array_neighbors_(x, y, left=0, right=500, radius=1):
    temp = [(x - radius, y), 
            (x + radius, y), 
            (x, y - radius), 
            (x, y + radius), 
            (x - radius, y - radius), 
            (x - radius, y + radius), 
            (x + radius, y - radius), 
            (x+radius, y + radius)]
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
    print("Number of nodes:", len(node_features))
    print("Number of edges:", G.number_of_edges())
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


def construct_dataset(G,
                      node_features, 
                      filename, 
                      sampling_method, 
                      num_srcs, 
                      samples_per_source, 
                      rows=100, 
                      cols=100, 
                      threshhold=0.2):
    edges, distances = to_pyg_graph(G)
    
    if sampling_method == 'single-source-random':
        src_nodes = np.random.choice(len(node_features), size=num_srcs)
        sampling_fn = random_sampling
    elif sampling_method == 'critical-point-source':
        node_features = np.array(node_features)
        c1 = node_features[:, 0].reshape(rows, cols)
        c2 = node_features[:, 1].reshape(rows, cols)
        c3 = node_features[:, 2].reshape(rows, cols)
        terrain = torch.tensor(np.stack([c1, c2, c3]), dtype=torch.float)
        terrain = np.transpose(terrain, (1, 2, 0))
        print(terrain.size())
        src_nodes = find_critical_points(terrain, threshhold)
        sampling_fn = random_sampling
    elif sampling_method == 'distance-based':
        src_nodes = np.random.choice(len(node_features), size=num_srcs)
        sampling_fn = distance_based
    else:
        raise NotImplementedError("please choose between 'single-source-random', 'critical-point-source', 'distance-based'")
    srcs = []
    tars = []
    lengths = []
    print("Number of source nodes:", len(src_nodes))
    print("Generating shortest path distances.....")
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
    parser.add_argument('--name', type=str)
    parser.add_argument('--raw-data', type=str)
    parser.add_argument('--filename', type=str)
    parser.add_argument('--graph-resolution', type=int)
    parser.add_argument('--dataset-size', type=int)
    parser.add_argument('--num-sources', type=int)
    parser.add_argument('--sampling-technique', type=str, default='random')
    parser.add_argument('--triangles', action='store_true')
    parser.add_argument('--edge-weight', action='store_true')
    parser.add_argument('--change-heights', action='store_true')
    parser.add_argument('--critical-point-threshhold', type=float, default=0.2)

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
        print('terrain shape:', elevations_n.shape)
        print('resolution:', res)
        if args.change_heights:
            for k in range(1, 60):
                filename = f'/data/sam/terrain/data/{args.name}/uncertainty/50/50k-{k}.npz'
                uncertainty = np.random.uniform(-0.050, 0.050, size=elevations_n.shape)
                elevations_n = uncertainty + elevations_n
                
                G, node_features = construct_nx_graph(xv_n, 
                                                      yv_n, 
                                                      elevations_n, 
                                                      triangles=args.triangles, 
                                                      scale=args.edge_weight)
                sz = args.dataset_size
                sampling_technique = args.sampling_technique
                
                construct_dataset(G = G, 
                                  node_features = node_features, 
                                  filename = filename, 
                                  num_srcs = args.num_sources,
                                  samples_per_source = args.dataset_size//args.num_sources,
                                  sampling_method=sampling_technique,
                                  rows = elevations_n.shape[0],
                                  cols = elevations_n.shape[1], 
                                  threshhold = args.critical_point_threshhold)
        else:
            G, node_features = construct_nx_graph(xv_n, 
                                                  yv_n, 
                                                  elevations_n, 
                                                  triangles=args.triangles, 
                                                  scale=args.edge_weight)
            filename = args.filename
            sz = args.dataset_size
            sampling_technique = args.sampling_technique
            
            construct_dataset(G = G, 
                                  node_features = node_features, 
                                  filename = filename, 
                                  num_srcs = args.num_sources,
                                  samples_per_source = args.dataset_size//args.num_sources,
                                  sampling_method=sampling_technique,
                                  rows = elevations_n.shape[0],
                                  cols = elevations_n.shape[1], 
                                  threshhold = args.critical_point_threshhold)


if __name__ == '__main__':
    main()