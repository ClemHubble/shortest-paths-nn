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

DATASET_INFO = {'norway': [10, False], 'phil': [3, True], 'holland': [1.524, True], 'la': [28.34, False]}


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
    x_sz = arr.shape[0]
    total_width_x = resolution * x_sz
    y_sz = arr.shape[1]
    total_width_y = resolution * y_sz
    x = np.linspace(0, total_width_x, x_sz)
    y = np.linspace(0, total_width_y, y_sz)
    xv, yv = np.meshgrid(x, y, indexing='ij')
    if visualize == True:
        plt.contourf(xv/1000, yv/1000, arr/1000, origin='upper')
        print("minimum elevation:", np.min(arr/1000), "maximum elevation:", np.max(arr/1000))
        plt.axis("scaled")
        plt.colorbar()
        plt.savefig('la-county-contour')
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
def construct_nx_graph(xv, yv, elevation, triangles=False, p=2, scale=False):
    
    n = elevation.shape[0]
    m = elevation.shape[1]
    counts = np.reshape(np.arange(0, n*m), (n, m))
    G = nx.Graph()

    node_features = []
    #fig = plt.figure()
    #ax = fig.add_subplot(projection='3d')
    for i in range(0, n):
        for j in range(0, m):
            idx1 = counts[i, j]
            G.add_node(idx1)
            node_features.append(np.array([xv[i, j], yv[i, j], elevation[i, j]]))
            neighbors = get_array_neighbors_(i, j, right=elevation.shape[0], bottom=elevation.shape[1], radius=1)
            for neighbor in neighbors:
                p1 = np.array([xv[i, j], yv[i, j], elevation[i, j]])

                p2 = np.array([xv[neighbor[0], neighbor[1]], yv[neighbor[0], neighbor[1]], elevation[neighbor[0], neighbor[1]]])
                if scale:
                    slope = (abs(p1[2] - p2[2]))/(abs(p1[0] - p2[0]) + abs(p1[1] - p2[1]))
                    # w = np.log(1 + slope)
                    deg_angle = np.arctan(slope) * (180/np.pi)
                    w = np.power(deg_angle, 1.2)
                else:
                    w = np.linalg.norm(p1 - p2, ord=p)
                #ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color='black')
                idx2 = counts[neighbor[0], neighbor[1]]
                G.add_edge(idx1, idx2, weight=w)
    if triangles:
        for i in range(0, n - 1):
            for j in range(0, m - 1):
                # index cell by top left coordinate
                triangle_edge = [(counts[i, j], counts[i + 1, j + 1]), (counts[i + 1, j], counts[i, j + 1])]
                edge = triangle_edge[np.random.choice(2)]
                for edge in triangle_edge:
                    p1 = node_features[edge[0]]
                    p2 = node_features[edge[1]]
                    if scale:
                        slope = (abs(p1[2] - p2[2]))/(abs(p1[0] - p2[0]) + abs(p1[1] - p2[1]))
                        # w = np.log(1 + slope)
                        deg_angle = np.arctan(slope) * (180/np.pi)
                        w = np.power(deg_angle, 1.2)
                    else:
                        w = np.linalg.norm(p1 - p2, ord=p)
                #ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color='black')
                    G.add_edge(edge[0], edge[1], weight=w)
    #fig.savefig('../images/norway-250.png')
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

    for e in G.edges(data=True):
        edges[0].append(e[0])
        edges[1].append(e[1])
        edges[0].append(e[1])
        edges[1].append(e[0])

        weights.append(e[2]['weight'])
        weights.append(e[2]['weight'])
    return edges, weights

def construct_patch_dataset(xv, yv, dem_array, patch_size, sz, triangles=False, scale=False):
    all_data = []
    print("size of dataset:", sz, "patch sizes:", patch_size)
    n = dem_array.shape[0]
    m = dem_array.shape[1]
    nc = 10
    cx = np.random.choice(n-patch_size, size=nc, replace=False)
    cy = np.random.choice(m - patch_size, size=nc, replace=False)
    for i in range(nc):
        xr = cx[i]
        yr = cy[i]
        xv_patch = xv[xr: xr + patch_size, yr: yr+patch_size]
        yv_patch = yv[xr: xr + patch_size, yr: yr+patch_size]
        patch = dem_array[xr: xr + patch_size, yr: yr+patch_size]
        graph, node_features = construct_nx_graph(xv_patch, 
                                                  yv_patch, 
                                                  patch, 
                                                  triangles=triangles, 
                                                  scale=scale)

        edge_index, weights = get_edge_index(graph)
        for m in trange(patch_size * patch_size):
            for n in range(m + 1, patch_size * patch_size):
                shortest_path = nx.shortest_path_length(graph, m, n, weight='weight')
                data=TerrainPatchesData(x=node_features, 
                                        edge_index = edge_index, 
                                        edge_attr=weights, 
                                        src=m, 
                                        tar=n, 
                                        length=shortest_path)
                all_data.append(data)

    # for i in trange(sz):
    #     c = np.random.randint(low = 0, high=5)
    #     #c = 0
    #     xr = cx[c]
    #     yr = cy[c]
    #     xv_patch = xv[xr: xr + patch_size, yr: yr+patch_size]
    #     yv_patch = yv[xr: xr + patch_size, yr: yr+patch_size]
    #     patch = dem_array[xr: xr + patch_size, yr: yr+patch_size]
    #     graph, node_features = construct_nx_graph(xv_patch, 
    #                                               yv_patch, 
    #                                               patch, 
    #                                               triangles=triangles, 
    #                                               scale=scale)

    #     edge_index, weights = get_edge_index(graph)
        
    #     src, tar = np.random.choice(len(node_features), [2, ], replace=False)
    #     if src == tar:
    #         continue
    #     shortest_path = nx.shortest_path_length(graph, src, tar, weight='weight')
    #     data=TerrainPatchesData(x=node_features, 
    #                             edge_index = edge_index, 
    #                             edge_attr=weights, 
    #                             src=src, 
    #                             tar=tar, 
    #                             length=shortest_path)
    #     all_data.append(data)
    return all_data, np.hstack((cx, cy))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str)
    parser.add_argument('--raw-data', type=str)
    parser.add_argument('--filename', type=str) # saves should be named `gr-{graph-resolution}-ps-{patch-size}-ol-{overlap}`
    parser.add_argument('--graph-resolution', type=int)
    parser.add_argument('--patch-size', type=int)
    parser.add_argument('--dataset-size', type=int)
    parser.add_argument('--triangles', action='store_true')
    parser.add_argument('--scale', action='store_true')

    args = parser.parse_args()

    dem_res = DATASET_INFO[args.name][0]
    imperial = DATASET_INFO[args.name][1]

    if args.name == 'la':
        dem_array = np.load(args.raw_data)
    else:
        dem_array = load_dem_data_(args.raw_data, imperial)

    xv, yv, elevations = get_dem_xv_yv_(dem_array, dem_res)
    print('total elevation shape:', elevations.shape)
    xv = xv[::args.graph_resolution, :2000:args.graph_resolution]
    yv = yv[::args.graph_resolution, :2000:args.graph_resolution]
    elevations = elevations[::args.graph_resolution, :2000:args.graph_resolution]
    print("DEM array shape", elevations.shape)

    all_data, centers = construct_patch_dataset(xv, 
                                                yv, 
                                                elevations, 
                                                args.patch_size, 
                                                args.dataset_size,
                                                triangles=args.triangles, 
                                                scale=args.scale)
    torch.save(all_data, args.filename+'.pt')
    torch.save(centers, args.filename + '-centers.pt')

if __name__ == '__main__':
    main()