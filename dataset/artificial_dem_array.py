import numpy as np 
import networkx as nx
from tqdm import tqdm, trange
import argparse
import matplotlib.pyplot as plt


def gaussian_2d(xv, yv, amplitude=1, center_x=0, center_y=0, sigma_x=1, sigma_y=1):
    z1 = (xv - center_x)**2/(2*(sigma_x**2))
    z2 = (yv - center_y)**2/(2*(sigma_y**2))
    z = amplitude * np.exp(-(z1 + z2))
    return z

def create_elevation_array(n, k=10):
    '''
    Function which creates an n x n elevation array with k critical points

    Returns: numpy array of elevations
    '''
    n0 = 10*n
    # create x and y range
    x = np.linspace(-10, 10, n0)
    y = np.linspace(-10, 10, n0)
    xv, yv = np.meshgrid(x, y)

    z_out = np.zeros((n0, n0))
    centers = [(50, 50), (200, 50), (50, 150), (250, 150), (130, 230)]
    for i in range(k):
        center = centers[i]
        x_c = xv[center[0], center[1]]
        y_c = yv[center[0], center[1]]
        centers.append(center)
        #a = np.random.uniform(low=1.0, high=5.0)
        a = 3.0
        # s_x = np.random.uniform(low=1.0, high=4.0)
        # s_y = np.random.uniform(low=1.0, high=4.0)
        s_x = 2.0
        s_y = 2.0
        z = gaussian_2d(xv, yv, amplitude=a, center_x = x_c, center_y = y_c, sigma_x=s_x, sigma_y=s_y)
        z_out += z
    return  z_out

def get_array_neighbors_(x, y, left=0, right=500, radius=1):
    temp = [(x - radius, y), (x + radius, y), (x, y - radius), (x, y + radius)]
    neighbors = temp.copy()

    for val in temp:
        if val[0] < left or val[0] >= right:
            neighbors.remove(val)
        elif val[1] < left or val[1] >= right:
            neighbors.remove(val)

    return neighbors

def construct_nx_graph(xv, yv, elevation, save_img=None):
    sz = elevation.shape[0]
    counts = np.reshape(np.arange(0, sz*sz), (sz, sz))
    G = nx.Graph()

    node_features = []
    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(projection='3d')

    for i in trange(0, len(elevation)):
        for j in range(0, len(elevation)):
            idx1 = counts[i, j]
            G.add_node(idx1)
            node_features.append(np.array([xv[i, j], yv[i, j], elevation[i, j]]))
            neighbors = get_array_neighbors_(i, j, right=elevation.shape[0])
            for n in neighbors:
                p1 = np.array([xv[i, j], yv[i, j], elevation[i, j]])
                p2 = np.array([xv[n[0], n[1]], yv[n[0], n[1]], elevation[n[0], n[1]]])
                w = np.linalg.norm(p1 - p2)
                if save_img != None:
                    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color='black')
                idx2 = counts[n[0], n[1]]
                G.add_edge(idx1, idx2, weight=w)
    print("Size of graph:", len(node_features))
    if save_img != None:
        print("saved in:", save_img)
        plt.savefig(save_img)
    return G, node_features

def get_elevated_points(node_features, threshhold=0.4):
    elevated_pts = []
    for i in range(len(node_features)):
        z = node_features[i][2]
        if z > 0.4:
            elevated_pts.append(i)
    return elevated_pts

# at least guarantee_rough_path percent of the dataset should be go through "elevated points"
def construct_pyg_dataset(G, node_features, filename, guarantee_rough_path = 0.2, size=100):
    Nodes = np.sort(list(G.nodes()))

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
    
    # Get elevated points
    elevated_pts = get_elevated_points(node_features, threshhold=0.6)

    srcs = []
    tars = []
    lengths = []
    print("Generating shortest paths......")
    for i in range(len(Nodes)):
        for j in range(i + 1, len(Nodes)):
            src = i
            tar = j
            srcs.append(i)
            tars.append(j)
            length = nx.shortest_path_length(G, src, tar, weight='weight')
            lengths.append(length)

    # for i in trange(size):
    #     if i < size*guarantee_rough_path and len(elevated_pts) > 0:
    #         src = np.random.choice(elevated_pts)
    #         tar = np.random.choice(len(node_features), replace=False)
    #     else:
    #         src, tar = np.random.choice(len(node_features), [2,], replace=False)
    #     srcs.append(src)
    #     tars.append(tar)
    #     length = nx.shortest_path_length(G, src, tar, weight='weight')
    #     lengths.append(length)
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
    parser.add_argument("--size", type=int)
    parser.add_argument("--train-dataset-size", type=int)
    parser.add_argument("--test-dataset-size", type=int)
    args = parser.parse_args()

    for k in range(1):
        
        upsample_save = f'/data/sam/terrain/data/artificial/for-lucas.npy'
        xv, yv, dem, upsampled_dem = create_elevation_array(n=args.size, k=5)
        np.save(upsample_save, upsampled_dem)
        img = f'../images/small-k-{k}.png'
        print("saved in", img)
        # G, node_features = construct_nx_graph(xv, yv, dem, save_img=img)
        # #for train_dataset_size in range(10000, 60000, 10000):
        # train_filename = f'/data/sam/terrain/data/artificial/small-k-{k}-train-full.npz'
        # construct_pyg_dataset(G, node_features, filename=train_filename, size=10)
        # test_filename = f'/data/sam/terrain/data/artificial/small-k-{k}-test-{args.test_dataset_size}.npz'

        
        # construct_pyg_dataset(G, node_features, filename=test_filename, size=args.test_dataset_size)
    return 0

if __name__=="__main__":
    main()