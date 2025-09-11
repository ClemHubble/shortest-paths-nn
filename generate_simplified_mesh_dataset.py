"""
Dataset Generation Script

This script processes vertex and edge files into npz datasets
for training and testing graph models.

Acknowledgment:
Special thanks to Caris for her contributions and support in developing this script.
"""
import numpy as np
import networkx as nx
from sklearn.model_selection import train_test_split
import argparse

def generate_dataset(vertex_file, edge_file, output_prefix):
    raw_vertices = np.loadtxt(vertex_file)
    vertex_ids = raw_vertices[:, 0].astype(int)
    coordinates = raw_vertices[:, 1:4]

    id_to_index = {vid: idx for idx, vid in enumerate(vertex_ids)}

    node_features = coordinates

    raw_edges = np.loadtxt(edge_file, dtype=int)

    remapped_edges = []
    for v1, v2 in raw_edges:
        if v1 in id_to_index and v2 in id_to_index:
            remapped_edges.append([id_to_index[v1], id_to_index[v2]])

    remapped_edges = np.array(remapped_edges).T
    num_vertices = node_features.shape[0]

    distances = []
    for src, tgt in remapped_edges.T:
        p1 = node_features[src]
        p2 = node_features[tgt]
        dist = np.linalg.norm(p1 - p2)
        distances.append(dist)
    distances = np.array(distances)

    G = nx.Graph()
    for (src, tgt, dist) in zip(remapped_edges[0], remapped_edges[1], distances):
        G.add_edge(int(src), int(tgt), weight=dist)

    srcs = []
    tars = []
    lengths = []
    node_indices = list(G.nodes)

    TRAIN_SIZE = 50000
    N = int(TRAIN_SIZE / 0.8)
    for _ in range(N):
        src, tar = np.random.choice(node_indices, size=2, replace=False)
        try:
            length = nx.shortest_path_length(G, source=src, target=tar, weight='weight')
            srcs.append(src)
            tars.append(tar)
            lengths.append(length)
        except nx.NetworkXNoPath:
            continue

    srcs = np.array(srcs)
    tars = np.array(tars)
    lengths = np.array(lengths)

    np.savez(f"{output_prefix}.npz",
             edge_index=remapped_edges,
             distances=distances,
             node_features=node_features,
             srcs=srcs,
             tars=tars,
             lengths=lengths)

    print(f"Saved full dataset: {output_prefix}.npz")

    srcs_train, srcs_test, tars_train, tars_test, lengths_train, lengths_test = train_test_split(
        srcs, tars, lengths, test_size=0.2, random_state=42
    )

    np.savez(f"{output_prefix}_train.npz",
             edge_index=remapped_edges,
             distances=distances,
             node_features=node_features,
             srcs=srcs_train,
             tars=tars_train,
             lengths=lengths_train)

    np.savez(f"{output_prefix}_test.npz",
             edge_index=remapped_edges,
             distances=distances,
             node_features=node_features,
             srcs=srcs_test,
             tars=tars_test,
             lengths=lengths_test)

    print(f"Saved:\nTrain -> {output_prefix}_train.npz\nTest  -> {output_prefix}_test.npz")
    print(f"Train size: {len(srcs_train)}, Test size: {len(srcs_test)}")

if __name__ == "__main__":
    import traceback
    parser = argparse.ArgumentParser()
    parser.add_argument("--vertex_file", required=True, help="Path to simplified_vertices.txt")
    parser.add_argument("--edge_file", required=True, help="Path to simplified_edges.txt")
    parser.add_argument("--output_prefix", required=True, help="Prefix for output files")

    args = parser.parse_args()
    try:
        generate_dataset(args.vertex_file, args.edge_file, args.output_prefix)
    except Exception as e:
        print("[ERROR] generate_dataset failed:", e)
        traceback.print_exc()
