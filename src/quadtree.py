'''
Hierarchical tree algorithm for approximating shortest paths on Terrain grid graph
'''

import numpy as np
import networkx as nx
from treelib import Tree, Node
from tqdm import tqdm, trange

def build_dem_qt_(array, lb, rb, ub, db, tree, parent, thresh=0.10):
    
    mid_row = abs(lb - rb) //2 if abs(lb - rb) % 2 == 0 else (abs(lb - rb) + 1) //2
    mid_col = abs(ub - db) //2 if abs(ub - db) % 2 == 0 else (abs(ub - db) + 1) //2
    anchor = (lb - 1 + mid_row, ub - 1 + mid_col)
    if array.shape[0] * array.shape[1] == 0:
        return
    if parent == None:
        node = tree.create_node(anchor)
        node_id = node.identifier
    else:
        node = tree.create_node(anchor, parent = parent, data=(lb, rb, ub, db))
        node_id = node.identifier
    # base case: 
    if array.shape[0] * array.shape[1] <=1:
        return array
    if abs(np.min(array) - np.max(array)) <= thresh:
        return array

    q1 = array[:mid_row, :mid_col]
    q2 = array[:mid_row, mid_col:]
    q3 = array[mid_row:, mid_col:]
    q4 = array[mid_row:, :mid_col]


    # if any of the quadrants have too large of a difference between min and max
    # split it again along some midpoint
    build_dem_qt_(q1, lb, lb + mid_row, ub, ub + mid_col, tree, node_id, thresh=thresh)
    build_dem_qt_(q2, lb + mid_row, rb, ub, ub + mid_col, tree, node_id, thresh=thresh)
    build_dem_qt_(q3, lb + mid_row, rb, ub + mid_col, db, tree, node_id, thresh=thresh)
    build_dem_qt_(q4, lb, lb + mid_row, ub + mid_col, db, tree, node_id, thresh=thresh)
    
# parameters: array, threshhold value
# return: treelib.Tree object
def build_quadtree(array, thresh=0.10):
    tree = Tree()
    build_dem_qt_(array, 0, array.shape[0], 0, array.shape[1], tree, None, thresh=thresh)
    return tree

def quadtree_simplification(tree, num_rows, num_cols):
    graph = nx.Graph()
    nodes = tree.all_nodes()
    leaf_list = []
    
    real_node_idxs = np.reshape(np.arange(num_rows*num_cols), (num_rows, num_cols))

    for node in nodes:
        if node.is_leaf():
            lb, rb, ub, db = node.data
            leaf_list.append((node, abs(rb - lb)*abs(ub - db)))
    print("loading quadtree simplification of terrain.....")

    for val in tqdm(leaf_list):
        
        lb, rb, ub, db = val[0].data
        rb = rb if rb < num_rows else num_rows - 1
        db = db if db < num_cols else num_cols - 1
        edge_candidates = [[(lb, ub), (lb, db)], 
                           [(lb, ub), (rb, ub)],
                           [(rb ,ub), (rb , db )],
                           [(lb, db ), (rb, db)]]
        for edge in edge_candidates:
            v1 = real_node_idxs[edge[0][0], edge[0][1]]
            v2 = real_node_idxs[edge[1][0], edge[1][1]]
            graph.add_edge(v1, v2)        
    
    ## prune extraneous edges:
    for edge in tqdm(graph.edges):
        idx1 = min(edge)
        idx2 = max(edge)

        v1_x = idx1 // num_rows
        v1_y = idx1 % num_cols 

        v2_x = idx2 // num_rows
        v2_y = idx2 % num_cols

        if v1_x == v2_x: 
            nodes_between = real_node_idxs[v1_x, v1_y+1:v2_y]
            remove_edge = not set(nodes_between).isdisjoint(graph.nodes)
            if remove_edge:
                graph.remove_edge(edge[0], edge[1])
        elif v1_y == v2_y:
            nodes_between = real_node_idxs[v1_x+1:v2_x, v1_y]
            remove_edge = not set(nodes_between).isdisjoint(graph.nodes)
            if remove_edge:
                graph.remove_edge(edge[0], edge[1])
        
    return graph

