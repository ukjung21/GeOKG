"""
node2vec
Aditya Grover and Jure Leskovec
"""

import numpy as np
import scipy.sparse as sp
import networkx as nx
from collections import Counter#
import time#

# Convert sparse matrix to tuple
def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape

# Get normalized adjacency matrix: A_norm
def preprocess_graph(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return sparse_to_tuple(adj_normalized)

# Prepare feed-dict for Tensorflow session
def construct_feed_dict(adj_normalized, adj, features, placeholders):
    # construct feed dictionary
    feed_dict = dict()
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['adj']: adj_normalized})
    feed_dict.update({placeholders['adj_orig']: adj})
    return feed_dict

# Perform train-test split
    # Takes in adjacency matrix in sparse format
    # Returns: adj_train, train_edges, val_edges, val_edges_false, 
        # test_edges, test_edges_false
def mask_test_edges(adj, row, col, rel_list, go_level, test_frac=.1, val_frac=.05, prevent_disconnect=True, verbose=False):
    # NOTE: Splits are randomized and results might slightly deviate from reported numbers in the paper.
    if verbose == True:
        print ('preprocessing...')

    # Remove diagonal elements
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    # Check that diag is zero:
    assert np.diag(adj.todense()).sum() == 0

    g = nx.from_scipy_sparse_array(adj)
    # g_und = nx.from_scipy_sparse_array(adj)
    print('number of edges: ', g.number_of_edges())
    orig_num_cc = nx.number_connected_components(g)

    # adj_triu = sp.triu(adj) # upper triangular portion of adj matrix
    # adj_tuple = sparse_to_tuple(adj_triu) # (coords, values, shape), edges only 1 way
    # edges = adj_tuple[0] # all edges, listed only once (not 2 ways)
    # edges = sparse_to_tuple(adj)[0] # ALL edges (includes both ways)
    edges = [[e1, e2] for e1, e2 in zip(row, col)]
    num_test = int(np.floor(len(edges) * test_frac)) # controls how large the test set should be
    num_val = int(np.floor(len(edges) * val_frac)) # controls how alrge the validation set should be
    print('num test: {}, num_val: {}'.format(num_test, num_val))
    
    # Store edges in list of ordered tuples (node1, node2) where node1 < node2
    edge_tuples = [(edge[0], edge[1]) for edge in edges]
    all_edge_tuples = set(edge_tuples)
    # train_edges = set(edge_tuples) # initialize train_edges to have all edges
    # test_edges = set()
    # val_edges = set()
    # train_edges = edge_tuples # initialize train_edges to have all edges
    edge_dict = {i : (edge[0], edge[1]) for i, edge in enumerate(edges)}
    train_edges = dict(edge_dict)
    test_edges = []
    val_edges = []
    
    rel_dict = {i : rel for i,rel in enumerate(rel_list)}
    test_rel = []
    val_rel = []
    
    rel_cnt = [(rel, cnt) for rel, cnt in zip(Counter(rel_list).keys(), Counter(rel_list).values())]
    rel_cnt = dict(rel_cnt)
    test_rel_cnt = dict([(0,0),(1,0),(2,0),(3,0),(4,0),(5,0),(6,0),(7,0),(8,0)])
    val_rel_cnt = dict([(0,0),(1,0),(2,0),(3,0),(4,0),(5,0),(6,0),(7,0),(8,0)])
    
    if verbose == True:
        print ('generating test/val sets...')

    # Iterate over shuffled edges, add to train/val sets
    # np.random.shuffle(edge_tuples)
    i=0#
    cycle_cnt=0
    n=len(rel_dict)
    print('# edges: {}'.format(n))
    start = time.time()
    
    for j, edge in edge_dict.items():
        # print edge
        i+=1#
        if i%1000 == 0:
            print('{}th edge: {:.2f}%  #test: {}  #valid: {}   time : {} min'.format(i, i/n*100, len(test_edges), len(val_edges), round((time.time() - start)/60)))
        node1 = edge[0]
        node2 = edge[1]
        r = rel_list[j]
        if go_level[node1]<4 or go_level[node1]>6:
            continue

        # If removing edge would disconnect a connected component, backtrack and move on
        # try:
        #     g.remove_edge(node1, node2)
        # except:
        #     cycle_cnt+=1
        #     g.add_edge(node1, node2)
        #     g.remove_edge(node1, node2)
        g.remove_edge(node1, node2)
        if prevent_disconnect == True:
            if nx.number_connected_components(g) > orig_num_cc:
                g.add_edge(node1, node2)
                continue

        # Fill test_edges first
        if len(test_edges) < num_test:
            if test_rel_cnt[r] > rel_cnt[r]*0.1:
                g.add_edge(node1, node2)
                continue
            test_rel_cnt[r]+=1
            test_edges.append(edge)
            del train_edges[j]
            test_rel.append(r)
            del rel_dict[j]

        # Then, fill val_edges
        elif len(val_edges) < num_val:
            if val_rel_cnt[r] > rel_cnt[r]*0.1:
                g.add_edge(node1, node2)
                continue
            val_rel_cnt[r]+=1
            val_edges.append(edge)
            del train_edges[j]
            val_rel.append(r)
            del rel_dict[j]

        # Both edge lists full --> break loop
        elif len(test_edges) == num_test and len(val_edges) == num_val:
            break
    
    train_rel = list(rel_dict.values())
    train_edges = list(train_edges.values())
    
    if (len(val_edges) < num_val or len(test_edges) < num_test):
        print ("WARNING: not enough removable edges to perform full train-test split!")
        print ("Num. (test, val) edges requested: (", num_test, ", ", num_val, ")")
        print ("Num. (test, val) edges returned: (", len(test_edges), ", ", len(val_edges), ")")

    if prevent_disconnect == True:
        assert nx.number_connected_components(g) == orig_num_cc
    
    print('cycle count : ', cycle_cnt)
    
    if verbose == True:
        print ('creating false test edges...')

    test_edges_false = []
    while len(test_edges_false) < num_test:
        idx_i = test_edges[len(test_edges_false)]
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue

        false_edge = (idx_i, idx_j)

        # Make sure false_edge not an actual edge, and not a repeat
        if false_edge in all_edge_tuples:
            continue
        if (idx_j, idx_i) in all_edge_tuples:
            continue
        if false_edge in test_edges_false:
            continue

        test_edges_false.append(false_edge)
    
    test_inv_false = []
    while len(test_inv_false) < num_test:
        idx_i = test_edges_false[len(test_inv_false)][1]
        idx_j = test_edges_false[len(test_inv_false)][0]
        if idx_i == idx_j:
            continue

        false_edge = (idx_i, idx_j)

        # Make sure false_edge not an actual edge, and not a repeat
        if false_edge in all_edge_tuples:
            continue
        if false_edge in test_inv_false:
            continue

        test_inv_false.append(false_edge)
        
#     if verbose == True:
#         print ('creating false val edges...')

#     val_edges_false = set()
#     while len(val_edges_false) < num_val:
#         idx_i = np.random.randint(0, adj.shape[0])
#         idx_j = np.random.randint(0, adj.shape[0])
#         if idx_i == idx_j:
#             continue

#         false_edge = (min(idx_i, idx_j), max(idx_i, idx_j))

#         # Make sure false_edge in not an actual edge, not in test_edges_false, not a repeat
#         if false_edge in all_edge_tuples or \
#             false_edge in test_edges_false or \
#             false_edge in val_edges_false:
#             continue
            
#         val_edges_false.add(false_edge)
    
#     if verbose == True:
#         print ('creating false train edges...')

#     train_edges_false = set()
#     while len(train_edges_false) < len(train_edges):
#         idx_i = np.random.randint(0, adj.shape[0])
#         idx_j = np.random.randint(0, adj.shape[0])
#         if idx_i == idx_j:
#             continue

#         false_edge = (min(idx_i, idx_j), max(idx_i, idx_j))

#         # Make sure false_edge in not an actual edge, not in test_edges_false, 
#             # not in val_edges_false, not a repeat
#         if false_edge in all_edge_tuples or \
#             false_edge in test_edges_false or \
#             false_edge in val_edges_false or \
#             false_edge in train_edges_false:
#             continue

#         train_edges_false.add(false_edge)
    
#     if verbose == True:
#         print ('final checks for disjointness...')

#     # assert: test, val, train positive edges disjoint
#     assert val_edges.isdisjoint(train_edges)
#     assert test_edges.isdisjoint(train_edges)
#     assert val_edges.isdisjoint(test_edges)
    
    if verbose == True:
        print ('creating adj_train...')

    # Re-build adj matrix using remaining graph
    adj_train = nx.adjacency_matrix(g)

    # Convert edge-lists to numpy arrays
    # train_edges = np.array(train_edges)
    # train_edges_false = np.array([list(edge_tuple) for edge_tuple in train_edges_false])
    # val_edges = np.array(val_edges)
    # # val_edges_false = np.array([list(edge_tuple) for edge_tuple in val_edges_false])
    # test_edges = np.array(test_edges)
    # test_edges_false = np.array(test_edges_false)
    # test_inv_false = np.array(test_inv_false)
    
    if verbose == True:
        print ('Done with train-test split!')
        print ('')

    # NOTE: these edge lists only contain single direction of edge!
    return adj_train, train_edges, val_edges, test_edges, test_edges_false, test_inv_false, train_rel, val_rel, test_rel

