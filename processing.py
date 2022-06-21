#!/usr/bin/env python
"""
# Author: Runze Li
# Created Time: 

# File Name: preprocessing.py
# Description: 

"""


import os
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='The module is used for transforming the general coordinate information from single-cell spatial transcriptome data into the adjacency matrix.')
    parser.add_argument('--coordinate', '-c', type=str, help='Input cell coordinate data path')
    parser.add_argument('--link_num', '-n', type=int, default=3, help='Select n cells closest to each cell to define the initial interaction network')
    parser.add_argument('--dist_cutoff', '-d', default=0.98, help='Distance cutoff based on the distance distribution of each cell with the selected n closest cells (>1: distance value, <1: distance quantile; default: 0.95)')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    # Import modules
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    from scipy.spatial.distance import pdist, squareform
    import copy
    from deeplinc.io import read_coordinate
    from deeplinc.plot import plot_histogram

    # Import data
    coord_df = read_coordinate(args.coordinate)
    coord = coord_df.values

    # Convert cell coordinates to the adjacency matrix based on selected distance cutoff and number of neighbors
    dist_matrix_rongyu = pdist(coord, 'euclidean')
    dist_matrix = squareform(dist_matrix_rongyu)

    dist_neighbor_n = np.array([])
    for i in range(0, dist_matrix.shape[0]):
        tmp = dist_matrix[i,:]
        neighbor_n_index = list(np.where(tmp <= np.sort(tmp)[args.link_num])[0])
        neighbor_n_index.remove(i)
        dist_neighbor_n = np.hstack((dist_neighbor_n,dist_matrix[i,neighbor_n_index].tolist()))

    if args.dist_cutoff < 1:
        cutoff_distance = np.percentile(dist_neighbor_n, args.dist_cutoff*100)
    else:
        cutoff_distance = args.dist_cutoff

    adjacency_matrix1 = copy.deepcopy(dist_matrix)
    adjacency_matrix1 = np.int64(adjacency_matrix1 < cutoff_distance)
    for j in range(0, adjacency_matrix1.shape[0]):
        adjacency_matrix1[j,j] = 0
    adjacency_matrix2 = copy.deepcopy(dist_matrix)
    for i in range(0, adjacency_matrix2.shape[0]):
        tmp = adjacency_matrix2[i,:]
        adjacency_matrix2[i,:] = np.int64(adjacency_matrix2[i,:] <= np.sort(tmp)[args.link_num])
        adjacency_matrix2[i,i] = 0
    adjacency_matrix3 = adjacency_matrix1 + adjacency_matrix2
    adjacency_matrix3 = np.int64(adjacency_matrix3 == 2)
    adj = copy.deepcopy(adjacency_matrix3)
    adj = adj + adj.T
    adj = (adj != 0).astype('int')
    adj = pd.DataFrame(adj)
    adj.to_csv('adj_dist%.2f'%cutoff_distance+'_neigh'+str(args.link_num)+'.csv', header=True, index=False)

    # Output the distance between each cell and its n nearest neighbors to monitor whether the threshold settings are reasonable
    plot_histogram(dist_matrix_rongyu, 'distance', 'count', 'Distance_distribution_for_each_cell_pair')
    plot_histogram(dist_neighbor_n, 'distance', 'count', 'Distance_distribution_between_each_cell_and_its_selected_n_nearest_neighbors')
    print("The distance cutoff is %.2f"%cutoff_distance+', and '+str(args.link_num)+' cells closest to each cell are selected to define the initial interaction network')

