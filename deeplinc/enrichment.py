#!/usr/bin/env python
"""
TODO:
# Author: 
# Created Time : 

# File Name: 
# Description: 

"""


import numpy as np
import scipy.sparse as sp
import random
import math


def connection_number_between_groups(adj, cell_type_label):
    single_cell_VS_cell_type = np.zeros((cell_type_label.shape[0], len(np.unique(cell_type_label))), dtype=int)
    for i1 in range(0, adj.shape[0]):
        single_cell_adjacency = adj[i1, :]
        single_cell_adjacency_index = np.where(single_cell_adjacency == 1)
        single_cell_adjacency_cell_type = cell_type_label[single_cell_adjacency_index[0]]
        single_cell_adjacency_cell_type_unique = np.sort(np.unique(single_cell_adjacency_cell_type))
        for i2 in single_cell_adjacency_cell_type_unique:
            single_cell_VS_cell_type[i1,i2] = list(single_cell_adjacency_cell_type).count(i2)

    cell_type_VS_cell_type = np.zeros((len(np.unique(cell_type_label)), len(np.unique(cell_type_label))), dtype=float)
    single_cell_VS_cell_type_usedforplot = {}
    for i3 in range(0,len(np.unique(cell_type_label))):
        cell_type_VS_cell_type[i3,:] = single_cell_VS_cell_type[np.where(cell_type_label == i3)[0],:].sum(axis=0)
        single_cell_VS_cell_type_usedforplot[i3] = single_cell_VS_cell_type[np.where(cell_type_label == i3)[0],:] #8个key-value，key是细胞类型ID，value是某种类型细胞的single cell_VS_cell type的邻接数矩阵

    return cell_type_VS_cell_type, single_cell_VS_cell_type_usedforplot


def randAdj(cell_number, edge_number):  #edge_number指最终生成的随机邻接矩阵中所有1值的数目
    tri_number = (cell_number*cell_number-cell_number)/2
    nums = np.zeros(int(tri_number))
    nums[:int(edge_number/2)] = 1
    np.random.shuffle(nums)
    adj = np.zeros((cell_number, cell_number))
    adj[np.triu_indices(cell_number, 1)] = nums  #偏移量设为1，是上三角矩阵；若要获取下三角矩阵，将偏移量设置为-1
    return adj + adj.T


def randAdj_long_edges(dist_matrix, cutoff_distance, edge_number):  #edge_number指最终生成的随机邻接矩阵中所有1值的数目
    adj_all_long_edges = np.int64(dist_matrix >= cutoff_distance)
    adj_all_long_edges = sp.csr_matrix(adj_all_long_edges)
    edges = sparse_to_tuple(sp.triu(adj_all_long_edges))[0]
    selected_long_edges = edges[random.sample(range(0, edges.shape[0]), edge_number),:]
    data = np.ones(selected_long_edges.shape[0])
    adj_selected = sp.csr_matrix((data, (selected_long_edges[:, 0], selected_long_edges[:, 1])), shape=adj_all_long_edges.shape)
    adj_selected = adj_selected + adj_selected.T
    return adj_selected.toarray()


def generate_adj_new_long_edges(dist_matrix, new_edges, all_new_edges_dist, cutoff_distance):
    selected_new_long_edges = new_edges[all_new_edges_dist >= cutoff_distance]
    mask = np.ones(selected_new_long_edges.shape[0])
    adj_new_long_edges = sp.csr_matrix((mask, (selected_new_long_edges[:, 0], selected_new_long_edges[:, 1])), shape=dist_matrix.shape)
    adj_new_long_edges = adj_new_long_edges + adj_new_long_edges.T
    return adj_new_long_edges


def edges_enrichment_evaluation(adj, cell_type_label, cell_type_name, N=2000, edge_type='all edges', **kwargs):
    if kwargs:
        dist_matrix = kwargs['dist_matrix']
        cutoff_distance = kwargs['cutoff_distance']

    cell_type_ID_name = {}
    cell_type_ID_number = {}
    for i in range(0, len(np.unique(cell_type_label))):
        cell_type_ID_name[str(i)] = str(i)
        cell_type_ID_number[i] = np.where(cell_type_label==i)[0].shape[0]

    cell_type_VS_cell_type_ID_number = np.zeros((len(np.unique(cell_type_label)),len(np.unique(cell_type_label))))
    for i in range(0, len(np.unique(cell_type_label))):
        for j in range(0, len(np.unique(cell_type_label))):
            cell_type_VS_cell_type_ID_number[i,j] = cell_type_ID_number[i] * cell_type_ID_number[j] #几何平均

    cell_type_VS_cell_type_shuffle_alltimes_1 = {}
    cell_type_VS_cell_type_shuffle_alltimes_2 = {}
    for i in cell_type_ID_name:
        for j in cell_type_ID_name:
            cell_type_VS_cell_type_shuffle_alltimes_1[i + '-' + j] = []
            cell_type_VS_cell_type_shuffle_alltimes_2[i + '-' + j] = []

    cell_type_VS_cell_type_true, _ = connection_number_between_groups(adj, cell_type_label)
    if edge_type == 'all edges':
        cell_type_VS_cell_type_true = cell_type_VS_cell_type_true/cell_type_VS_cell_type_ID_number

    def merge(test_onetime, test_alltime):
        for i in range(0, test_onetime.shape[0]):
            for j in range(0, test_onetime.shape[0]):
                thelist = test_alltime['%s-%s'%(i,j)]
                thelist.append(test_onetime[i,j])

    if type == 'all edges':
        cell_type_shuffle = cell_type_label
    if type == 'all edges':
        for num in range(0, N):
            random.shuffle(cell_type_shuffle)
            cell_type_VS_cell_type_shuffle_onetime, _ = connection_number_between_groups(adj, cell_type_shuffle)
            cell_type_VS_cell_type_shuffle_onetime = cell_type_VS_cell_type_shuffle_onetime/cell_type_VS_cell_type_ID_number
            if num+1 <= N/2:
                merge(cell_type_VS_cell_type_shuffle_onetime, cell_type_VS_cell_type_shuffle_alltimes_1)
            else:
                merge(cell_type_VS_cell_type_shuffle_onetime, cell_type_VS_cell_type_shuffle_alltimes_2)
            if num+1%100 == 0:
                print('%s times of permutations have completed calculating ...'%num+1)
    elif type == 'long edges':
        for num in range(0, N):
            adj_shuffle = randAdj_long_edges(dist_matrix, cutoff_distance, int(cell_type_VS_cell_type_true.sum()/2))
            cell_type_VS_cell_type_shuffle_onetime, _ = connection_number_between_groups(adj_shuffle, cell_type_label)
            if num+1 <= N/2:
                merge(cell_type_VS_cell_type_shuffle_onetime, cell_type_VS_cell_type_shuffle_alltimes_1)
            else:
                merge(cell_type_VS_cell_type_shuffle_onetime, cell_type_VS_cell_type_shuffle_alltimes_2)
            if num+1%100 == 0:
                print('%s times of permutations have completed calculating ...'%num+1)

    #计算右侧P value，即假定两类细胞之间是enrichment/interaction的，计算P value
    cell_type_VS_cell_type_enrichment_P = np.zeros((len(np.unique(cell_type_label)), len(np.unique(cell_type_label))), dtype=float)
    for i in range(0,len(np.unique(cell_type_label))):
        for j in range(0,len(np.unique(cell_type_label))):
            P_tmp = len(np.where(np.array(cell_type_VS_cell_type_shuffle_alltimes_1['%s-%s'%(i,j)]) >= cell_type_VS_cell_type_true[i,j])[0]) / (N/2)
            cell_type_VS_cell_type_enrichment_P[i,j] =  P_tmp

    #计算左侧P value，即假定两类细胞之间是depletion/avoidance的，计算P value（用负值与enrichment区分）
    cell_type_VS_cell_type_depletion_P = np.zeros((len(np.unique(cell_type_label)), len(np.unique(cell_type_label))), dtype=float)
    for i in range(0,len(np.unique(cell_type_label))):
        for j in range(0,len(np.unique(cell_type_label))):
            P_tmp = len(np.where(np.array(cell_type_VS_cell_type_shuffle_alltimes_2['%s-%s'%(i,j)]) <= cell_type_VS_cell_type_true[i,j])[0]) / (N/2)
            cell_type_VS_cell_type_depletion_P[i,j] = P_tmp

    #合并在一起，用于画热度图
    cell_type_VS_cell_type_merge_P = np.zeros((len(np.unique(cell_type_label)), len(np.unique(cell_type_label))), dtype=float)
    P_enrichment = []
    P_depletion = []
    for i in range(0,len(np.unique(cell_type_label))):
        for j in range(0,len(np.unique(cell_type_label))):
            if cell_type_VS_cell_type_enrichment_P[i,j] == 0.5:
                if cell_type_VS_cell_type_depletion_P[i,j] == 0:
                    cell_type_VS_cell_type_merge_P[i,j] = -3
                elif cell_type_VS_cell_type_depletion_P[i,j] == 1:
                    cell_type_VS_cell_type_merge_P[i,j] = +3
                elif cell_type_VS_cell_type_depletion_P[i,j] <= 0.5 and cell_type_VS_cell_type_depletion_P[i,j] > 0:
                    cell_type_VS_cell_type_merge_P[i,j] = - (-math.log10(cell_type_VS_cell_type_depletion_P[i,j])) #负值表示depletion/avoidance
                    P_depletion.append(- (-math.log10(cell_type_VS_cell_type_depletion_P[i,j])))
                elif cell_type_VS_cell_type_depletion_P[i,j] > 0.5 and cell_type_VS_cell_type_depletion_P[i,j] < 1:
                    cell_type_VS_cell_type_merge_P[i,j] = -math.log10(1-cell_type_VS_cell_type_depletion_P[i,j]) #正值表示enrichment/interaction
                    P_enrichment.append(-math.log10(1-cell_type_VS_cell_type_depletion_P[i,j]))
            elif cell_type_VS_cell_type_enrichment_P[i,j] == 0:
                cell_type_VS_cell_type_merge_P[i,j] = +3
            elif cell_type_VS_cell_type_enrichment_P[i,j] == 1:
                cell_type_VS_cell_type_merge_P[i,j] = -3
            elif cell_type_VS_cell_type_enrichment_P[i,j] < 0.5 and cell_type_VS_cell_type_enrichment_P[i,j] > 0:
                if cell_type_VS_cell_type_depletion_P[i,j] == 1:
                    cell_type_VS_cell_type_merge_P[i,j] = +3
                elif cell_type_VS_cell_type_depletion_P[i,j] > 0.5 and cell_type_VS_cell_type_depletion_P[i,j] < 1:
                    cell_type_VS_cell_type_merge_P[i,j] = -math.log10(cell_type_VS_cell_type_enrichment_P[i,j])
                    P_enrichment.append(-math.log10(cell_type_VS_cell_type_enrichment_P[i,j]))
                elif cell_type_VS_cell_type_enrichment_P[i,j] <= cell_type_VS_cell_type_depletion_P[i,j] and cell_type_VS_cell_type_depletion_P[i,j] < 0.5:
                    cell_type_VS_cell_type_merge_P[i,j] = -math.log10(cell_type_VS_cell_type_enrichment_P[i,j])
                    P_enrichment.append(-math.log10(cell_type_VS_cell_type_enrichment_P[i,j]))
                elif cell_type_VS_cell_type_depletion_P[i,j] < cell_type_VS_cell_type_enrichment_P[i,j] and cell_type_VS_cell_type_depletion_P[i,j] > 0:
                    cell_type_VS_cell_type_merge_P[i,j] = - (-math.log10(cell_type_VS_cell_type_depletion_P[i,j]))
                    P_depletion.append(- (-math.log10(cell_type_VS_cell_type_depletion_P[i,j])))
                elif cell_type_VS_cell_type_depletion_P[i,j] == 0:
                    cell_type_VS_cell_type_merge_P[i,j] = -3
            elif cell_type_VS_cell_type_enrichment_P[i,j] > 0.5 and cell_type_VS_cell_type_enrichment_P[i,j] < 1:
                if cell_type_VS_cell_type_depletion_P[i,j] == 0:
                    cell_type_VS_cell_type_merge_P[i,j] = -3
                elif cell_type_VS_cell_type_depletion_P[i,j] < 0.5 and cell_type_VS_cell_type_depletion_P[i,j] > 0:
                    cell_type_VS_cell_type_merge_P[i,j] = - (-math.log10(cell_type_VS_cell_type_depletion_P[i,j]))
                    P_depletion.append(- (-math.log10(cell_type_VS_cell_type_depletion_P[i,j])))
                elif cell_type_VS_cell_type_depletion_P[i,j] <= cell_type_VS_cell_type_enrichment_P[i,j] and cell_type_VS_cell_type_depletion_P[i,j] > 0.5:
                    cell_type_VS_cell_type_merge_P[i,j] = - (-math.log10(1-cell_type_VS_cell_type_enrichment_P[i,j]))
                    P_depletion.append(- (-math.log10(1-cell_type_VS_cell_type_enrichment_P[i,j])))
                elif cell_type_VS_cell_type_depletion_P[i,j] > cell_type_VS_cell_type_enrichment_P[i,j] and cell_type_VS_cell_type_depletion_P[i,j] < 1:
                    cell_type_VS_cell_type_merge_P[i,j] = -math.log10(1-cell_type_VS_cell_type_depletion_P[i,j])
                    P_enrichment.append(-math.log10(1-cell_type_VS_cell_type_depletion_P[i,j]))
                elif cell_type_VS_cell_type_depletion_P[i,j] == 1:
                    cell_type_VS_cell_type_merge_P[i,j] = +3

    tmp1 = [x1 for y1 in cell_type_VS_cell_type_true for x1 in y1]
    tmp2 = [x2 for y2 in cell_type_VS_cell_type_merge_P for x2 in y2]
    tmp3 = [x3 for x3 in cell_type_name for y3 in range(len(cell_type_name))]
    tmp4 = [x4 for y4 in range(len(cell_type_name)) for x4 in cell_type_name]
    test_result = np.array([tmp3,tmp4,tmp1,tmp2]).T

    return test_result, cell_type_VS_cell_type_merge_P, cell_type_VS_cell_type_enrichment_P, cell_type_VS_cell_type_depletion_P

