#!/usr/bin/env python
"""
TODO:
# Author: 
# Created Time : 

# File Name: 
# Description: 
    Input: 
        
    Output:
        1. 
        2. 
        3. 
"""


import numpy as np
import pandas as pd
import pickle as pkl
import json
from sklearn.preprocessing import MinMaxScaler, StandardScaler
# from collections import Counter


def read_pickle(inputfile):
    return pkl.load(open(inputfile, "rb"))


def write_pickle(outputfile, filename):
    with open(filename+'.pkl','wb') as f:
        pkl.dump(outputfile,f)


def filter_gene(df, filter_num):
    rows_nonzero = []
    cols_nonzero = []
    for i in range(df.values.shape[0]):
        rows_nonzero.append(np.where(df.values[i,:]!=0)[0].shape[0])

    for i in range(df.values.shape[1]):
        cols_nonzero.append(np.where(df.values[:,i]!=0)[0].shape[0])

    # result = Counter(cols_nonzero)

    not_filtered_gene = []
    for i in df.columns.tolist():
        if np.where(df[i].values!=0)[0].shape[0] >= filter_num:
            not_filtered_gene.append(i)

    return df[not_filtered_gene]


def log_transform(df, add_number):
    df_add = df + add_number
    df_log = np.log(df_add)
    df_log_zscore = StandardScaler().fit_transform(df_log).T
    df_log_zscore = StandardScaler().fit_transform(df_log_zscore).T
    return pd.DataFrame(df_log_zscore, columns=df_add.columns)


def read_dataset(input_exp, input_adj, filter_num=None, add_number=None):
    exp = pd.read_csv(open(input_exp))
    adj = pd.read_csv(open(input_adj))

    if not (adj.values==adj.values.T).all():
        raise ImportError('The input adjacency matrix is not a symmetric matrix, please check the input.')
    if not np.diag(adj.values).sum()==0:
        raise ImportError('The diagonal elements of input adjacency matrix are not all 0, please check the input.')

    if filter_num is not None:
        exp = filter_gene(exp, filter_num)
    if add_number is not None:
        exp = log_transform(exp, add_number)

    return exp, adj


def read_coordinate(input_coord):
    coord = pd.read_csv(open(input_coord))  
    return coord


def read_cell_label(input_label):
    label = pd.read_csv(open(input_label))  
    return label


# def read_processed_dataset():







def write_csv_matrix(matrix, filename, ifindex=False, ifheader=True, rownames=None, colnames=None, transpose=False):
    if transpose:
        matrix = matrix.T
        rownames, colnames = colnames, rownames
        ifindex, ifheader = ifheader, ifindex

    pd.DataFrame(matrix, index=rownames, columns=colnames).to_csv(filename+'.csv', index=ifindex, header=ifheader)



def write_json(object, filename):
    with open(filename+'.json', 'w') as f:
        json.dump(object, f)


def output_sensitive_gene(occlu_deta_score, threshold):
    gene_list = {k:v for k,v in occlu_deta_score.items() if v>threshold}
    f = open('Highly_sensitive_genes_'+str(threshold)+'.txt',"a")
    for list_mem in list(gene_list.keys()):
        f.write(list_mem + "\n")
    f.close()


def output_gene_sensitivity(occlu_deta_score):
    gene_list = sorted(occlu_deta_score.items(), key=lambda item:item[1], reverse=True)
    f = open('Gene_sensitivity.txt',"a")
    for list_mem in gene_list:
        f.write(list_mem[0] + "\t")
        f.write(str(list_mem[1]) + "\n")
    f.close()








