#!/usr/bin/env python
"""
TODO:
# Author: 
# Created Time : 

# File Name: 
# Description: 

"""


import os
import tensorflow as tf
import numpy as np
from .io import write_json, output_sensitive_gene, output_gene_sensitivity
from .utils import preprocess_graph
from .metrics import linkpred_metrics
from .plot import plot_histogram, plot_top10_gene_sensitivity


def get_weight(model_path):
    files = os.listdir(model_path)
    files_meta = list(filter(lambda x: x[-4:]=='meta', files))
    sess = tf.Session()
    saver = tf.train.import_meta_graph(model_path+'/'+files_meta[0])
    saver.restore(sess,tf.train.latest_checkpoint(model_path))
    e_dense_1_weights = sess.run("DeepLinc/Encoder/e_dense_1_vars/weights:0")
    e_dense_2_weights = sess.run("DeepLinc/Encoder/e_dense_2_vars/weights:0")
    e_dense_3_weights = sess.run("DeepLinc/Encoder/e_dense_3_vars/weights:0")
    return e_dense_1_weights, e_dense_2_weights, e_dense_3_weights


def get_sensitivity(exp_df, feas, model_path):
    gene_name = exp_df.columns.tolist()
    adj_norm = preprocess_graph(feas['adj_orig'])
    e_dense_1_weights, e_dense_2_weights, e_dense_3_weights = get_weight(model_path)

    sess = tf.Session()

    def single_gene_occlusion(features):
        # Get embeddings with occluded gene expression
        hidden1 = np.dot(np.dot(adj_norm.toarray(), features), e_dense_1_weights)
        hidden1 = (np.abs(hidden1) + hidden1) / 2.0
        mean = np.dot(np.dot(adj_norm.toarray(), hidden1), e_dense_2_weights)
        std = np.dot(np.dot(adj_norm.toarray(), hidden1), e_dense_3_weights)
        H = mean + tf.random_normal([adj.shape[0], 125]).eval(session=sess) * tf.exp(std).eval(session=sess)
        # Calculate test score with occluded gene expression
        roc_score, ap_score, acc_score, _ = linkpred_metrics(feas['test_edges'], feas['test_edges_false']).get_roc_score(H, feas)
        return roc_score, ap_score, acc_score

    # Calculate test score with original gene expression 
    roc_score_orig, ap_score_orig, acc_score_orig = single_gene_occlusion(exp_df.values)

    # Calculate the test score for each gene in a loop
    single_gene_roc_score = dict()
    single_gene_ap_score = dict()
    for i in range(0,exp_df.values.shape[1]):
        col_all_roc_score = []
        col_all_ap_score = []
        col_all_acc_score = []
        for j in range(30):
            exp_occlu = copy.deepcopy(exp_df.values)
            np.random.shuffle(exp_occlu[:,i])
            roc_score, ap_score, _ = single_gene_occlusion(exp_occlu)
            col_all_roc_score.append(roc_score)
            col_all_ap_score.append(ap_score)
            del exp_occlu
        single_gene_roc_score.update({gene_name[i]: col_all_roc_score})
        single_gene_ap_score.update({gene_name[i]: col_all_ap_score})

    # Get gene sensitivity
    occlu_deta_ap = {}
    occlu_deta_roc = {}
    for k,v in occlu_ap.items():
        occlu_deta_ap[k] = np.abs(float(ap_score_orig) - np.array(v).mean())
    for k,v in occlu_roc.items():
        occlu_deta_roc[k] = np.abs(float(roc_score_orig) - np.array(v).mean())

    # Save results
    write_json(single_gene_roc_score, 'single_gene_occlusion_roc_score')
    write_json(single_gene_ap_score, 'single_gene_occlusion_ap_score')
    f = open("single_gene_occlusion_score_orig.txt","a")
    f.write(str(roc_score_orig) + " ")
    f.write(str(ap_score_orig) + " ")
    f.close()
    output_gene_sensitivity(occlu_deta_ap)
    plot_histogram(list(occlu_deta_ap.values()), 'sensitivity distribution', xlabel='sensitivity', ylabel='density', ifylog=True)
    plot_top10_gene_sensitivity(occlu_deta_ap, xlabel='gene name', ylabel='sensitivity', filename='Top10_gene_sensitivity')


