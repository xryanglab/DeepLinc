#!/usr/bin/env python
"""
TODO:
# Author: 
# Created Time : 

# File Name: 
# Description: 

"""


import numpy as np
from sklearn.cluster import KMeans
from sklearn import metrics
from .plot import plot_cluster_score


def optimal_cluster_num(latent_feature, number_range, plot=False):
    K_2_max_cluster_number = []
    K_2_max_score = []

    latent_feature_umap2 = umap.UMAP(n_neighbors=5, min_dist=0.3, n_components=2).fit_transform(latent_feature)
    all_score = {}
    for k in list(range(number_range[0], number_range[1])):
        K_2 = KMeans(n_clusters=int(k)).fit(latent_feature_umap2)
        all_score[k] = metrics.calinski_harabaz_score(latent_feature_umap2, K_2.labels_)

    if plot:
        plot_cluster_score(list(all_score.keys()), list(all_score.keys()), 'cluster_number', 'calinski_harabaz_score', 'The_change_of_calinski_harabaz_score')

    K_2_max_cluster_number = max(all_score.items(), key=lambda x: x[1])[0]
    K_2_max_score = max(all_score.items(), key=lambda x: x[1])[1]
    return K_2_max_cluster_number, K_2_max_score


def clustering(latent_feature, cluster_number=None, number_range=[2,7], plot_cluster_score=False):
    if cluster_number == None:
        cluster_number, _ = optimal_cluster_num(latent_feature, number_range, plot_cluster_score)

    K_2 = KMeans(n_clusters=cluster_number).fit(latent_feature)
    cluster_label = np.hstack((np.array(list(range(latent_feature.shape[0]))).reshape(-1,1),K_2.labels_.reshape(-1,1)))

    return cluster_label





