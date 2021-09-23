#!/usr/bin/env python
"""
TODO:
# Author: 
# Created Time : 

# File Name: 
# Description: 

"""


from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics  import accuracy_score
from sklearn import metrics
# from munkres import Munkres, print_matrix
import numpy as np
import copy
from scipy.special import expit


class linkpred_metrics():
    def __init__(self, edges_pos, edges_neg):
        self.edges_pos = edges_pos
        self.edges_neg = edges_neg

    def get_roc_score(self, emb, feas):
        # if emb is None:
        #     feed_dict.update({placeholders['dropout']: 0})
        #     emb = sess.run(model.z_mean, feed_dict=feed_dict)

        def sigmoid(x):
            if x >= 0:      #对sigmoid函数的优化，避免了出现极大的数据溢出
                return 1.0/(1+np.exp(-x))
            else:
                return np.exp(x)/(1+np.exp(x))

        # Predict on test set of edges
        adj_rec = np.dot(emb, emb.T)
        preds = []
        pos = []
        for e in self.edges_pos:
            preds.append(sigmoid(adj_rec[e[0], e[1]]))

        preds_neg = []
        neg = []
        for e in self.edges_neg:
            preds_neg.append(sigmoid(adj_rec[e[0], e[1]]))

        preds_all = np.hstack([preds, preds_neg])
        labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds))])

        roc_score = roc_auc_score(labels_all, preds_all)
        ap_score = average_precision_score(labels_all, preds_all)
        acc_score = accuracy_score(labels_all, np.round(preds_all))

        return roc_score, ap_score, acc_score, emb

    def get_prob(self, emb, feas):
        # if emb is None:
        #     feed_dict.update({placeholders['dropout']: 0})
        #     emb = sess.run(model.z_mean, feed_dict=feed_dict)

        def sigmoid(x):
            if x >= 0:      #对sigmoid函数的优化，避免了出现极大的数据溢出
                return 1.0/(1+np.exp(-x))
            else:
                return np.exp(x)/(1+np.exp(x))

        # Predict on test set of edges
        adj_rec = np.dot(emb, emb.T)
        preds = []
        pos = []
        for e in self.edges_pos:
            preds.append(sigmoid(adj_rec[e[0], e[1]]))

        preds_neg = []
        neg = []
        for e in self.edges_neg:
            preds_neg.append(sigmoid(adj_rec[e[0], e[1]]))

        labels_all = np.hstack((np.array(['connections between not_disrupted cells' for i in range(len(preds))]), np.array(['connections between disrupted cells' for i in range(len(preds))])))
        preds_all = np.hstack([preds, preds_neg])

        return np.hstack((labels_all.reshape(-1,1),preds_all.reshape(-1,1)))


class select_optimal_threshold():
    def __init__(self, edges_pos, edges_neg):
        self.edges_pos = edges_pos
        self.edges_neg = edges_neg

    def select(self, emb, feas):
        # if emb is None:
        #     feed_dict.update({placeholders['dropout']: 0})
        #     emb = sess.run(model.z_mean, feed_dict=feed_dict)

        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        # Predict on test set of edges
        adj_rec = sigmoid(np.dot(emb, emb.T))
        preds = []
        pos = []
        for e in self.edges_pos:
            preds.append(adj_rec[e[0], e[1]])
            # pos.append(feas['adj_orig'][e[0], e[1]])

        preds_neg = []
        neg = []
        for e in self.edges_neg:
            preds_neg.append(adj_rec[e[0], e[1]])
            # neg.append(feas['adj_orig'][e[0], e[1]])

        labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds))])

        all_acc_score = {}
        max_acc_score = 0
        optimal_threshold = 0
        for threshold in np.arange(0.01,1,0.005):
            preds_all = np.hstack([preds, preds_neg])
            preds_all = (preds_all>threshold).astype('int')
            acc_score = accuracy_score(labels_all, preds_all)
            all_acc_score[threshold] = acc_score
            if acc_score > max_acc_score:
                max_acc_score = acc_score
                optimal_threshold = threshold

        for i in range(0, adj_rec.shape[0]):
            adj_rec[i,i] = 0

        adj_rec_1 = copy.deepcopy(adj_rec)
        adj_rec_1 = (adj_rec_1>optimal_threshold).astype('int')
        for j in range(0, adj_rec_1.shape[0]):
            adj_rec_1[j,j] = 0

        def add_limit(adj_rec, adj_rec_1, top_num, type):
            adj_rec_new_tmp = copy.deepcopy(adj_rec)
            for z in range(0, adj_rec_new_tmp.shape[0]):
                tmp = adj_rec_new_tmp[z,:]
                adj_rec_new_tmp[z,:] = (adj_rec_new_tmp[z,:] >= np.sort(tmp)[-top_num]).astype('int')
            adj_rec_new = adj_rec_1 + adj_rec_new_tmp
            adj_rec_new = (adj_rec_new == 2).astype('int')
            adj_rec_new = adj_rec_new + adj_rec_new.T
            if type == 'union':  #并集：重构的网络中每个细胞至少是n条边连接
                adj_rec_new = (adj_rec_new != 0).astype('int')
            elif type == 'intersection':  #交集：重构的网络中每个细胞连接的细胞一定在它的top n中，这样可能存在一些细胞没有连接
                adj_rec_new = (adj_rec_new == 2).astype('int')
            return adj_rec_new

        adj_rec_2 = add_limit(adj_rec, adj_rec_1, 3, 'union')
        adj_rec_3 = add_limit(adj_rec, adj_rec_1, 5, 'intersection')

        print((adj_rec_1==adj_rec_1.T).all())
        print((adj_rec_2==adj_rec_2.T).all())
        print((adj_rec_2==adj_rec_2.T).all())
        print((adj_rec_2==adj_rec_3).all())

        return adj_rec, adj_rec_1, adj_rec_2, adj_rec_3, all_acc_score, max_acc_score, optimal_threshold


