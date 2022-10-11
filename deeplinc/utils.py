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
import inspect
try:
    import tensorflow as tf
except ImportError:
    raise ImportError('DeepLinc requires TensorFlow. Please follow instructions'
                      ' at https://www.tensorflow.org/install/ to install'
                      ' it.')


# =============== Data processing ===============
# ===============================================

def sparse2tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape


def preprocess_graph(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return adj_normalized


def ismember(tmp1, tmp2, tol=5):
    """
	Judge whether there are overlapping elements in tmp1 and tmp2
	"""
    rows_close = np.all(np.round(tmp1 - tmp2[:, None], tol) == 0, axis=-1)
    if True in np.any(rows_close, axis=-1).tolist():
        return True
    elif True not in np.any(rows_close, axis=-1).tolist():
        return False


def retrieve_name(var):
    callers_local_vars = list(inspect.currentframe().f_back.f_locals.items())
    return [var_name for var_name, var_val in callers_local_vars if var_val is var]


def set_placeholder(adj, latent_dim):
    var_placeholders = {
        'features': tf.sparse_placeholder(tf.float32),
        'adj': tf.sparse_placeholder(tf.float32),
        'adj_orig': tf.sparse_placeholder(tf.float32),
        'dropout': tf.placeholder_with_default(0., shape=()),
        'real_distribution': tf.placeholder(dtype=tf.float32, shape=[adj.shape[0], latent_dim],
                                            name='real_distribution')
    }

    return var_placeholders


def construct_feed_dict(adj_normalized, adj, features, placeholders):
    # construct feed dictionary
    feed_dict = dict()
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['adj']: adj_normalized})
    feed_dict.update({placeholders['adj_orig']: adj})  #注意这里的adj_orig经过重新定义已经不是feas['adj_orig']了，在constructor中的update函数中是adj_label，是只有训练集
    return feed_dict


# def unified_data_format(exp_values, adj_values):
#     exp_values = sp.csr_matrix(exp_values)
#     adj_values = sp.csr_matrix(adj_values)

    










# =============== Training and testing set splitting ===============
# ==================================================================

def sampling_test_edges_neg(n, test_edges, edges_double):
    test_edges_false = []
    while len(test_edges_false) < len(test_edges):
        idx_i = np.random.randint(0, n)
        idx_j = np.random.randint(0, n)
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], edges_double):
            continue
        if ismember([idx_j, idx_i], edges_double):
            continue
        if test_edges_false:
            if ismember([idx_j, idx_i], np.array(test_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(test_edges_false)):
                continue
        test_edges_false.append([idx_i, idx_j])
    return test_edges_false


def train_test_split(adj_values, test_ratio=0.1):
    # Get the id of all edges
    edges_single = sparse2tuple(sp.triu(adj_values))[0]  #single direction of edges
    edges_double = sparse2tuple(adj_values)[0]  #double direction of edges

    if test_ratio > 1:
        test_ratio = test_ratio/edges_single.shape[0]

    # Split into train and test sets
    num_test = int(np.floor(edges_single.shape[0] * test_ratio))
    all_edges_idx = list(range(edges_single.shape[0]))
    np.random.shuffle(all_edges_idx)
    test_edges_idx = all_edges_idx[:num_test]
    test_edges = edges_single[test_edges_idx]
    if (adj_values.shape[0]**2-adj_values.sum()-adj_values.shape[0])/2 < 2*len(test_edges):
        raise ImportError('The network is too dense, please reduce the proportion of test set or delete some edges in the network.')
    else:
        test_edges_false = sampling_test_edges_neg(adj_values.shape[0], test_edges, edges_double)
    train_edges = np.delete(edges_single, test_edges_idx, axis=0)

    # Mark the train and test sets in the adj matrix
    data = np.ones(train_edges.shape[0])
    adj_train = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])), shape=adj_values.shape)
    adj_train = adj_train + adj_train.T
    data = np.ones(test_edges.shape[0])
    adj_test = sp.csr_matrix((data, (test_edges[:, 0], test_edges[:, 1])), shape=adj_values.shape)
    adj_test = adj_test + adj_test.T

    return adj_train, adj_test, train_edges, test_edges, test_edges_false  #return single direction of edges


def packed_data(exp_values, adj_values, test_ratio=0.1):
    exp_values = sp.csr_matrix(exp_values)
    adj_values = sp.csr_matrix(adj_values)

    adj_train, adj_test, train_edges, test_edges, test_edges_false = train_test_split(adj_values, test_ratio)

    adj_norm = sparse2tuple(preprocess_graph(adj_train))

    num_nodes = adj_train.shape[0]

    features = sparse2tuple(exp_values.tocoo())
    num_features = features[2][1]
    features_nonzero = features[1].shape[0]

    pos_weight = float(adj_train.shape[0]**2-adj_train.sum())/adj_train.sum()
    norm = adj_train.shape[0]**2/float((adj_train.shape[0]*adj_train.shape[0]-adj_train.sum())*2)

    adj_label = adj_train + sp.eye(adj_train.shape[0])
    adj_label = sparse2tuple(adj_label)

    items = [adj_train, num_features, num_nodes, features_nonzero, pos_weight, 
              norm, adj_norm, adj_label, features, train_edges, test_edges, test_edges_false]
    # feas = {}
    # for item in items:
    #     feas[retrieve_name(item).pop()] = item
    feas = {}
    feas.update({'adj_train': adj_train})
    feas.update({'num_features': num_features})
    feas.update({'num_nodes': num_nodes})
    feas.update({'features_nonzero': features_nonzero})
    feas.update({'pos_weight': pos_weight})
    feas.update({'norm': norm})
    feas.update({'adj_norm': adj_norm})
    feas.update({'adj_label': adj_label})
    feas.update({'features': features})
    feas.update({'train_edges': train_edges})
    feas.update({'test_edges': test_edges})
    feas.update({'test_edges_false': test_edges_false})

    return feas










# =============== Building optimizer ===============
# ==================================================

class OptimizerVAE(object):
    def __init__(self, preds, labels, model, num_nodes, pos_weight, norm, d_real, d_fake, learning_rate_1, learning_rate_2):
        preds_sub = preds
        labels_sub = labels

        # Discrimminator Loss
        dc_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(d_real), logits=d_real))
        dc_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(d_fake), logits=d_fake))
        self.dc_loss = dc_loss_fake + dc_loss_real

        # Generator loss
        self.generator_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(d_fake), logits=d_fake))

        self.cost = norm * tf.reduce_mean(
            tf.nn.weighted_cross_entropy_with_logits(logits=preds_sub, targets=labels_sub, pos_weight=pos_weight))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate_1)  # Adam Optimizer

        all_variables = tf.trainable_variables()
        dc_var = [var for var in all_variables if 'dc_' in var.op.name]
        en_var = [var for var in all_variables if 'e_' in var.op.name]


        with tf.variable_scope(tf.get_variable_scope(), reuse=False):
            self.discriminator_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate_2,
                                                                  beta1=0.9, name='adam1').minimize(self.dc_loss, var_list=dc_var)#minimize(dc_loss_real, var_list=dc_var)

            self.generator_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate_2,
                                                              beta1=0.9, name='adam2').minimize(self.generator_loss,
                                                                                                var_list=en_var)

        self.cost = norm * tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=preds_sub, targets=labels_sub, pos_weight=pos_weight))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate_1)  # Adam Optimizer

        # Latent loss
        self.log_lik = self.cost
        self.kl = (0.5 / num_nodes) * tf.reduce_mean(tf.reduce_sum(1 + 2 * model.z_log_std - tf.square(model.z_mean) -
                                                                   tf.square(tf.exp(model.z_log_std)), 1))
        self.cost -= self.kl

        self.opt_op = self.optimizer.minimize(self.cost)
        self.grads_vars = self.optimizer.compute_gradients(self.cost)


def set_optimizer(model, discriminator, placeholders, pos_weight, norm, num_nodes, lr, dc_lr):
    opt = OptimizerVAE(preds = model.reconstructions,
                        labels = tf.reshape(tf.sparse_tensor_to_dense(placeholders['adj_orig'],
                                                                     validate_indices=False), [-1]),
                        model = model, 
                        num_nodes = num_nodes,
                        pos_weight = pos_weight,
                        norm = norm,
                        d_real = discriminator.construct(placeholders['real_distribution']),
                        d_fake = discriminator.construct(model.embeddings, reuse=True),
                        learning_rate_1 = lr,
                        learning_rate_2 = dc_lr)
    return opt










# =============== Building model updating function ===============
# ================================================================

def update(model, opt, sess, adj_norm, adj_label, features, placeholders, adj, dropout_rate, latent_dim):
    # Construct feed dictionary
    feed_dict = construct_feed_dict(adj_norm, adj_label, features, placeholders)
    feed_dict.update({placeholders['dropout']: dropout_rate})

    feed_dict.update({placeholders['dropout']: 0})

    emb_hidden1 = sess.run(model.h1, feed_dict=feed_dict)
    emb_hidden2 = sess.run(model.embeddings, feed_dict=feed_dict)

    z_real_dist = np.random.randn(adj.shape[0], latent_dim)
    feed_dict.update({placeholders['real_distribution']: z_real_dist})

    for j in range(20):
        _, reconstruct_loss = sess.run([opt.opt_op, opt.cost], feed_dict=feed_dict)
    d_loss, _ = sess.run([opt.dc_loss, opt.discriminator_optimizer], feed_dict=feed_dict)
    g_loss, _ = sess.run([opt.generator_loss, opt.generator_optimizer], feed_dict=feed_dict)

    avg_cost = reconstruct_loss

    return emb_hidden1, emb_hidden2, avg_cost










# =============== Others ===============
# ======================================

def ranked_partial(adj_orig, adj_rec, coord, size):  #size是list，[3,5]代表把总图切成宽3份(x)、高5份(y)的子图
    x_gap = (coord[:,0].max()-coord[:,0].min())/size[0]
    y_gap = (coord[:,1].max()-coord[:,1].min())/size[1]
    x_point = np.arange(coord[:,0].min(), coord[:,0].max(), x_gap).tolist()
    if coord[:,0].max() not in x_point:
        x_point += [coord[:,0].max()]
    y_point = np.arange(coord[:,1].min(), coord[:,1].max(), y_gap).tolist()
    if coord[:,1].max() not in y_point:
        y_point += [coord[:,1].max()]

    x_interval = [[x_point[i],x_point[i+1]] for i in range(len(x_point)) if i!=len(x_point)-1]
    y_interval = [[y_point[i],y_point[i+1]] for i in range(len(y_point)) if i!=len(y_point)-1]

    id_part = {}
    subregion_mark = []
    for i in x_interval:
        for j in y_interval:
            id_list = np.where((coord[:,0]>=i[0]) & (coord[:,0]<i[1]) & (coord[:,1]>=j[0]) & (coord[:,1]<j[1]))[0].tolist()  #左开右闭，上开下闭
            adj_orig_tmp = adj_orig[id_list,:][:,id_list]
            adj_rec_tmp = adj_rec[id_list,:][:,id_list]
            if adj_orig_tmp.shape[0]*adj_orig_tmp.shape[1] == 0:
                break
            else:
                diff = np.where((adj_orig_tmp-adj_rec_tmp)!=0)[0].shape[0] / (adj_orig_tmp.shape[0]*adj_orig_tmp.shape[1])
                id_part[diff] = id_list
                subregion_mark.append([i,j])

    return sorted(id_part.items(), key=lambda item:item[0], reverse=True), subregion_mark






