#!/usr/bin/env python
"""
TODO:
# Author: 
# Created Time : 

# File Name: 
# Description: 

"""


import os
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='TODO')

    # IO and norm options
    parser.add_argument('--exp', '-e', type=str, help='TODO: Input gene expression data path')
    parser.add_argument('--adj', '-a', type=str, help='Input adjacency matrix data path')
    parser.add_argument('--coordinate', '-c', type=str, help='Input cell coordinate data path')
    parser.add_argument('--reference', '-r', type=str, help='Input cell type label path')
    parser.add_argument('--verbose', action='store_true', help='Print loss of training process')
    parser.add_argument('--outdir', '-o', type=str, default='output/', help='Output path')
    parser.add_argument('--outpostfix', '-n', type=str, help='The postfix of the output file')
    parser.add_argument('--log_add_number', type=int, default=None, help='Perform log10(x+log_add_number) transform')
    parser.add_argument('--fil_gene', type=int, default=None, help='Remove genes expressed in less than fil_gene cells')
    parser.add_argument('--latent_feature', '-l', default=None, help='')

    # Training options
    parser.add_argument('--test_ratio', '-t', type=int, default=0.1, help='Testing set ratio 大于1时，代表边数；小于1时，代表比例 (default: 0.1)')    
    parser.add_argument('--iteration', '-i', type=int, default=2, help='Iteration (default: 40)')
    parser.add_argument('--encode_dim', type=int, nargs=2, default=[125, 125], help='Encoder structure')
    parser.add_argument('--regularization_dim', type=int, nargs=2, default=[150, 125], help='Adversarial regularization structure') #TODO:[125, 125, ]
    parser.add_argument('--lr1', type=float, default=0.0004, help='TODO')
    parser.add_argument('--lr2', type=float, default=0.0008, help='TODO')
    parser.add_argument('--weight_decay', type=float, default=0, help='Weight for L2 loss on latent features')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability)')
    parser.add_argument('--features', type=int, default=1, help='Whether to use features (1) or not (0)')
    parser.add_argument('--seed', type=int, default=7, help='Random seed for repeat results') #TODO:50,7,1
    parser.add_argument('--activation', type=str, default='relu', help="Activation function of hidden units (default: relu)")
    parser.add_argument('--init', type=str, default='glorot_uniform', help="Initialization method for weights (default: glorot_uniform)")
    parser.add_argument('--optimizer', type=str, default='Adam', help="Optimization method (default: Adam)")

    # Clustering options
    parser.add_argument('--cluster', action='store_true', help='TODO')
    parser.add_argument('--cluster_num', type=int, default=None, help='TODO')    

    parser.add_argument('--gpu', '-g', type=int, default=0, help='Select gpu device number for training')
    # tf.test.is_built_with_cuda()

    # parser.set_defaults(transpose=False,
    #                     testsplit=False,
    #                     saveweights=False,
    #                     sizefactors=True,
    #                     batchnorm=True,
    #                     checkcounts=True,
    #                     norminput=True,
    #                     hyper=False,
    #                     debug=False,
    #                     tensorboard=False,
    #                     loginput=True)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    # Select gpu device number  
    import os 
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)  #if you use GPU, you must be sure that there is at least one GPU available in your device
    except:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  #set only using cpu

    # Import modules
    try:
        import tensorflow as tf  #import tf and the rest module after parse_args() to make argparse help show faster
    except ImportError:
        raise ImportError('DeepLinc requires TensorFlow. Please follow instructions'
                          ' at https://www.tensorflow.org/install/ to install'
                          ' it.')
    import numpy as np
    import pandas as pd
    import scipy.sparse as sp
    from scipy.spatial.distance import pdist, squareform
    import copy
    from deeplinc.io import *
    from deeplinc.plot import *
    from deeplinc.utils import sparse2tuple, packed_data, set_placeholder, set_optimizer, update, ranked_partial
    from deeplinc.models import Deeplinc, Discriminator
    from deeplinc.metrics import linkpred_metrics, select_optimal_threshold
    from deeplinc.enrichment import connection_number_between_groups, generate_adj_new_long_edges, edges_enrichment_evaluation
    from deeplinc.sensitivity import get_sensitivity
    from deeplinc.cluster import clustering

    # Set random seed
    seed = args.seed
    np.random.seed(seed)
    tf.set_random_seed(seed)

    # Import and pack datasets
    exp_df, adj_df = read_dataset(args.exp, args.adj, args.fil_gene, args.log_add_number)
    exp, adj = exp_df.values, adj_df.values
    coord_df = read_coordinate(args.coordinate)
    coord = coord_df.values
    cell_label_df = read_cell_label(args.reference)
    cell_label = cell_label_df.values

    feas = packed_data(exp, adj, args.test_ratio)
    var_placeholders = set_placeholder(feas['adj_train'], args.encode_dim[1])

    # Output some basic information
    cell_num = exp.shape[0]
    gene_num = exp.shape[1]
    predefined_edge_num = np.where(adj==1)[0].shape[0]/2

    print("\n**************************************************************************************************************")
    print("  DeepLinc: De novo reconstruction of cell interaction landscapes from single-cell spatial transcriptome data  ")
    print("**************************************************************************************************************s\n")
    print("======== Parameters ========")
    print('Cell number: {}\nGene number: {}\nPredefined local connection number: {}\niteration: {}'.format(
            cell_num, gene_num, predefined_edge_num, args.iteration))
    print("============================")

    # Building model and optimizer
    # dims = []
    deeplinc = Deeplinc(var_placeholders, feas['num_features'], feas['num_nodes'], feas['features_nonzero'], args.encode_dim[0], args.encode_dim[1])
    deeplinc_discriminator = Discriminator(args.encode_dim[1], args.regularization_dim[0], args.regularization_dim[1])
    opt = set_optimizer(deeplinc, deeplinc_discriminator, var_placeholders, feas['pos_weight'], feas['norm'], feas['num_nodes'], args.lr1, args.lr2)

################################################################################################################
    # Fitting model
    # Saver
    saver = tf.train.Saver(max_to_keep=1)

    # Initialize session
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # Metrics list
    train_loss = []
    test_ap = []
    # latent_feature = None
    max_test_ap_score = 0

    # Train model
    for epoch in range(args.iteration):

        emb_hidden1_train, emb_hidden2_train, avg_cost_train = update(deeplinc, opt, sess, feas['adj_norm'], feas['adj_label'], feas['features'], var_placeholders, feas['adj_train'], args.dropout, args.encode_dim[1])
        train_loss.append(avg_cost_train)

        lm_train = linkpred_metrics(feas['test_edges'], feas['test_edges_false'])

        roc_score, ap_score, acc_score, _ = lm_train.get_roc_score(emb_hidden2_train, feas)
        test_ap.append(ap_score)
        print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(avg_cost_train), "test_roc=", "{:.5f}".format(roc_score), "test_ap=", "{:.5f}".format(ap_score))

        if ap_score > max_test_ap_score:
            max_test_ap_score = ap_score

            saver.save(sess, './model/'+args.outpostfix, global_step=epoch+1)

            np.save("emb_hidden1_"+str(epoch+1)+'.npy', emb_hidden1_train)
            np.save("emb_hidden2_"+str(epoch+1)+'.npy', emb_hidden2_train)

            latent_feature = copy.deepcopy(emb_hidden2_train)

    plot_evaluating_metrics(test_ap, "epoch", "score", ["AUPRC"], "AUPRC")
    write_pickle(feas, 'feas')

################################################################################################################

    ### Output ###
    # 3.1. 概率、连接和阈值
    adj_reconstructed_prob, adj_reconstructed, _, _, all_acc_score, max_acc_score, optimal_threshold = select_optimal_threshold(feas['test_edges'], feas['test_edges_false']).select(latent_feature, feas)
    print(optimal_threshold)
    print(max_acc_score)

    write_json(all_acc_score, 'acc_diff_threshold_'+args.outpostfix)
    write_json({'optimal_threshold':optimal_threshold,'max_acc_score':max_acc_score}, 'threshold_'+args.outpostfix)
    write_csv_matrix(adj_reconstructed_prob, 'adj_reconstructed_prob_'+args.outpostfix)
    write_csv_matrix(adj_reconstructed, 'adj_reconstructed_'+args.outpostfix)

    # 3.2. 距离分布、距离矩阵
    adj_diff = adj - adj_reconstructed
    adj_diff = (adj_diff == -1).astype('int')
    adj_diff = sp.csr_matrix(adj_diff)

    dist_matrix_rongyu = pdist(coord, 'euclidean')
    dist_matrix = squareform(dist_matrix_rongyu)

    #这里好像直接adj_diff乘上dist_matrix就可以
    new_edges = sparse2tuple(sp.triu(sp.csr_matrix(adj_diff)))[0]
    all_new_edges_dist = dist_matrix[new_edges[:,0].tolist(),new_edges[:,1].tolist()]
    plot_histogram(all_new_edges_dist, xlabel='distance', ylabel='density', filename='all_new_edges_distance', color="coral")
    write_csv_matrix(dist_matrix*adj_diff, 'all_new_edges_dist_matrix')

    # 3.3. 连接可视化
    id_subgraph, _ = ranked_partial(adj, adj_reconstructed, coord, [10,15])  #返回的是[(diff,[id_list]),(diff,[id_list])...]这种形式
                                                                                                #adj_rec1:[10,15], adj_rec2:[3,5]
    rank = 0
    for item in id_subgraph:
        cell_type_subgraph = cell_label[item[1],:][:,[0,1]]
        cell_type_subgraph[:,0] = np.array(list(range(cell_type_subgraph.shape[0]))) + 1  #需要对X重新生成细胞的id，这里以1开始
        coord_subgraph = coord[item[1],:]
        adj_reconstructed_subgraph = adj_reconstructed[item[1],:][:,item[1]]
        rank += 1
        # adjacency_visualization(cell_type_subgraph, coord_subgraph, adj_reconstructed_subgraph, filename='spatial_network_rank'+str(rank)+'_diff'+str('%.3f'%item[0]))

    # 4. 互作强度
    cutoff_distance = np.percentile(all_new_edges_dist,99)

    connection_number, _ = connection_number_between_groups(adj, cell_label[:,1])
    write_csv_matrix(connection_number, 'connection_number_between_groups')

    adj_new_long_edges = generate_adj_new_long_edges(dist_matrix, new_edges, all_new_edges_dist, cutoff_distance)
    write_csv_matrix(adj_new_long_edges.todense(), 'adj_new_long_edges')

    print('------permutations calculating------')
    cell_type_name = [np.unique(cell_label[cell_label[:,1]==i,2])[0] for i in np.unique(cell_label[:,1])]
    test_result, _, _, _ = edges_enrichment_evaluation(adj, cell_label[:,1], cell_type_name, edge_type='all edges')
    write_csv_matrix(test_result, 'all_edges_enrichment_evaluation', colnames=['cell type A','cell type B','average_connectivity','significance'])
    test_result, _, _, _ = edges_enrichment_evaluation(adj_new_long_edges.toarray(), cell_label[:,1], cell_type_name, edge_type='long edges', dist_matrix=dist_matrix, cutoff_distance=cutoff_distance)
    write_csv_matrix(test_result, 'long_edges_enrichment_evaluation', colnames=['cell type A','cell type B','connection_number','significance'])

    # 5. 敏感性
    # get_sensitivity(exp_df, feas, './model/'+args.outpostfix)

    # 6. 聚类
    if args.cluster:
        cluster_num = args.cluster_num
        cluster_label = clustering(latent_feature, cluster_num)
        write_csv_matrix(cluster_label, 'label', colnames=['cell_id','cluster_id'])







