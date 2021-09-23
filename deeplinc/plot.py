#!/usr/bin/env python
"""
TODO:
# Author: 
# Created Time : 

# File Name: 
# Description: 

"""


import os
os.environ['PYTHONHASHSEED'] = '0'

import matplotlib
havedisplay = "DISPLAY" in os.environ
if havedisplay:  #if you have a display use a plotting backend
    matplotlib.use('TkAgg')
else:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import seaborn as sns
import networkx as nx


def plot_histogram(data, xlabel, ylabel, filename, ifhist=True, ifrug=False, ifkde=False, ifxlog=False, ifylog=False, figsize=(15,10), color="cornflowerblue"):
    figure, ax = plt.subplots(figsize=figsize, dpi=100)
    sns.distplot(data, hist=ifhist, rug=ifrug, kde=ifkde, color=color)
    if ifxlog:
        plt.xscale("log")
    if ifylog:
        plt.yscale("log")  #plt.yscale("log",basey=10), where basex or basey are the bases of log

    plt.tick_params(labelsize=30)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Arial') for label in labels]

    font1 = {'family':'Arial','weight':'normal','size':30,}
    plt.xlabel(xlabel, font1)
    plt.ylabel(ylabel, font1)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    plt.savefig(filename+'.tif')
    plt.close()


def plot_evaluating_metrics(metrics_list, fig_xlabel, fig_ylabel, fig_legend, filename, figsize=(13,9), color='orange'):
    figure, ax = plt.subplots(figsize=figsize, dpi=100)

    # ax = plt.subplot(111, facecolor='linen')

    # 多项式拟合
    f1 = np.polyfit(range(1,len(metrics_list)+1), np.array(metrics_list), 6)
    p1 = np.poly1d(f1)
    yvals1 = p1(range(1,len(metrics_list)+1))
    # 指定函数拟合
    # def func(x,a,b,c):  #指数函数拟合
    #     return a*np.exp(b/x)+c
    # def func(x,a,b,c):  #非线性最小二乘法拟合
    #     return a*np.sqrt(x)*(b*np.square(x)+c)
    # popt1, pcov1 = curve_fit(func, range(1,len(metrics_list)+1), np.array(metrics_list))  #popt里面是拟合系数：a=popt[0]，b=popt[1]，c=popt[2]
    # yvals1 = func(range(1,len(metrics_list)+1), *popt1)

    ax.plot(range(1,len(metrics_list)+1), yvals1, color=color, linestyle='-', linewidth=6)

    plt.tick_params(labelsize=20)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Arial') for label in labels]

    font1 = {'family':'Arial','weight':'normal','size':30,}
    plt.xlabel(fig_xlabel, font1)
    plt.ylabel(fig_ylabel, font1)

    font2 = {'family':'Arial','size':25,}
    leg = plt.legend(fig_legend, bbox_to_anchor=(1.02, 0), loc='lower right', borderaxespad=0, prop=font2)
    leg.get_frame().set_linewidth(0.0)

    plt.savefig('{}.tif'.format(filename))
    plt.close()


def plot_cluster_score(cluster_num, score, xlabel, ylabel, filename, line_mode="bx-"):
    figure, ax = plt.subplots(figsize=(15.36,7.67), dpi=100)

    plt.plot(cluster_num, score, line_mode)

    plt.tick_params(labelsize=20)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Arial') for label in labels]

    font1 = {'family':'Arial','weight':'normal','size':30,}
    plt.xlabel(xlabel, font1)
    plt.ylabel(ylabel, font1)

    figure.subplots_adjust(right=0.75)

    plt.savefig('{}.tif'.format(filename))
    plt.close()


def adjacency_visualization(cell_type, coord, adj, filename):
    colors = ['beige', 'royalblue', 'maroon', 'olive', 'tomato', 'mediumpurple', 'paleturquoise', 'brown', 
              'firebrick', 'mediumturquoise', 'lightsalmon', 'orchid', 'dimgray', 'dodgerblue', 'mistyrose', 
              'sienna', 'tan', 'teal', 'chartreuse']

    X = np.hstack((cell_type, coord))

    #获取节点信息
    class_cellid = []
    for num in range(55):
        class_cellid.append(list(X[X[:,1]==num, 0].astype('int')-1))

    #获取坐标信息
    class_coord = []
    for num in range(55):
        class_coord.append(X[X[:,1]==num, 2:].tolist())

    #获取节点与坐标之间的映射关系，分cell type存储用于画节点
    pos_usedfor_nodes = []
    for num in range(55):
        pos_usedfor_nodes.append(dict(zip(class_cellid[num],class_coord[num])))

    #获取节点与坐标之间的映射关系，不分cell type存储用于画边
    pos_usedfor_edges = {}
    pos_usedfor_edges = pos_usedfor_nodes[0].copy()
    for num in range(1,55):
        pos_usedfor_edges.update(pos_usedfor_nodes[num])

    edges_tmp1 = np.where(adj == 1)
    edges1 = []
    edges_cluster = [edges1]
    edges_tmp_cluster = [edges_tmp1]

    for z2,z3 in enumerate(edges_tmp_cluster):
        edges_num = z3[0].shape[0]
        for z4 in range(0, edges_num):
            edges_cluster[z2].append((z3[0][z4],z3[1][z4]))

    # 循环画三种方式计算的连接图
    for z5 in edges_cluster:
        ax = plt.axes([0.042, 0.055, 0.9, 0.9])#[xmin,ymin,xmax,ymax]
        ax.set_xlim(min(X[:,2])-15,max(X[:,2])+15)
        ax.set_ylim(min(X[:,3])-15,max(X[:,3])+15)
        ax.xaxis.set_major_locator(plt.MultipleLocator(400.0))
        ax.xaxis.set_minor_locator(plt.MultipleLocator(200.0))#设置x从坐标间隔
        ax.yaxis.set_major_locator(plt.MultipleLocator(400.0))#设置y主坐标间隔
        ax.yaxis.set_minor_locator(plt.MultipleLocator(200.0))#设置y从坐标间隔
        ax.grid(which='major', axis='x', linewidth=0.3, linestyle='-', color='0.3')#由每个x主坐标出发对x主坐标画垂直于x轴的线段 
        ax.grid(which='minor', axis='x', linewidth=0.1, linestyle='-', color='0.1')
        ax.grid(which='major', axis='y', linewidth=0.3, linestyle='-', color='0.3') 
        ax.grid(which='minor', axis='y', linewidth=0.1, linestyle='-', color='0.1') 
        ax.set_xticklabels([i for i in range(int(min(X[:,2])-15),int(max(X[:,2])+15),400)])
        ax.set_yticklabels([i for i in range(int(min(X[:,3])-15),int(max(X[:,3])+15),400)])
        G = nx.Graph()
        for i2 in pos_usedfor_nodes:
            cellid_oneclass = list(i2.keys())
            nx.draw_networkx_nodes(G, i2, cellid_oneclass, node_size=150, node_color=colors[pos_usedfor_nodes.index(i2)])  #HDST_cancer和seqFISH的node_size是150，MERFISH是145
        nx.draw_networkx_edges(G, pos_usedfor_edges, z5, width=0.8)
        nx.draw(G)

        plt.savefig(filename + '.tif')


def plot_top10_gene_sensitivity(occlu_deta_score, xlabel, ylabel, filename, linewidth=1.5, figsize=(16,10), color='silver'):
    gene_list = sorted(occlu_deta_score.items(), key=lambda item:item[1], reverse=True)

    sorted_names = [i[0] for i in gene_list[:10]]
    sorted_scores = [i[1] for i in gene_list[:10]]
    data = {"sorted_names":sorted_names,"sorted_scores":sorted_scores}
    data = pd.DataFrame(data)

    figure, ax = plt.subplots(figsize=figsize, dpi=100)
    sns.barplot(data=data, x='sorted_names', y='sorted_scores', facecolor=color, linewidth=linewidth)

    plt.tick_params(labelsize=30)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Arial') for label in labels]

    plt.xticks(rotation=45)

    font1 = {'family':'Arial','weight':'normal','size':30,}
    ax.set_xlabel(xlabel, font1)
    ax.set_ylabel(ylabel, font1)

    figure.subplots_adjust(right=0.9)
    plt.subplots_adjust(left=0.15, right=0.9, top=0.95, bottom=0.31)
    plt.show()


def plot_spatial_cluster(cluster_label, coord, filename, figsize=(15.36,7.67)):
    colors = ['beige', 'royalblue', 'maroon', 'olive', 'tomato', 'mediumpurple', 'paleturquoise', 'brown', 
              'firebrick', 'mediumturquoise', 'lightsalmon', 'orchid', 'dimgray', 'dodgerblue', 'mistyrose', 
              'sienna', 'tan', 'teal', 'chartreuse']

    X = np.hstack((cluster_label, coord))

    figure, ax = plt.subplots(figsize=figsize, dpi=100)
    for cluster in np.unique(X[:,1]):
        plt.scatter(X[X[:,1] == cluster, 2], X[X[:,1] == cluster, 3], color=colors[int(cluster)], s=38, alpha = 1, label='D'+str(cluster))

    figure.subplots_adjust(right=0.67)
    plt.tick_params(labelsize=38)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Arial') for label in labels]

    font1 = {'family':'Arial','weight':'normal','size':38,}
    plt.xlabel('x coordinates', font1)
    plt.ylabel('y coordinates', font1)

    font2 = {'family':'Arial','size':38,}
    plt.legend(bbox_to_anchor=(1.005, 0), loc=3, borderaxespad=0, prop=font2)

    plt.savefig(filename + '.tif')




# def plot_embedding():







