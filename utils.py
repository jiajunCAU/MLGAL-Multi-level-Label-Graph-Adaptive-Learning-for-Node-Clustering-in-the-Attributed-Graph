import pickle as pkl
import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score
import sklearn.preprocessing as preprocess
from clustering_metric import *
from sklearn.manifold import TSNE
from sklearn import manifold,datasets
from tqdm import *
import math

def update_similarity(z, pos_rate):
    neg_rate = 1 - pos_rate
    z = torch.FloatTensor(z).cuda()
    cosine = torch.matmul(z, z.T).cpu().detach().numpy()
    # row, col = np.diag_indices_from(cosine)
    # cosine[row,col] = 0
    cosine = cosine.reshape([-1,])
    pos_num = int(pos_rate * len(cosine))
    neg_num = int(neg_rate * len(cosine))
    pos_inds = np.argpartition(-cosine, pos_num)[0:pos_num]
    neg_inds = np.argpartition(cosine, neg_num)[0:neg_num]
    
    return np.array(pos_inds), np.array(neg_inds)

def upd_sim(z, pos_thre, neg_thre):
    z = torch.FloatTensor(z).cuda()
    cosine = torch.matmul(z, z.T).cpu().detach().numpy()
    row, col = np.diag_indices_from(cosine)
    cosine[row,col] = -1
    cosine = cosine.reshape([-1,])

    pos_inds = np.array(np.where(cosine>pos_thre)).reshape(-1)
    neg_inds = np.array(np.where(cosine<neg_thre) ).reshape(-1)
    
    return pos_inds, neg_inds

def labels2sim(label):
# 通过labels计算出similarity
    sim = [[] for i in range(len(label))]
    temp_dir = {}
    for i in range(len(label)):
        temp = []
        if label[i] in temp_dir:
            sim[i] = temp_dir[label[i]]
            continue
        for j in range(len(label)):
            if label[i] == label[j]:
                temp.append(1)
            else:
                temp.append(0)
        sim[i] = temp 
        temp_dir[label[i]] = temp
    sim = np.array(sim)
    row, col = np.diag_indices_from(sim)
    sim[row,col] = 0
    sim = sim.reshape([-1,])
    pos_inds = np.where(sim == 1)
    neg_inds = np.where(sim == 0)
    np.random.shuffle(pos_inds)
    np.random.shuffle(neg_inds)
    return np.array(pos_inds).reshape(-1),np.array(neg_inds).reshape(-1)
import time

def get_sim(label,n_class):
    t_start = time.time()
    # label = label.reshape(label.shape[0],1) 
    # label[np.where(label==4)] = 
    # label_sim = label*label.T
    # sim = np.zeros(label_sim.shape)
    # for i in range(n_class):
    #     sim[np.where(label_sim == i**2)] = 1
    # sim = torch.FloatTensor(sim).cuda()
    # print("cal sim time:{}".format(time.time()-t_start))
    # return sim
    
# 通过labels计算出similarity
    lens = len(label)
    sim = np.zeros([lens,lens])
    for i in range(n_class):
        ind = np.where(label==i)
        for i in ind[0]:
            for j in ind[0]:
                sim[(i,j)] = 1


    # sim = [[] for i in range(len(label))]
    # temp_dir = {}
    # for i in range(len(label)):
    #     temp = []
    #     if label[i] in temp_dir:
    #         sim[i] = temp_dir[label[i]]
    #         continue
    #     for j in range(len(label)):
    #         if label[i] == label[j]:
    #             temp.append(1)
    #         else:
    #             temp.append(0)
    #     sim[i] = temp 
    #     temp_dir[label[i]] = temp
    sim = torch.FloatTensor(sim).cuda()
    print("cal sim time:{}".format(time.time()-t_start))
    return sim

def get_k(X,L):
    k_list = []
    I = np.zeros([X.shape[0],X.shape[0]])
    for i in range(X.shape[1]):
        x = X[:,i]
        a = np.matmul(np.matmul(np.matmul(np.matmul(x.T,L.T),L),L),x).tolist()[0][0]
        b = -(np.matmul(np.matmul(np.matmul(x.T,L),L),x) + np.matmul(np.matmul(np.matmul(x.T,L.T),L),x)).tolist()[0][0]
        if a==0:
            k = 1
        else:
            k = -b/(2*a)
        if k>0 and k<2:
            k_list.append(k)
    return sum(k_list)/len(k_list)

def cal_smoothness(x,l):
    res = 0
    for i in range(x.shape[1]):
        xl = np.matmul(x[:,i].T,l)
        xlx = np.matmul(xl,x[:,i])
        xx = np.matmul(x[:,i].T,x[:,i])
        if not math.isnan(xlx.tolist()[0][0]/xx):
            res+= xlx.tolist()[0][0]
            # res+= xlx.tolist()[0][0] / xx
    return 1/res


def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def preprocess_graph(adj,lam1,lam2, layer, norm='sym', renorm=True):
    adj = sp.coo_matrix(adj)
    ident = sp.eye(adj.shape[0])
    
    if renorm:
        adj_ = adj + ident
    else:
        adj_ = adj
    
    rowsum = np.array(adj_.sum(1))
    if norm == 'sym':
        degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
        adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
        laplacian = ident - adj_normalized
    elif norm == 'left':
        degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -1.).flatten())
        adj_normalized = degree_mat_inv_sqrt.dot(adj_).tocoo()
        laplacian = ident - adj_normalized

    adjs = []
    for i in range(len(lam1)):
        adjs.append(lam1[i]*ident-(lam2[i] * laplacian))
    return adjs,adj_normalized

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

from collections import Counter
def load_data(dataset):
    # load the data: x, tx, allx, graph
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    if dataset == 'wiki':
        adj, features, label = load_wiki()
        
        ident = sp.eye(adj.shape[0])
        adj_ = adj + ident
        adj_ = torch.FloatTensor(np.array(adj_.todense()))
        labels = torch.LongTensor(label)
        train_nums = 210
        val_nums = 500
        idx_train = range(train_nums)
        idx_val = range(train_nums, train_nums + val_nums)
        idx_train = torch.LongTensor(idx_train)
        idx_val = torch.LongTensor(idx_val)

        return adj,adj_, features, labels, idx_train,idx_val,0

    for i in range(len(names)):
        '''
        fix Pickle incompatibility of numpy arrays between Python 2 and 3
        https://stackoverflow.com/questions/11305790/pickle-incompatibility-of-numpy-arrays-between-python-2-and-3
        '''
        with open("data/ind.{}.{}".format(dataset, names[i]), 'rb') as rf:
            u = pkl._Unpickler(rf)
            u.encoding = 'latin1'
            cur_data = u.load()
            objects.append(cur_data)
        # objects.append(
        #     pkl.load(open("data/ind.{}.{}".format(dataset, names[i]), 'rb')))
    x, y, tx, ty, allx, ally, graph = tuple(objects)
    
    test_idx_reorder = parse_index_file(
        "data/ind.{}.test.index".format(dataset))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(
            min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_extended
    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    adj_ = sp.coo_matrix(adj)
    ident = sp.eye(adj.shape[0])
    adj_ = adj_ + ident
    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]
    
    idx_test = test_idx_range.tolist()
    train_nums = 210
    val_nums = 400
    idx_train = range(train_nums)
    idx_val = range(train_nums, train_nums + val_nums)

    adj_ = torch.FloatTensor(np.array(adj_.todense()))
    features = torch.FloatTensor(features.todense())    
    labels = torch.LongTensor(np.argmax(labels, 1))

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)
    return adj,adj_, features, labels, idx_train, idx_val, idx_test

def Tsne_my(X):
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
    X_tsne = tsne.fit_transform(X)
         
    '''嵌入空间可视化'''
    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化
    # X_norm_x,X_norm_y = np.percentile(X_norm[:,0],95),np.percentile(X_norm[:,1],95)
    # X_norm = X_norm[np.where(X_norm[:,0]<X_norm_x)] 
    # X_norm = X_norm[np.where(X_norm[:,1]<X_norm_y)] 
     
    return X_norm




def clustering(Cluster, feature, true_labels):
    # feature = torch.FloatTensor(feature).cuda()
    # f_adj = torch.mm(feature,feature.T).cpu().detach().numpy()
    f_adj = feature
    # f_adj = np.matmul(feature,feature.T)
    # print(f_adj.shape)
    predict_labels = Cluster.fit_predict(f_adj)
    
    cm = clustering_metrics(true_labels, predict_labels)
    db = -metrics.davies_bouldin_score(f_adj, predict_labels)
    acc, nmi, adj,q = cm.evaluationClusterModelFromLabel(tqdm)

    return db, acc, nmi, adj,predict_labels,q


def load_wiki():
    f = open('data/graph.txt','r')
    adj, xind, yind = [], [], []
    for line in f.readlines():
        line = line.split()
        
        xind.append(int(line[0]))
        yind.append(int(line[1]))
        adj.append([int(line[0]), int(line[1])])
    f.close()
    ##print(len(adj))

    f = open('data/group.txt','r')
    label = []
    for line in f.readlines():
        line = line.split()
        label.append(int(line[1]))
    f.close()

    f = open('data/tfidf.txt','r')
    fea_idx = []
    fea = []
    adj = np.array(adj)
    adj = np.vstack((adj, adj[:,[1,0]]))
    adj = np.unique(adj, axis=0)
    
    labelset = np.unique(label)
    labeldict = dict(zip(labelset, range(len(labelset))))
    label = np.array([labeldict[x] for x in label])
    adj = sp.csr_matrix((np.ones(len(adj)), (adj[:,0], adj[:,1])), shape=(len(label), len(label)))

    for line in f.readlines():
        line = line.split()
        fea_idx.append([int(line[0]), int(line[1])])
        fea.append(float(line[2]))
    f.close()

    fea_idx = np.array(fea_idx)
    features = sp.csr_matrix((fea, (fea_idx[:,0], fea_idx[:,1])), shape=(len(label), 4973)).toarray()
    scaler = preprocess.MinMaxScaler()
    #features = preprocess.normalize(features, norm='l2')
    features = scaler.fit_transform(features)
    features = torch.FloatTensor(features)

    return adj, features, label