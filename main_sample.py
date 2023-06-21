SEED = 42
import argparse
import time
import random
import math
import random, time
import numpy as np
import scipy.sparse as sp
import torch
np.random.seed(SEED)
torch.manual_seed(SEED)
from torch import feature_alpha_dropout, optim
import torch.nn.functional as F
from model import *
from utils import *
from sklearn.cluster import SpectralClustering, KMeans
from clustering_metric import clustering_metrics
from tqdm import tqdm
from sklearn.preprocessing import normalize, MinMaxScaler
from sklearn import metrics
import matplotlib.pyplot as plt
parser = argparse.ArgumentParser()
parser.add_argument('--gnnlayers', type=int, default=2, help="Number of gnn layers")
parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
parser.add_argument('--l', type=float, default=0.9, help='Initial learning rate.')
parser.add_argument('--hidden', type=int, default=500, help='Number of units in hidden gcn layer')
parser.add_argument('--pos_st', type=float, default=0.1, help='Upper Threshold start.')
parser.add_argument('--pos_ed', type=float, default=0.001, help='Lower Threshold start.')
parser.add_argument('--neg_st', type=float, default=0.9, help='Upper Threshold end.')
parser.add_argument('--neg_ed', type=float, default=0.5, help='Lower Threshold end.')
parser.add_argument('--upd', type=int, default=10, help='Update epoch.')
parser.add_argument('--bs', type=int, default=2000, help='Batchsize.')
parser.add_argument('--dataset', type=str, default='cora', help='type of dataset.')
parser.add_argument('--fx', type=str, default='x', help='type of adaptive fx.')
parser.add_argument('--k', type=float, default=1, help='filter parameter 1')
parser.add_argument('--spsize', type=int, default=2000, help='filter parameter 2')
parser.add_argument("--cluster", type=str, default="km", help='clustering algorithm')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
args = parser.parse_args()
# cuda_use = not args.no_cuda and torch.cuda.is_available()

color_list = ["r","g","b","m","k","y","c","maroon","mistyrose","purple","steelblue","olive"
    ,"peru","sienna","palegreen","navy","deepskyblue"
]


def clustering_r(Cluster, feature, true_labels):
    f_adj = feature
    # f_adj = np.matmul(feature,feature.T)
    predict_labels = Cluster.fit_predict(f_adj)
    
    cm = clustering_metrics(true_labels, predict_labels)
    db = -metrics.davies_bouldin_score(f_adj, predict_labels)
    acc, nmi, adj,q = cm.evaluationClusterModelFromLabel(tqdm)

    return db, acc, nmi, adj,predict_labels,q

def main(args):
    data_info = {
    "cora":[2708,1433,7],
    "citeseer":[3327,3703,6],
    "wiki":[2405,4973,17],
    "pubmed":[19717,500,3]
    }
    ## define the Cluster ------------------------------------------------------------
    if args.cluster == "km":
        Cluster = KMeans(n_clusters= data_info[args.dataset][2])
    elif args.cluster == "sp":
        Cluster = SpectralClustering(n_clusters=data_info[args.dataset][2], affinity = 'precomputed', random_state=0)

    best_acc = 0
    ## load data------------------------------------------------------------
    adj, adj_self, features, true_labels_raw, idx_train, idx_val, idx_test = load_data(args.dataset,3)
    print("load data end")

    # lap smoothing ------------------------------------------------------------
    low_freq = features.numpy()
    adj_sp = sp.coo_matrix(adj)
    ident = sp.eye(adj_sp.shape[0])    
    adj_ = adj_sp + ident
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    laplacian = ident - adj_normalized


    # extract low-frequency
    lap = laplacian.todense()
    print("k:",args.k)
    adjs1 = [ident-(args.k * laplacian) for _ in range(args.gnnlayers)]
    for a in adjs1:
        low_freq = a.dot(low_freq)
    # with open("./revise/adaptive_k_{}_new2.csv".format(args.dataset),"a+")as f:
    #         f.write("{:.4f}\t{:.4f}\n".format(args.k,cal_re(features.numpy(),lap)))
    # return
    n_class = data_info[args.dataset][2]
    node_idlist = [i for i in range(low_freq.shape[0])]
    t_start = time.time()
    model = LinTrans(in_dim = low_freq.shape[1],hidden_dim = args.hidden,out_dim = 7).cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    best_acc = 0
    patience = 0
    pos_rate = args.pos_st
    time_list = []
    truth_label2 = get_sim(true_labels_raw.numpy(), n_class)
    n_all = low_freq.shape[0]
    for epoch in tqdm(range(args.epochs)):
        time_st = time.time()
        sample_idx = random.sample(node_idlist, args.spsize)
        features = torch.FloatTensor(low_freq[sample_idx]).cuda()
        true_labels = true_labels_raw[sample_idx]
        out = model(features)

        labels = Cluster.fit_predict(out.cpu().detach().numpy())
        sim_ = get_sim(labels,n_class)
        # up_nums = 0
        # pos_rate = pos_f(up_nums,args.fx)  
        n_nodes = features.shape[0]
        # patience = 0
        pos_inds,neg_inds = update_similarity(normalize(low_freq[sample_idx]),pos_rate)
        
        model.train()
        batch_size = args.bs
        start_idx,end_idx = 0,0+batch_size
        while end_idx < len(pos_inds):
            sampled_neg = torch.LongTensor(np.random.choice(neg_inds, size=end_idx-start_idx)).cuda() 
            sampled_pos = torch.LongTensor(pos_inds[start_idx:end_idx]).cuda()
            train_batch = torch.cat([sampled_pos,sampled_neg],0)
            optimizer.zero_grad()
            xind = train_batch // n_nodes
            yind = train_batch % n_nodes
            x = torch.index_select(features, 0, xind)
            y = torch.index_select(features, 0, yind)
            zx = model(x)
            zy = model(y)
            batch_label = torch.cat((torch.ones(end_idx-start_idx), torch.zeros(end_idx-start_idx))).cuda()
            batch_pred = model.dcs(zx, zy)


            batch_label2 = sim_[xind,yind]
            loss1 = F.binary_cross_entropy_with_logits(batch_pred, batch_label)
            loss2 = F.binary_cross_entropy_with_logits(batch_pred, batch_label2)
            loss = args.l*loss1 + (1-args.l)*loss2
            loss.backward()
            optimizer.step()
            start_idx += batch_size
            if end_idx < len(pos_inds) and end_idx + batch_size > len(pos_inds):
                end_idx += len(pos_inds) - end_idx
            else:
                end_idx += batch_size

        time_list.append(time.time()-time_st)
        if epoch % 10  == 0 :
            model.eval()
            hidden_emb = model(torch.FloatTensor(low_freq).cuda()).cpu().data.numpy()
            t_start = time.time()
            db, acc, nmi, adj_score,labels,pred = clustering_r(Cluster, hidden_emb, true_labels_raw.numpy())
            print("ACC:{:.4f}\t|\tNMI:{:.4f}\t|\tARI:{:.4f}\t|\tTime:{:.4f}\n".format(acc, nmi, adj_score, time.time()-t_start))
            if acc <= best_acc:
                patience += 1
            if patience >= 15:
                break
            if acc > best_acc:
                best_acc = acc
                best_nmi = nmi
                best_ari = adj_score
                patience = 0
            sim_ = get_sim(labels, n_class)
            labels = torch.LongTensor(pred)
            bin_false = torch.sum(sim_ != truth_label2)/n_all**2
            multi_false = torch.sum(labels != true_labels_raw)/n_all
            print("binary-label false propotion:{}".format(bin_false))
            print("multi-label false propotion:{}".format(multi_false))
            print("-*"*10)
            


if __name__ == "__main__":
        main(args)     
               