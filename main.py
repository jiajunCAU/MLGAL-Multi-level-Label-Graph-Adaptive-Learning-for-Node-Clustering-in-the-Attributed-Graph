SEED = 42
import argparse
import time
import random
import numpy as np
import scipy.sparse as sp
import torch
from sklearn import manifold
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
parser.add_argument('--epochs', type=int, default=400, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
parser.add_argument('--l', type=float, default=0.5, help='Initial learning rate.')
parser.add_argument('--hidden', type=int, default=500, help='Number of units in hidden gcn layer')
parser.add_argument('--pos_st', type=float, default=0.005, help='Upper Threshold start.')
parser.add_argument('--pos_ed', type=float, default=0.001, help='Lower Threshold start.')
parser.add_argument('--neg_st', type=float, default=0.9, help='Upper Threshold end.')
parser.add_argument('--neg_ed', type=float, default=0.5, help='Lower Threshold end.')
parser.add_argument('--upd', type=int, default=10, help='Update epoch.')
parser.add_argument('--bs', type=int, default=1000, help='Batchsize.')
parser.add_argument('--dataset', type=str, default='cora', help='type of dataset.')
parser.add_argument('--fx', type=str, default='x', help='type of adaptive fx.')
parser.add_argument('--k', type=float, default=0, help='filter parameter 1')
parser.add_argument('--r', type=float, default=1, help='random is or not')
parser.add_argument('--tao', type=float, default=1, help='filter parameter 2')
parser.add_argument('--nc', type=int, default=3, help='cluster nums.')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
args = parser.parse_args()
# cuda_use = not args.no_cuda and torch.cuda.is_available()

color_list = ["r","g","b","m","k","y","c","maroon","mistyrose","purple","steelblue","olive"
    ,"peru","sienna","palegreen","navy","deepskyblue"
]

def pos_f(i,t):
    if t == "ex":
        b = np.log(args.pos_ed/args.pos_st) / (args.epochs/10)
        y = args.pos_st*np.e**(b*i)
        return y
    elif t == "a/x":
        up_n = args.epochs/10
        a = (args.pos_st-args.pos_ed)/((up_n-1)/up_n)
        b = args.pos_st-a
        return a/(i+1) + b
    elif t == "ax2":
        up_n = args.epochs/10
        st,ed = args.pos_st,args.pos_ed
        c = st
        a = (st-ed)/(up_n**2)
        b = -2*a*up_n
        return a*i*i+b*i+c
    elif t == "x":
        up_n = args.epochs/10
        st,ed = args.pos_st,args.pos_ed
        return (ed-st)/up_n*i+st
    else:
        return "non choose fx"

def clustering_r(Cluster, feature, true_labels):
    # f_adj = np.matmul(feature,feature.T)
    f_adj = feature
    predict_labels = Cluster.fit_predict(f_adj)
    
    cm = clustering_metrics(true_labels, predict_labels)
    db = -metrics.davies_bouldin_score(f_adj, predict_labels)
    acc, nmi, adj,q = cm.evaluationClusterModelFromLabel(tqdm)

    return db, acc, nmi, adj,predict_labels,q


def Tsne(X):
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
    X_tsne = tsne.fit_transform(X)
    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化
    return X_norm

def draw_kmeans(features,Cluster,name):
    predict_labels = Cluster.fit_predict(np.matmul(features, features.T) )
    visual_x = Tsne(features)
    plt.figure(figsize=(10,8))
    color_list = np.array(["lightgreen","steelblue","darkorange","lightblue","seagreen","lightcoral","deepskyblue"])
    plt.yticks(fontsize=20)
    plt.xticks(fontsize=20)
    counter = [predict_labels.tolist().count(i) for i in range(6)]
    couter_idx = [sorted(counter).index(u) for u in counter]
    color_list_sorted = color_list[couter_idx]
    for i in range(6):
        plt.scatter(
            visual_x[:,0][np.where(predict_labels==i)],
            visual_x[:,1][np.where(predict_labels==i)],
            color = color_list_sorted[i]
        )
    plt.savefig("./fig/tsne_{}.pdf".format(name),bbox_inches='tight')


def main(args):
    data_info = {
    "cora":[2708,1433,7,0.97],
    "citeseer":[3327,3703,6,1],
    "wiki":[2405,4973,17,0.5],
    "pubmed":[19717,500,3,1.04]
    }
    ## define the Cluster ------------------------------------------------------------
    Cluster = KMeans(n_clusters= data_info[args.dataset][2])
    # Cluster = SpectralClustering(n_clusters=data_info[args.dataset][2], affinity = 'precomputed', random_state=0)
    
    best_acc = 0
    ## load data------------------------------------------------------------
    adj, adj_self, features, true_labels, idx_train, idx_val, idx_test = load_data(args.dataset)

    degree = adj.sum(1)
    print("edges:{}".format(adj.sum()))
    from collections import Counter
    res = Counter(degree)
    print(sorted(res.items(), key = lambda x:x[0]) )
    print("load data end")
    # draw_kmeans(features.numpy(),Cluster,"initial")
    raw_feat = features.numpy()

    # lap smoothing ------------------------------------------------------------
    low_freq = features.numpy()
    adj_sp = sp.coo_matrix(adj)
    ident = sp.eye(adj_sp.shape[0])    
    adj_self = adj_sp + ident
    D_self = np.array(adj_self.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(D_self, -0.5).flatten())
    adj_normalized = adj_self.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    laplacian = ident - adj_normalized
    L_sym = laplacian.todense()
    if args.k == 0:
        k = get_k(features.numpy(),L_sym)
    else:
        k = args.k
    print("k:{}".format(k))
    # k = data_info[args.dataset][-1]
    adjs1 = [ident-(k * laplacian) for _ in range(args.gnnlayers)]
    for a in adjs1:
        low_freq = a.dot(low_freq)
    
    # transfer data to GPU if cuda is available ------------------------------------------------------------
    features = torch.FloatTensor(features.numpy()).cuda()
    up_nums = 0
    n_class = data_info[args.dataset][2]
    pos_rate = pos_f(up_nums,args.fx)
    pos_inds,neg_inds = update_similarity(normalize(low_freq),pos_rate)
    print(pos_inds.shape,neg_inds.shape)

    db, acc, nmi, adj_score,labels,pred = clustering_r(Cluster, raw_feat, true_labels.numpy())
    print("Based Acc:{},NMI:{},ARI:{}".format(acc,nmi,adj_score))
    
    db, acc, nmi, adj_score,labels,pred = clustering_r(Cluster, low_freq, true_labels.numpy())
    print("Filtered Acc:{},NMI:{},ARI:{}".format(acc,nmi,adj_score))
    sim_ = get_sim(labels,n_class)
    labels = torch.LongTensor(pred)
    n_nodes = adj.shape[0]
    
    print('Start Training...')
    best_acc = 0
    patience = 0
    model = LinTrans(in_dim = features.shape[1],hidden_dim = args.hidden,out_dim = 7).cuda()
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    truth_label2 = get_sim(true_labels.numpy(), n_class)


    for epoch in tqdm(range(args.epochs)):
        t_start = time.time()
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
            batch_pred2 = model.dcs(zx, zy)
            loss1 = F.binary_cross_entropy_with_logits(batch_pred, batch_label)
            loss2 = F.binary_cross_entropy_with_logits(batch_pred2, batch_label2)
            loss = args.l*loss1 + (1-args.l)*loss2
            loss.backward()
            optimizer.step()
            
            if epoch == 0 and start_idx == 0:
                from thop import profile
                flops, params = profile(model, inputs = (x,))
                print("#params:{}, FLOPs:{}".format(params/(1000**2),flops/(1000**3)))
                print(x.shape)
            start_idx += batch_size
            if end_idx < len(pos_inds) and end_idx + batch_size > len(pos_inds):
                end_idx += len(pos_inds) - end_idx
            else:
                end_idx += batch_size
        if epoch == 0 :
            print("training time:{}".format(time.time() - t_start))
        if (epoch+1) %  10 == 0 :
            model.eval()
            hidden_emb = model(features).cpu().data.numpy()
            up_nums += 1
            pos_rate = pos_f(up_nums,args.fx)
            pos_inds, neg_inds = update_similarity(normalize(hidden_emb), pos_rate)
            db, acc, nmi, adj_score,labels,pred = clustering_r(Cluster, hidden_emb, true_labels.numpy())
            # labels = torch.LongTensor(labels).cuda()
            sim_ = get_sim(labels, n_class)
            labels = torch.LongTensor(pred)
            if patience >= 20:
                break
            
            print("Epoch: {},acc:{:5f},nmi:{:.5f},adj:{:.5f},loss:{:.5f}".format(epoch,acc,nmi,adj_score,loss.cpu().detach()))
            patience += 1
            if acc > best_acc:
                best_acc = acc
                best_nmi = nmi
                best_ari = adj_score
                patience = 0
            bin_false = torch.sum(sim_ != truth_label2)/n_nodes**2
            multi_false = torch.sum(labels != true_labels)/n_nodes
            print("binary-label false propotion:{}".format(bin_false))
            print("multi-label false propotion:{}".format(multi_false))
            print("-*"*10)
            
    
if __name__ == "__main__":
    main(args)          