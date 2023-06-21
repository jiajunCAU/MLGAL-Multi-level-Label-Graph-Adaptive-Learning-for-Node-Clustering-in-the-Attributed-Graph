import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import normalize
class SampleDecoder(nn.Module):
    def __init__(self, act=F.relu):
        super(SampleDecoder, self).__init__()
        self.act = act

    def forward(self, zx, zy):
        sim = zx*zy
        sim = torch.mean(sim,dim=1)
        return sim.reshape(-1)


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(MLP, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.linear(x)

class Network(nn.Module):
    def __init__(self, resnet, feature_dim, class_num):
        super(Network, self).__init__()
        self.resnet = resnet
        self.feature_dim = feature_dim
        self.cluster_num = class_num
        self.instance_projector = nn.Sequential(
            nn.Linear(self.resnet.rep_dim, self.resnet.rep_dim),
            nn.ReLU(),
            nn.Linear(self.resnet.rep_dim, self.feature_dim),
        )
        self.cluster_projector = nn.Sequential(
            nn.Linear(self.resnet.rep_dim, self.resnet.rep_dim),
            nn.ReLU(),
            nn.Linear(self.resnet.rep_dim, self.cluster_num),
            nn.Softmax(dim=1)
        )

    def forward(self, x_i, x_j):
        h_i = self.resnet(x_i)
        h_j = self.resnet(x_j)

        z_i = normalize(self.instance_projector(h_i), dim=1)
        z_j = normalize(self.instance_projector(h_j), dim=1)

        c_i = self.cluster_projector(h_i)
        c_j = self.cluster_projector(h_j)

        return z_i, z_j, c_i, c_j

    def forward_cluster(self, x):
        h = self.resnet(x)
        c = self.cluster_projector(h)
        c = torch.argmax(c, dim=1)
        return c

class InstanceLoss(nn.Module):
    def __init__(self, batch_size, temperature):
        super(InstanceLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature

        self.mask = self.mask_correlated_samples(batch_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        mask = mask.bool()
        return mask

    def forward(self, z_i, z_j):
        N = 2 * self.batch_size
        z = torch.cat((z_i, z_j), dim=0)

        sim = torch.matmul(z, z.T) / self.temperature
        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_samples = sim[self.mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N

        return loss


class LinTrans(nn.Module):
    def __init__(self, in_dim,hidden_dim, out_dim, ):
        super(LinTrans, self).__init__()

        self.linear1 = nn.Linear(in_dim, hidden_dim)
        self.linear2 = nn.Linear(in_dim, out_dim)
        self.dcs = SampleDecoder()
        self.mlp = MLP(hidden_dim, out_dim)

    def scale(self, z):
        
        zmax = z.max(dim=1, keepdim=True)[0]
        zmin = z.min(dim=1, keepdim=True)[0]
        z_std = (z - zmin) / (zmax - zmin)
        z_scaled = z_std
    
        return z_scaled

    def forward(self, x1):
        x = self.linear1(x1) 
        x = self.scale(x)
        x = F.normalize(x)
        return x


class GCN_BASE(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, hidden,out_features, dropout=0., act=F.relu,nc = 10):
        super(GCN_BASE, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.act = act
        self.weight = nn.Parameter(torch.FloatTensor(in_features, hidden))
        # self.adj_lr = nn.Parameter(torch.ones(nc,hidden))
        self.MLP = nn.Linear(hidden,out_features)
        # self.MLP2 = nn.Linear(hidden,hidden)
        
        self.reset_parameters()
        # self.encoder1 = [nn.Linear(hidden,hidden).cuda()  for _ in range(nc)]       
        # self.encoder = nn.LSTM(hidden,hidden,batch_first = False)
        self.nc = nc

        
    def scale(self, z):
        zmax = z.max(dim=1, keepdim=True)[0]
        zmin = z.min(dim=1, keepdim=True)[0]
        z_std = (z - zmin) / (zmax - zmin)
        z_scaled = z_std
    
        return z_scaled

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, input, adj):

        output = torch.mm(input, self.weight)    
        output = torch.spmm(adj, output)
        output = self.act(output)
        output = self.MLP(output)
        output = self.scale(output)
        output = F.normalize(output)
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

    def embed(self,input,adj):
        output = torch.mm(input, self.weight)
        output = torch.spmm(adj, output)
        output = self.act(output)
        output = self.scale(output)
        output = F.normalize(output)     

        return output

    def get_adj(self,adj):
        return adj+self.adj_lr*self.adj_add

    
import torch.nn.functional as F
from cogdl.layers import GATLayer
import networkx as nx
from typing import Optional, Type, Any
import torch.nn as nn


class BaseModel(nn.Module):
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        pass

    @classmethod
    def build_model_from_args(cls, args):
        """Build a new model instance."""
        raise NotImplementedError("Models must implement the build_model_from_args method")

    def __init__(self):
        super(BaseModel, self).__init__()
        self.model_name = self.__class__.__name__
        self.loss_fn = None
        self.evaluator = None

    def _forward_unimplemented(self, *input: Any) -> None:  # abc warning
        pass

    def forward(self, *args):
        raise NotImplementedError

    def predict(self, data):
        return self.forward(data)

    @property
    def device(self):
        return next(self.parameters()).device

    def set_loss_fn(self, loss_fn):
        self.loss_fn = loss_fn



class DAEGC(BaseModel):
    def __init__(self, in_dim, hid_dim, out_dim, dropout = 0.5, alpha = 1, nheads = 1):
        super(DAEGC, self).__init__()
        self.dropout = dropout
        self.alpha = 0.5
        self.hid_dim = hid_dim
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.attentions = [GraphAttentionLayer(in_dim, hid_dim, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        self.out_att = GraphAttentionLayer(hid_dim * nheads, out_dim, dropout=dropout, alpha=alpha, concat=False)


    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return x

    def _prepare_attentional_mechanism_input(self, Wh):
            # Wh.shape (N, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (N, 1)
        # e.shape (N, N)
        Wh1 = torch.matmul(Wh, self.a[:self.hid_dim, :])
        Wh2 = torch.matmul(Wh, self.a[self.hid_dim:, :])
        # broadcast add
        e = Wh1 + Wh2.T
        return self.leakyrelu(e)
  

class GCN(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, hidden,out_features):
        super(GCN, self).__init__()
        self.MLP = nn.Linear(hidden,out_features)
        self.encoder = nn.Linear(in_features,hidden)

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)
        
    def scale(self, z):
        
        zmax = z.max(dim=1, keepdim=True)[0]
        zmin = z.min(dim=1, keepdim=True)[0]
        z_std = (z - zmin) / (zmax - zmin)
        z_scaled = z_std
    
        return z_scaled
    def forward(self, input, adj):
        # input = input.reshape([1,input.shape[0],-1])
        output = self.encoder(input)
        output = F.relu(output)
        output = self.MLP(output)
        output = self.scale(output)
        output = F.normalize(output)
        return output

    def embed(self,input,adj):
        output = self.encoder(input)
        output = F.relu(output)
        output = self.scale(output)
        output = F.normalize(output)
        # output = self.MLP(output)
        return output

class GAT(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, dropout=0.1, alpha=0.2, nheads=8):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout
        self.attentions = [GraphAttentionLayer(in_dim, hid_dim, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(hid_dim * nheads, out_dim, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))

        return x


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.W = nn.Linear(in_features, out_features)
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        Wh = self.W(h) # h.shape: (N, in_features), Wh.shape: (N, out_features)
        e = self._prepare_attentional_mechanism_input(Wh)

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime
        


    def _prepare_attentional_mechanism_input(self, Wh):
        # Wh.shape (N, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (N, 1)
        # e.shape (N, N)
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        # broadcast add
        e = Wh1 + Wh2.T
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
