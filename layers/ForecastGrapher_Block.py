import math

import torch
import torch.nn.functional as F
from torch import nn, Tensor
import numpy as np
from torch.nn import Parameter

class GNNBlock(nn.Module):
    def __init__(self, c_out, d_model, z, gcn_depth, dropout, propalpha, seq_len, node_dim):
        super(GNNBlock, self).__init__()

        self.nodevec1 = nn.Parameter(torch.randn(c_out, node_dim), requires_grad=True)
        self.nodevec2 = nn.Parameter(torch.randn(node_dim, c_out), requires_grad=True)
        self.gnn = GFC(z, z, gcn_depth, dropout)
        # self.gnn = mixprop(z, z, gcn_depth, dropout, propalpha)
        # self.gnn = GAT(d_model, d_model, d_model, dropout, 0.2, 1, z)
        # self.gnn = GCN(d_model, d_model, d_model, dropout, z)
        self.gelu = nn.GELU()

        self.norm = nn.LayerNorm(d_model)
        self._initialize_weights()
    #
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    #x in(B, conv, N, d_model)
    def forward(self, x, static_adj):
        if static_adj is None:
            adj = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
        else:
            adj = static_adj

        out = self.gelu(self.gnn(x, adj))
        return self.norm(x + out)


class GFC(nn.Module):
    def __init__(self,c_in, c_out, gcn_depth, dropout , seg=4):
        super(GFC, self).__init__()
        self.nconv = nconv()
        self.mlp = linear(c_in , c_out)
        self.dropout = dropout
        self.alpha = 0.5
        self.seg = seg
        self.seg_dim = c_in // self.seg
        self.pad = c_in % self.seg
        self.agg = nn.ModuleList()
        self.agg.append(SeparableConv2d(c_in//seg, c_in//seg, kernel_size=[1,3], stride=1, padding=[0,1]))
        self.agg.append(SeparableConv2d(c_in//seg, c_in//seg, kernel_size=[1,5], stride=1, padding=[0,2]))
        self.agg.append(SeparableConv2d(c_in//seg, c_in//seg, kernel_size=[1,7], stride=1, padding=[0,3]))


    #(B, c, N, d_model)
    def forward(self, x, adj):
        #adj
        adj = adj + torch.eye(adj.size(0)).to(x.device)
        d = adj.sum(1)
        a = adj / d.view(-1, 1)

        #split
        if self.pad == 0:
            x = x.split([self.seg_dim] * self.seg, dim=1)
            out = [x[0]]
            # (B, c, N ,d_model)
            for i in range(1, self.seg):
                h = self.agg[i-1](x[i])
                h = self.alpha * (h + x[i]) + (1 - self.alpha) * self.nconv(h, a)
                out.append(h)
        else:
            y = x[:, :self.seg_dim + self.pad, :, :]
            out = [y]
            x = x[:, self.seg_dim + self.pad:, :, :]
            x = x.split([self.seg_dim] * (self.seg-1),dim=1)
            # (B, c, N ,d_model)
            for i in range(0, self.seg-1):
                h = self.agg[i](x[i])
                h = self.alpha * (h + x[i]) + (1 - self.alpha) * self.nconv(h, a)
                out.append(h)

        out = torch.cat(out,dim=1)
        out = self.mlp(out)
        return  out




class mixprop(nn.Module):
    def __init__(self,c_in, c_out, gdep, dropout, alpha, seg=4):
        super(mixprop, self).__init__()
        self.nconv = nconv()
        self.mlp = linear((gdep+1)*c_in,c_out)
        self.gdep = gdep
        self.dropout = dropout
        self.alpha = alpha

        self.seg = seg
        self.seg_dim = c_in // self.seg
        self.pad = c_in % self.seg
        self.agg = nn.ModuleList()
        self.agg.append(SeparableConv2d(c_in // seg, c_in // seg, kernel_size=[1, 3], stride=1, padding=[0, 1]))
        self.agg.append(SeparableConv2d(c_in // seg, c_in // seg, kernel_size=[1, 5], stride=1, padding=[0, 2]))
        self.agg.append(SeparableConv2d(c_in // seg, c_in // seg, kernel_size=[1, 7], stride=1, padding=[0, 3]))

    def forward(self, x, adj):
        adj = adj + torch.eye(adj.size(0)).to(x.device)
        d = adj.sum(1)
        a = adj / d.view(-1, 1)

        h = x
        out = [h]
        for j in range(self.gdep):
            x = h.split([self.seg_dim] * self.seg, dim=1)
            out_tmp = [x[0]]
            for i in range(1, self.seg):
                # (B, c//4, N ,d_model)
                h2 = self.agg[i - 1](x[i])
                h2 = self.alpha * (h2 + x[i]) + (1 - self.alpha) * self.nconv(h2, a)
                out_tmp.append(h2)
            out_tmp = torch.cat(out_tmp, dim=1)
            out.append(out_tmp)

        ho = torch.cat(out, dim=1)
        ho = self.mlp(ho)
        return ho

        # h = x
        # out = [h]
        # for i in range(self.gdep):
        #     h = self.alpha*x + (1-self.alpha)*self.nconv(h,a)
        #     out.append(h)
        # ho = torch.cat(out,dim=1)
        # ho = self.mlp(ho)
        # return ho



class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads, c_in, seg=4):
        super(GAT, self).__init__()
        self.dropout = dropout
                                           # in_features, out_features, dropout, alpha, concat=True
        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True)
                           for _ in range(nheads)]
        # for i, attention in enumerate(self.attentions):
        #     self.add_module('attention_{}'.format(i), attention)
        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

        self.seg = seg
        self.seg_dim = c_in // self.seg
        self.pad = c_in % self.seg
        self.agg = nn.ModuleList()
        self.agg.append(SeparableConv2d(c_in // seg, c_in // seg, kernel_size=[1, 3], stride=1, padding=[0, 1]))
        self.agg.append(SeparableConv2d(c_in // seg, c_in // seg, kernel_size=[1, 5], stride=1, padding=[0, 2]))
        self.agg.append(SeparableConv2d(c_in // seg, c_in // seg, kernel_size=[1, 7], stride=1, padding=[0, 3]))

    #(B, c, N, d_model)
    def forward(self, x, adj):

        x = x.split([self.seg_dim] * self.seg, dim=1)
        out = [x[0]]
        # (B, c, N ,d_model)
        for i in range(1, self.seg):
            h = self.agg[i-1](x[i])
            h = F.dropout(h, self.dropout, training=self.training)
            h = torch.cat([att(h, adj) for att in self.attentions], dim=-1)
            h = F.dropout(h, self.dropout, training=self.training)
            h = F.elu(self.out_att(h, adj))
            h = h + x[i]
            out.append(h)
        out = torch.cat(out,dim=1)
        return F.log_softmax(out, dim=1)

        # x = F.dropout(x, self.dropout, training=self.training)
        # x = torch.cat([att(x, adj) for att in self.attentions], dim=-1)
        # x = F.dropout(x, self.dropout, training=self.training)
        # x = F.elu(self.out_att(x, adj))
        # return F.log_softmax(x, dim=1)


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
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.W = nn.Parameter(torch.empty(size=(in_features, out_features))).to(device)
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1))).to(device)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    #(B, c, N, d)
    def forward(self, h, adj):
        Wh = torch.einsum('bcni,io->bcno', (h, self.W))
        e = self._prepare_attentional_mechanism_input(torch.mean(torch.mean(Wh, dim=0),dim=0))
        zero_vec = -9e15 * torch.ones_like(e)
        #N*N attetion
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.einsum('bcno,jn->bcjo',Wh,attention)
        # h_prime = torch.matmul(attention, Wh)

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



class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, c_in, seg=4):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

        self.seg = seg
        self.seg_dim = c_in // self.seg
        self.pad = c_in % self.seg
        self.agg = nn.ModuleList()
        self.agg.append(SeparableConv2d(c_in // seg, c_in // seg, kernel_size=[1, 3], stride=1, padding=[0, 1]))
        self.agg.append(SeparableConv2d(c_in // seg, c_in // seg, kernel_size=[1, 5], stride=1, padding=[0, 2]))
        self.agg.append(SeparableConv2d(c_in // seg, c_in // seg, kernel_size=[1, 7], stride=1, padding=[0, 3]))

    def forward(self, x, adj):
        x = x.split([self.seg_dim] * self.seg, dim=1)
        out = [x[0]]
        for i in range(1, self.seg):
            h = self.agg[i-1](x[i])
            h = F.relu(self.gc1(h, adj))
            h = F.dropout(h, self.dropout, training=self.training)
            h = self.gc2(h, adj)
            h = h + x[i]
            out.append(h)
        out = torch.cat(out, dim=1)
        return F.log_softmax(out)

        # x = F.relu(self.gc1(x, adj))
        # x = F.dropout(x, self.dropout, training=self.training)
        # x = self.gc2(x, adj)
        # return F.log_softmax(x)


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'



class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()

        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.conv(x)
        x = self.pointwise(x)
        # x = self.conv(x)
        return x


class nconv(nn.Module):
    def __init__(self):
        super(nconv,self).__init__()

    def forward(self,x, A):
        x = torch.einsum('ncwl,vw->ncvl',(x,A))
        return x.contiguous()



class Predict(nn.Module):
    def __init__(self,  individual, c_out, seq_len, pred_len ,dropout):
        super(Predict, self).__init__()
        self.individual = individual
        self.c_out = c_out

        if self.individual:
            self.seq2pred = nn.ModuleList()
            self.dropout = nn.ModuleList()
            for i in range(self.c_out):
                self.seq2pred.append(nn.Linear(seq_len , pred_len))
                self.dropout.append(nn.Dropout(dropout))
        else:
            self.seq2pred = nn.Linear(seq_len , pred_len)
            self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        if self.individual:
            out = []
            for i in range(self.c_out):
                per_out = self.seq2pred[i](x[:,i,:])
                per_out = self.dropout[i](per_out)
                out.append(per_out)
            out = torch.stack(out,dim=1)
        else:
            out = self.seq2pred(x)
            out = self.dropout(out)

        return out


class linear(nn.Module):
    def __init__(self,c_in,c_out,bias=True):
        super(linear,self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0,0), stride=(1,1), bias=bias)

    def forward(self,x):
        return self.mlp(x)
