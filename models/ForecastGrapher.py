import torch
import torch.nn as nn
import pickle
from layers.Embed import DataEmbedding_GNN
from layers.ForecastGrapher_Block import GNNBlock, Predict

class GNNLayer(nn.Module):
    def __init__(self, configs):
        super(GNNLayer, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.k = configs.k
        self.c_out = configs.c_out

        self.gnn_blocks = nn.ModuleList()
        for i in range(self.k):
            self.gnn_blocks.append(GNNBlock(configs.c_out , configs.d_model , configs.z,
                        configs.gcn_depth , configs.dropout, configs.propalpha ,configs.seq_len,
                           configs.node_dim))

        self.scaler = nn.Conv2d(1 , configs.z, (1, 1))
        self.group_concatenate = nn.Conv2d(configs.z, configs.d_model , (1, configs.d_model))

   #(B, N, d_model)
    def forward(self, x, static_adj):
        out = x.unsqueeze(1)
        # (B, z, N, d_model)
        out = self.scaler(out)
        for i in range (self.k):
            out = self.gnn_blocks[i](out, static_adj)

        out = self.group_concatenate(out).squeeze(-1)
        out = out.transpose(2,1)
        out = out + x

        return out

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.c_out = configs.c_out
        self.use_norm = configs.use_norm

        self.static_adj = None
        if configs.adj_path != None:
            try:
                with open(configs.adj_path, "rb") as f:
                    self.static_adj = torch.tensor(pickle.load(f)).to(self.device)
            except Exception as e:
                print(e)
                return

        self.model = nn.ModuleList([GNNLayer(configs) for _ in range(configs.e_layers)])
        self.enc_embedding = DataEmbedding_GNN(configs.seq_len, configs.d_model, configs.embed, configs.freq,
                                           self.c_out, configs.dropout)
        self.layer = configs.e_layers
        self.layer_norm = nn.LayerNorm(configs.d_model)
        self.predict_linear = nn.Linear(self.seq_len, self.pred_len + self.seq_len)
        self.projection = nn.Linear(configs.d_model, configs.pred_len, bias=True)
        self.seq2pred = Predict(configs.individual ,configs.c_out,
                                configs.seq_len, configs.pred_len, configs.dropout)

        # self.times = 0
        # self.dtw_distance = torch.zeros((self.c_out, self.c_out))

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        # Normalization from Non-stationary Transformer.
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(
                torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out, self.static_adj))

        #out (B, N, d_model)
        dec_out = self.projection(enc_out)
        dec_out = dec_out.transpose(2,1)[:, :, :self.c_out]
        # dec_out = self.seq2pred(dec_out.transpose(1, 2)).transpose(1, 2)

        # De-Normalization from Non-stationary Transformer
        if self.use_norm:
            dec_out = dec_out * \
                      (stdev[:, 0, :].unsqueeze(1).repeat(
                          1, self.pred_len, 1))
            dec_out = dec_out + \
                      (means[:, 0, :].unsqueeze(1).repeat(
                          1, self.pred_len, 1))

        return dec_out[:, -self.pred_len:, :]



