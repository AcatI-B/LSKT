import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from sklearnex import patch_sklearn
patch_sklearn()
from sklearn.cluster import KMeans
from collections import deque

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

MIN_SEQ_LEN = 5
class LSKT(nn.Module):
    def __init__(
        self,
        n_questions,
        n_pid=0,
        d_model=256,
        d_fc=512,
        n_heads=8,
        dropout=0.05,
        shortcut=False,
        device = 'cpu',
        batch_size = 16,
        emb_method = '3pl'
    ):
        super().__init__()
        self.d_model = d_model
        self.n_questions = n_questions
        self.q_embed = nn.Embedding(n_questions + 1, d_model)
        self.s_embed = nn.Embedding(2, d_model)

        if n_pid > 0:
            self.q_diff_embed = nn.Embedding(n_questions + 1, d_model)
            self.s_diff_embed = nn.Embedding(2, d_model)
            self.pl1_diff_embed = nn.Embedding(n_pid + 1, 1)
            self.pl2_diff_embed = nn.Embedding(n_pid + 1, d_model)

        self.n_heads = n_heads
        self.block1 = DTransformerLayer(d_model, n_heads, dropout)
        self.num_channels_large = [64, 64 ,128] 
        self.conv_block1 = TemporalConvNet(num_inputs=d_model, outputs=d_model, num_channels=self.num_channels_large, kernel_size=3)
        self.learning_1 = nn.Linear(d_model +d_model, d_model)             
        torch.nn.init.xavier_uniform_(self.learning_1.weight)

        self.center_num = 4    #Number of cluster centers
        self.culster = Cluster(d_model,self.center_num,0)
        self.state_weight = LrState(d_model, n_heads, dropout)
        self.concat_q = nn.Linear(d_model,d_model)
        self.drop_layer = nn.Dropout(p = 0.5)
        self.concat_twodiff = nn.Linear(2*d_model,d_model)

        self.concat1 = nn.Linear(2*d_model,d_model)
        self.concat2 = nn.Linear(2*d_model,d_model)

        self.out = nn.Sequential(
            nn.Linear(d_model * 2, d_fc),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_fc, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

        self.dropout_rate = dropout
        self.device = device
        self.emb_method = emb_method
    
    def forward(self, q_emb, s_emb, q_emb_att, lens, n=1):     # LSKT
        sub= self.conv_block1(s_emb)
        state_weight, culster_lable = self.culster(sub)
        lr_weight = self.state_weight(sub, sub, peek_cur=True, n=n, state_weight=state_weight)
        hs = self.block1(q_emb_att, q_emb, s_emb, lens, peek_cur=True, n=n, state_weight=lr_weight)
        s_emb = self.learning_1(torch.cat((sub, hs), 2))
        return s_emb, hs, sub, culster_lable


    def predict(self, q, s, pid=None, n=2):
        seqlen = q.size(1) - n + 1
        q = q.masked_fill(q < 0, 0)
        lens = (s[:, :seqlen] >= 0).sum(dim=1)
        s = s.masked_fill(s < 0, 0)
        q_emb = self.q_embed(q)
        s_emb = self.s_embed(s) + q_emb

        if pid is not None:
            pid = pid.masked_fill(pid < 0, 0)
            p_diff_pl2 = self.pl2_diff_embed(pid)
            pl2_emb = self.concat_q(p_diff_pl2)
            pl2_emb = self.drop_layer(pl2_emb)

            pl1_emb = self.pl1_diff_embed(pid).repeat(1, 1, self.d_model)
            q_try = self.q_diff_embed(q)
            s_try = self.s_diff_embed(s) + q_try
            if self.emb_method.lower() == '1pl':
                q_emb = self.concat1(torch.cat((q_emb , q_try * pl1_emb),dim=-1))
                s_emb = self.concat2(torch.cat((s_emb , s_try * pl1_emb),dim=-1))
            elif self.emb_method.lower() =='2pl':
                all_emb = self.concat_twodiff(torch.cat((pl1_emb , pl2_emb),dim=-1))
                q_emb = q_emb + all_emb * q_try          
                s_emb = s_emb + all_emb * s_try
            else:
                q_diff_emb = self.q_embed(q[:, 1:])
                s_diff_emb = self.s_embed(rand_answer(s[:, 1:]).to(q_diff_emb.device)) 
                mask = torch.rand_like(q_diff_emb[:, :, 0]) < random.random()
                s_diff_emb[mask.unsqueeze(2).repeat(1, 1, q_diff_emb.size(2))] = 0
                s_diff_emb += q_diff_emb
                s_diff_emb = torch.cat([s_diff_emb, torch.zeros_like(q_diff_emb[:, :1, :])], dim=1)
                all_emb = self.concat_twodiff(torch.cat((pl1_emb , pl2_emb),dim=-1))
                q_emb =  q_emb + all_emb * q_try          
                s_emb = s_diff_emb + s_emb + all_emb * s_try
            
        h, hs, sub, culster_lable = self(
            q_emb[:, :seqlen, :],
            s_emb[:, :seqlen, :],
            q_emb[:, 1:, :],
            lens,
            n,
        )
        y = self.out(torch.cat([q_emb[:, n - 1 :, :], h], dim=-1)).squeeze(-1)
        if pid is not None:
            return y, h, (p_diff_pl2**2).sum() * 1e-5
        else:
            return y, h, 0.0

    def get_loss(self, q, s, pid=None):
        logits, _, reg_loss = self.predict(q, s, pid)
        s=s[:,1:]
        masked_labels = s[s >= 0].float()
        masked_logits = logits[s >= 0]
        return (
            F.binary_cross_entropy_with_logits(
                masked_logits, masked_labels, reduction="mean"
            )
            + reg_loss
        )

class Cluster(nn.Module):
    def __init__(self, d_model, n_clusters=5, random_state=0):
        super().__init__()
        self.d_model = d_model
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.state_pool = deque(maxlen=10)
        self.centers_pool = np.random.randn(n_clusters, d_model)
    
    def forward(self, learning_state):
        bs, seq_len, _ = learning_state.size()
        reshaped_tensor = learning_state.reshape(-1, self.d_model)
        
        if len(self.state_pool) != 0:
            kmeans_pool = KMeans(n_clusters = self.n_clusters, random_state = self.random_state)
            state_now =torch.cat(list(self.state_pool),dim=0)
            _ = torch.tensor(kmeans_pool.fit_predict(state_now.cpu().detach().numpy()),dtype=torch.long)
            self.centers_pool = kmeans_pool.cluster_centers_
        kmeans = KMeans(n_clusters = self.n_clusters, random_state = self.random_state)
        kmeans.fit(self.centers_pool)
        predicted_labels = torch.tensor(kmeans.predict(reshaped_tensor.cpu().detach().numpy())).view(bs, seq_len)
        pre_labels = predicted_labels.unsqueeze(1).repeat(1, seq_len, 1)
        labels_T = pre_labels.permute(0, 2, 1)
        mask = pre_labels == labels_T  
        state_weight = torch.where(mask, torch.tensor(1.), torch.tensor(0)).to(learning_state.device)

        self.state_pool.append(reshaped_tensor)
        return state_weight, predicted_labels

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size
 
    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()
 
class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.init_weights()
        self.batch_norm = nn.BatchNorm1d(n_outputs) 
        
    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)
 
    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.batch_norm(out + res)
 
 
class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, outputs, num_channels, kernel_size=3, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        self.num_channels = num_channels
        self.num_levels = len(num_channels)
        for i in range(self.num_levels):
            dilation_size = 1
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]
 
        self.network = nn.Sequential(*layers)
        networks = nn.ModuleList()
        for layer in layers:
            networks.append(nn.Sequential(layer, nn.ReLU()))  # Adding ReLU after each TemporalBlock
        self.networks = networks
        self.linears = nn.ModuleList([nn.Linear(num_channels[i], outputs) for i in range(self.num_levels)])
        self.fusion_weights = nn.ParameterList([nn.Parameter(torch.Tensor(1)) for _ in range(self.num_levels)])
        for weight in self.fusion_weights:
            nn.init.uniform_(weight, a=0, b=1) 
        self.deconv = nn.Conv1d(self.num_levels * outputs, outputs, kernel_size=1)

    def forward(self, x):
        x_transformed = x.permute(0, 2, 1)
        outputs = []
        for i, net in enumerate(self.networks):
            x_transformed = net(x_transformed)
            out = x_transformed.permute(0, 2, 1)
            out = nn.functional.normalize(out, p=1, dim=-1)
            outputs.append(out)

        result = [linear(output) for linear, output in zip(self.linears, outputs)]
        total_correlation = 0
        for i in range(len(result)):
            for j in range(i+1, len(result)):
                total_correlation += correlation(result[i], result[j])
        result = self.deconv(torch.cat(result, dim = 2).permute(0, 2, 1)).permute(0, 2, 1)
        return result 

def correlation(t1, t2):
    t1_flat = t1.view(t1.size(0), -1)
    t2_flat = t2.view(t2.size(0), -1)
    correlation = torch.nn.functional.cosine_similarity(t1_flat, t2_flat, dim=1)
    return correlation.abs().mean() 

class LrState(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        self.d_k = d_model // n_heads
        self.h = n_heads
        self.gammas = nn.Parameter(torch.zeros(n_heads, 1, 1))
        torch.nn.init.xavier_uniform_(self.gammas)

    def device(self):
        return next(self.parameters()).device

    def forward(self, query, key, peek_cur=False, n=1, state_weight = None):    
        bs, seqlen, d_k = query.size()
        mask = torch.ones(seqlen, seqlen).tril(0 if peek_cur else -1)
        mask = (mask.bool())[None, None, :, :].to(self.device())
        mask = mask.expand(query.size(0), -1, -1, -1).contiguous()
        q = query.view(bs, -1, self.h, self.d_k).transpose(1, 2)
        k = key.view(bs, -1, self.h, self.d_k).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
        gamma = self.gammas
        if gamma is not None:
            gamma = -1.0 * F.softplus(gamma).unsqueeze(0)
            total_effect = torch.clamp((gamma).exp(), min=1e-5, max=1e5)
            scores *= total_effect
            if state_weight is not None: 
                state_weight = state_weight.unsqueeze(1)
                new_mask = state_weight.bool() & mask
                scores.masked_fill_(new_mask == 0, 0)
        return scores

def rand_answer(tensor):
    random_binary_tensor = torch.randint(0, 2, size=tensor.size())
    return random_binary_tensor

class DTransformerLayer(nn.Module):
    def __init__(self, d_model, n_heads, dropout, kq_same=True):
        super().__init__()
        self.masked_attn_head = MultiHeadAttention(d_model, n_heads, kq_same)
        self.dropout_rate = dropout
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def device(self):
        return next(self.parameters()).device

    def forward(self, query, key, values, lens, peek_cur=False, n=1, state_weight = None):    
        seqlen = query.size(1)
        mask = torch.ones(seqlen, seqlen).tril(0 if peek_cur else -1)        
        mask = (mask.bool())[None, None, :, :].to(self.device())
        # mask manipulation
        if self.training:
            mask = mask.expand(query.size(0), -1, -1, -1).contiguous()
            for b in range(query.size(0)):
                if lens[b] < MIN_SEQ_LEN:
                    continue
                idx = random.sample(
                    range(lens[b] - 1), max(1, int(lens[b] * self.dropout_rate))
                )
                for i in idx:
                    mask[b, :, i + 1 :, i] = 0
        query_ = self.masked_attn_head(query, key, values, mask, state_weight)
        query = query + self.dropout(query_)
        return self.layer_norm(query)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, kq_same=True, bias=True):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_model // n_heads
        self.h = n_heads
        self.q_linear = nn.Linear(d_model, d_model, bias=bias)
        if kq_same:
            self.k_linear = self.q_linear
        else:
            self.k_linear = nn.Linear(d_model, d_model, bias=bias)
        self.v_linear = nn.Linear(d_model, d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

    def forward(self, q, k, v, mask, state_weight = None):
        bs = q.size(0)

        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)
        v_ = attention(
            q,
            k,
            v,
            mask,
            state_weight,
        )
        concat = v_.transpose(1, 2).contiguous().view(bs, -1, self.d_model)
        output = self.out_proj(concat)
        return output


def attention(q, k, v, mask, state_weight=None):
    d_k = k.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)    
    scores.masked_fill_(mask == 0, -1e32)
    if state_weight is not None:
        scores = scores + state_weight
        scores.masked_fill_(mask == 0, -1e32)
        scores = F.softmax(scores, dim=-1)
    else:
        scores = F.softmax(scores, dim=-1)
    scores = scores.masked_fill(mask == 0, 0)  # set to hard zero to avoid leakage
    output = torch.matmul(scores, v)
    return output
