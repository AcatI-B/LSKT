import math
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


MIN_SEQ_LEN = 5


class DTransformer(nn.Module):
    def __init__(
        self,
        n_questions,
        args,
        n_pid=0,
        d_model=128,
        d_fc=256,
        n_heads=8,
        n_know=16,
        n_layers=1,
        dropout=0.05,
        lambda_cl=0.1,
        proj=False,
        hard_neg=True,
        window=1,
        shortcut=False,
    ):
        super().__init__()
        self.n_questions = n_questions
        self.q_embed = nn.Embedding(n_questions + 1, d_model)  
        self.s_embed = nn.Embedding(2, d_model)   
        self.at_embed = nn.Embedding(args.max_at + 10, d_model)  
        self.hint_embed = nn.Embedding(300 + 10, d_model)  
        self.att_embed = nn.Embedding(400 + 10, d_model)  
        self.fc_at_q = nn.Linear(4*d_model , d_model)  
        torch.nn.init.xavier_uniform_(self.fc_at_q.weight)

        if n_pid > 0:
            self.q_diff_embed = nn.Embedding(n_questions + 1, d_model)  
            self.s_diff_embed = nn.Embedding(2, d_model)   
            self.p_diff_embed = nn.Embedding(n_pid + 1, 1)  

        self.n_heads = n_heads
        self.block1 = DTransformerLayer(d_model, n_heads, dropout)
        self.block2 = DTransformerLayer(d_model, n_heads, dropout)
        self.block3 = DTransformerLayer(d_model, n_heads, dropout)
        self.block4 = DTransformerLayer(d_model, n_heads, dropout, kq_same=False)
        
        self.n_seq = 10
        self.block5 = DTransformerLayer(d_model, self.n_seq, dropout, if_sub=True)
        self.block6 = DTransformerLayer(d_model, n_heads, dropout)


        self.n_know = n_know    
        self.know_params = nn.Parameter(torch.empty(n_know, d_model))  
        torch.nn.init.uniform_(self.know_params, -1.0, 1.0)  

        self.out = nn.Sequential(
            nn.Linear(d_model * 2, d_fc),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_fc, d_fc // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_fc // 2, 1),
        )

        if proj:
            self.proj = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU())
        else:
            self.proj = None

        self.dropout_rate = dropout
        self.lambda_cl = lambda_cl
        self.hard_neg = hard_neg
        self.shortcut = shortcut
        self.n_layers = n_layers
        self.window = window
        self.sig = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, q_emb, s_emb, s_emb_extr, lens, at):

        bs, seqlen, d_model = q_emb.size()
        n_know = self.n_know

        if self.shortcut:
            # AKT
            hq, _ = self.block1(q_emb, q_emb, q_emb, lens, peek_cur=True)
            hs, scores = self.block2(s_emb, s_emb, s_emb, lens, peek_cur=True)
            return self.block3(hq, hq, hs, lens, peek_cur=False), scores, None

        if self.n_layers == 1:
            hq = q_emb
            p, q_scores = self.block1(q_emb, q_emb, s_emb, lens, at, peek_cur=True)
        elif self.n_layers == 2:
            hq = q_emb
            hs, _ = self.block1(s_emb, s_emb, s_emb, lens, at, peek_cur=True)
            p, q_scores = self.block2(hq, hq, hs, lens, at, peek_cur=True)
        else:
            hq, _ = self.block1(q_emb, q_emb, q_emb, lens, at, peek_cur=True)
            hs, _ = self.block2(s_emb, s_emb, s_emb, lens, at, peek_cur=True)
            p, q_scores = self.block3(hq, hq, hs, lens, at, peek_cur=True)

        query = (  
            self.know_params[None, :, None, :]   
            .expand(bs, -1, seqlen, -1)
            .contiguous()
            .view(bs * n_know, seqlen, d_model)
        )
        hq = hq.unsqueeze(1).expand(-1, n_know, -1, -1).reshape_as(query)
        p = p.unsqueeze(1).expand(-1, n_know, -1, -1).reshape_as(query)

        z, k_scores = self.block4(
            query, hq, p, torch.repeat_interleave(lens, n_know), at, peek_cur=False  
        )
        z = (
            z.view(bs, n_know, seqlen, d_model)  
            .transpose(1, 2)  
            .contiguous()
            .view(bs, seqlen, -1)   
        )
        k_scores = (
            k_scores.view(bs, n_know, self.n_heads, seqlen, seqlen)  
            .permute(0, 2, 3, 1, 4)  
            .contiguous()
        )
        return z, q_scores, k_scores

    def embedding(self, q, s, at, hint, att, pid=None):
        lens = (s >= 0).sum(dim=1)
        # set prediction mask
        q = q.masked_fill(q < 0, 0)  
        s = s.masked_fill(s < 0, 0)
        at = at.masked_fill(at < 0, 0)
        hint = hint.masked_fill(hint < 0, 0)
        att = att.masked_fill(att < 0, 0)

        at_emb = self.at_embed(at)
        hint_emb = self.hint_embed(hint)
        att_emb = self.att_embed(att)
        q_emb = self.q_embed(q)
        qq_emb = q_emb
        s_emb = self.s_embed(s) + q_emb
        s_emb_extr = s_emb

        p_diff = 0.0

        if pid is not None:
            pid = pid.masked_fill(pid < 0, 0)
            p_diff = self.p_diff_embed(pid)  

            q_diff_emb = self.q_diff_embed(q)
            q_emb += q_diff_emb * p_diff
            qq_emb += q_diff_emb * p_diff

            s_diff_emb = self.s_diff_embed(s) + q_diff_emb
            s_emb += s_diff_emb * p_diff
            s_emb_extr = s_emb

        return q_emb, qq_emb, s_emb, s_emb_extr, lens, p_diff

    def readout(self, z, query):
        bs, seqlen, _ = query.size()
        key = (   
            self.know_params[None, None, :, :]
            .expand(bs, seqlen, -1, -1)
            .view(bs * seqlen, self.n_know, -1)
        )
        value = z.reshape(bs * seqlen, self.n_know, -1)  

        beta = torch.matmul(
            key,   
            query.reshape(bs * seqlen, -1, 1),   
        ).view(bs * seqlen, 1, self.n_know)   
        alpha = torch.softmax(beta, -1)
        return torch.matmul(alpha, value).view(bs, seqlen, -1) 

    def predict(self, q, s, at, hint, att,  pid=None, n=1):
        q_emb, qq_emb, s_emb, s_emb_extr, lens, p_diff = self.embedding(q, s, at, hint, att, pid)   
        
        z, q_scores, k_scores = self(q_emb, s_emb, s_emb_extr, lens, at)

        # predict T+N
        if self.shortcut:
            assert n == 1, "AKT does not support T+N prediction"
            h = z
        else:
            query = q_emb[:, n - 1 :, :]
            h = self.readout(z[:, : query.size(1), :], query)  

        y = self.out(torch.cat([query,   h], dim=-1)).squeeze(-1)  

        if pid is not None:
            return y, z, q_emb, (p_diff**2).mean() * 1e-3, (q_scores, k_scores)
        else:
            return y, z, q_emb, 0.0, (q_scores, k_scores)

    def get_loss(self, q, s, pid=None):
        logits, _, _, reg_loss, _ = self.predict(q, s, pid)
        masked_labels = s[s >= 0].float()
        masked_logits = logits[s >= 0]
        return (
            F.binary_cross_entropy_with_logits(
                masked_logits, masked_labels, reduction="mean"
            )
            + reg_loss
        )

    def get_cl_loss(self, q, s, at=None, hint=None, att=None,  pid=None):
        bs = s.size(0)   #batch_size

        # skip CL for batches that are too short
        lens = (s >= 0).sum(dim=1)  
        minlen = lens.min().item()
        if minlen < MIN_SEQ_LEN:
            return self.get_loss(q, s, pid)  

        # augmentation
        q_ = q.clone()
        s_ = s.clone()
        at_ = at.clone()
        hint_ = hint.clone()
        att_ = att.clone()

        if pid is not None:
            pid_ = pid.clone()
        else:
            pid_ = None

        # manipulate order  #控制顺序
        for b in range(bs):
            idx = random.sample(
                range(lens[b] - 1), max(1, int(lens[b] * self.dropout_rate))
            )  
            for i in idx:
                q_[b, i], q_[b, i + 1] = q_[b, i + 1], q_[b, i]
                s_[b, i], s_[b, i + 1] = s_[b, i + 1], s_[b, i]
                at_[b, i], at_[b, i + 1] = at_[b, i + 1], at_[b, i]
                hint_[b, i], hint_[b, i + 1] = hint_[b, i + 1], hint_[b, i]
                att_[b, i], att_[b, i + 1] = att_[b, i + 1], att_[b, i]
                if pid_ is not None:
                    pid_[b, i], pid_[b, i + 1] = pid_[b, i + 1], pid_[b, i]

        # hard negative
        s_flip = s.clone() if self.hard_neg else s_
        for b in range(bs):
            # manipulate score
            idx = random.sample(
                range(lens[b]), max(1, int(lens[b] * self.dropout_rate))
            )
            for i in idx:
                s_flip[b, i] = 1 - s_flip[b, i]    
        if not self.hard_neg:    
            s_ = s_flip

        # model
        logits, z_1, q_emb, reg_loss, _ = self.predict(q, s, at, hint, att, pid)  
        masked_logits = logits[s >= 0]
        _, z_2, *_ = self.predict(q_, s_, at_, hint_, att_, pid_)  
        if self.hard_neg:
            _, z_3, *_ = self.predict(q, s_flip, pid)

        # CL loss
        input = self.sim(z_1[:, :minlen, :], z_2[:, :minlen, :])
        if self.hard_neg:
            hard_neg = self.sim(z_1[:, :minlen, :], z_3[:, :minlen, :])
            input = torch.cat([input, hard_neg], dim=1)
        target = (
            torch.arange(s.size(0))[:, None]
            .to(self.know_params.device)
            .expand(-1, minlen)
        )
        cl_loss = F.cross_entropy(input, target)

        # prediction loss
        masked_labels = s[s >= 0].float()
        pred_loss = F.binary_cross_entropy_with_logits(
            masked_logits, masked_labels, reduction="mean"
        )
        for i in range(1, self.window):
            label = s[:, i:]
            query = q_emb[:, i:, :]
            h = self.readout(z_1[:, : query.size(1), :], query)
            y = self.out(torch.cat([query, h], dim=-1)).squeeze(-1)

            pred_loss += F.binary_cross_entropy_with_logits(
                y[label >= 0], label[label >= 0].float()
            )
        pred_loss /= self.window

        # TODO: weights
        return pred_loss + cl_loss * self.lambda_cl + reg_loss, pred_loss, cl_loss

    def sim(self, z1, z2):
        bs, seqlen, _ = z1.size()
        z1 = z1.unsqueeze(1).view(bs, 1, seqlen, self.n_know, -1)
        z2 = z2.unsqueeze(0).view(1, bs, seqlen, self.n_know, -1)
        if self.proj is not None:
            z1 = self.proj(z1)
            z2 = self.proj(z2)
        return F.cosine_similarity(z1.mean(-2), z2.mean(-2), dim=-1) / 0.05

class DTransformerLayer(nn.Module):
    def __init__(self, d_model, n_heads, dropout, if_sub=False, kq_same=True):
        super().__init__()
        self.masked_attn_head = MultiHeadAttention(d_model, n_heads, if_sub, kq_same)

        self.dropout_rate = dropout
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

        self.if_sub = if_sub
        self.n_heads = n_heads

    def device(self):
        return next(self.parameters()).device

    def forward(self, query, key, values, lens, at=None, peek_cur=False):
        # construct mask
        if self.if_sub:
            seqlen = query.size(1)//self.n_heads 
        else:
            seqlen = query.size(1)
        bs = query.size(0)
        mask = torch.ones(seqlen, seqlen).tril(0 if peek_cur else -1)  
        mask = mask.bool()[None, None, :, :].to(self.device())  

        # mask manipulation  
        if self.training and not self.if_sub:
            mask = mask.expand(bs, -1, -1, -1).contiguous()  

            for b in range(bs):
                # sample for each batch
                if lens[b] < MIN_SEQ_LEN:
                    # skip for short sequences
                    continue
                idx = random.sample(
                    range(lens[b] - 1), max(1, int(lens[b] * self.dropout_rate))
                )
                for i in idx:
                    mask[b, :, i + 1 :, i] = 0
        
         # apply transformer layer
        query_, scores = self.masked_attn_head(
            query, key, values, mask, maxout=not peek_cur
        )
        query = query + self.dropout(query_)
        return self.layer_norm(query), scores


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, if_sub=False, kq_same=True, bias=True):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_model // n_heads
        self.h = n_heads
        self.if_sub = if_sub  

        self.q_linear = nn.Linear(d_model, d_model, bias=bias)
        if kq_same:   
            self.k_linear = self.q_linear
        else:
            self.k_linear = nn.Linear(d_model, d_model, bias=bias)
        self.v_linear = nn.Linear(d_model, d_model, bias=bias)

        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        self.gammas = nn.Parameter(torch.zeros(n_heads, 1, 1))   
        torch.nn.init.xavier_uniform_(self.gammas)

    def forward(self, q, k, v, mask, at=None, maxout=False):
        bs = q.size(0)
        seq_len = q.size(1)

        if self.if_sub:  
            q = self.q_linear(q).view(bs, self.h, seq_len//self.h, self.d_model)
            k = self.k_linear(k).view(bs, self.h, seq_len//self.h, self.d_model)
            v = self.v_linear(v).view(bs, self.h, seq_len//self.h, self.d_model)

        else:
            # perform linear operation and split into h heads  
            q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
            k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
            v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

            # transpose to get dimensions bs * h * sl * d_k
            k = k.transpose(1, 2)
            q = q.transpose(1, 2)
            v = v.transpose(1, 2)

        # calculate attention using function we will define next  
        v_, scores = attention(  
            q,
            k,
            v,
            mask,
            at,
            self.gammas,
            maxout,
        )

        # concatenate heads and put through final linear layer  
        if self.if_sub: 
            concat = v_.view(bs, -1, self.d_model).contiguous()
        else:
            concat = v_.transpose(1, 2).contiguous().view(bs, -1, self.d_model)
        output = self.out_proj(concat)  
        return output, scores


def attention(q, k, v, mask, at, gamma=None, maxout=False):
    # attention score with scaled dot production  
    d_k = k.size(-1)  
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    bs, head, seqlen, _ = scores.size()

    if at.size(0) == bs and at.size(1) == seqlen:  
        at = at.masked_fill(at < 0, 0)
        distcum_at = torch.cumsum(at, dim=-1).view(bs, 1, seqlen).expand(-1, seqlen, -1).float()
        distcum_at = distcum_at.masked_fill((mask == 0).squeeze(), 0).unsqueeze(1)
        disttotal_at,_ = torch.max(distcum_at, dim=-1, keepdim=True)
        dist_at = (disttotal_at - distcum_at)+ 1.0
        exponent = 0.7  
        dist_at = torch.pow(dist_at.masked_fill((mask == 0), 0), exponent)

    # include temporal effect  
    if gamma is not None:
        x1 = torch.arange(seqlen).float().expand(seqlen, -1).to(gamma.device)  
        x2 = x1.transpose(0, 1).contiguous()
        with torch.no_grad():
            scores_ = scores.masked_fill(mask == 0, -1e32)
            scores_ = F.softmax(scores_, dim=-1)
            distcum_scores = torch.cumsum(scores_, dim=-1)
            disttotal_scores = torch.sum(scores_, dim=-1, keepdim=True)
            position_effect = torch.abs(x1 - x2)[None, None, :, :]  
            if at.size(0) == bs and at.size(1) == seqlen:
                dist_scores = torch.clamp(
                    (disttotal_scores - distcum_scores) *  dist_at, min=0.0
                    
                )
            else :
                dist_scores = torch.clamp(
                    (disttotal_scores - distcum_scores) * position_effect , min=0.0
                )
            dist_scores = dist_scores.sqrt().detach()

        gamma = -1.0 * gamma.abs().unsqueeze(0)  
        total_effect = torch.clamp((dist_scores * gamma).exp(), min=1e-5, max=1e5)
        scores *= total_effect

    # normalize attention score
    scores.masked_fill_(mask == 0, -1e32)
    scores = F.softmax(scores, dim=-1)
    scores = scores.masked_fill(mask == 0, 0)  
    

    # max-out scores (bs, n_heads, seqlen, seqlen)
    if maxout:
        scale = torch.clamp(1.0 / scores.max(dim=-1, keepdim=True)[0], max=5.0)
        scores *= scale

    # calculate output
    output = torch.matmul(scores, v)
    return output, scores
