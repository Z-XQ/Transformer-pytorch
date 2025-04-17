import copy
import math

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


def lower_triangular_mask(seq_len):
    upper_mask = np.triu(np.ones(shape=(1, seq_len, seq_len)), k=1)
    lower_mask = torch.from_numpy(upper_mask == 0)
    return lower_mask  # (1, seq_len, seq_len)

def create_masks(src_input, trg_input, padding_num=1):
    src_mask = (src_input != padding_num).unsqueeze(-2)  # (b,1,seq_len1)
    if trg_input is not None:
        trg_mask = (trg_input != padding_num).unsqueeze(-2)  # (b,1,seq_len2)

        seq_len2 = trg_input.size(1)
        lower_mask = lower_triangular_mask(seq_len2)  # (1,seq_len2,seq_len2)
        trg_mask = trg_mask & lower_mask  # (b,seq_len2,seq_len2)
    else:
        trg_mask = None
    return src_mask, trg_mask


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embed(x)


class PositionEmbedding(nn.Module):
    """
    scale token-embedding:
    position-embed:
    add:
    dropout:
    return: (1,seq_len,d_model)
    """
    def __init__(self, max_seq_len, d_model, dropout=0.1):
        super().__init__()
        self.pe = torch.zeros(size=(max_seq_len, d_model))
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                self.pe[pos, i] = math.sin(pos / (10000 ** (2*i/d_model)))
                self.pe[pos, i+1] = math.cos(pos / (10000 ** (2*i/d_model)))

        # (max_seq_len,d_model) -> (1,max_seq_len,d_model)
        self.pe = self.pe.unsqueeze(0)
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model

    def forward(self, token_embed):
        token_embed = token_embed * math.sqrt(self.d_model)
        cur_position_embed = self.pe[:, :token_embed.size(1), :]
        embed = token_embed + cur_position_embed
        return self.dropout(embed)


def get_clones(module, n_layers):
    return nn.ModuleList([copy.deepcopy(module) for i in range(n_layers)])


class FeatureNorm(nn.Module):
    """
    alpha*(x-m)/std + bias
    """
    def __init__(self, d_model, eps=1e-6):
        self.alpha = nn.Parameter(torch.ones(size=(d_model,)))  # (d_model,)
        self.bias = nn.Parameter(torch.ones(size=(d_model,)))
        self.eps = eps

    def foward(self, x):
        # (d_model) * ((b,seq_len,d_model) - (b,seq_len, 1))
        # (d_model) * ((b,seq_len,d_model) - (b,seq_len, d_model))
        # (1, 1, d_model) * (b,seq_len,d_model)
        x = self.alpha * (x - x.mean(dim=-1, keepdim=True)) / (self.eps + x.std(dim=-1, keepdim=True)) + self.bias
        return x


class FeedForward(nn.Module):
    """
    linear -> relu -> dropout
    linear
    """
    def __init__(self, d_model, hidden_dim=2048, dropout=0.1):
        super().__init__()
        self.fn1 = nn.Linear(d_model, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.fn2 = nn.Linear(hidden_dim, d_model)

    def forward(self, x):
        x = self.dropout(self.relu(self.fn1(x)))
        return self.fn2(x)


class MultiHeadAttention(nn.Module):
    """
    fn1,fn2,fn3: (b,seq_len,d_model) -> (b,seq_len,d_model)
    split: view + transpose, (b,seq_len,d_model) -> (b,h,seq_len,d_k)
    attention: (b,h,seq_len,d_k) -> (b,h,seq_len,d_k)
    concat: (b,h,seq_len,d_k) -> (b,seq_len,h*d_k)
    out: linear, (b,seq_len,h*d_k) -> (b,seq_len,d_model)
    """
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.fn1 = nn.Linear(d_model, d_model)
        self.fn2 = nn.Linear(d_model, d_model)
        self.fn3 = nn.Linear(d_model, d_model)

        self.d_k = d_model // heads
        self.heads = heads

        self.dropout = nn.Dropout(dropout)

        self.out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, src_mask):
        """
        q: (b, seq_len1, d_model);
        k, v: (b, seq_len2, d_model);
        src_mask: (b,1,seq_len2). 用于处理源序列（src）中的填充（padding）部分。
        return: (b,seq_len1,d_model)"""
        # fn
        q = self.fn1(q)
        k = self.fn2(k)
        v = self.fn3(v)

        # split
        q = q.view(q.size(0), -1, self.heads, self.d_k).transpose(1, 2)  # (b,seq_len1,d_model) -> (b,h,seq_len1,d_k)
        k = k.view(k.size(0), -1, self.heads, self.d_k).transpose(1, 2)  # (b,seq_len2,d_model) -> (b,h,seq_len2,d_k)
        v = v.view(v.size(0), -1, self.heads, self.d_k).transpose(1, 2)

        # attention
        scores = self.attention(q, k, v, src_mask)   # (b,h,seq_len1,d_k)

        # concat
        scores = scores.transpose(1, 2).contiguous().view(q.size(0), -1, self.heads*self.d_k)

        # out
        return self.out(scores)  # (b,seq_len1,d_model)

    def attention(self, q, k, v, mask):
        """
        q:  (b,h,seq_len1,d_k);
        k,v: (b,h,seq_len2,d_k);
        mask: (b,1,seq_len2)
        return: (b,h,seq_len1,d_k)

        q*k^T -> mask -> softmax -> dropout -> scores * v
        """
        # (b,h,seq_len1,d_k) * (b,h,d_k,seq_len2)
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.d_k)  # (b,h,seq_len1,seq_len2)

        # mask
        if mask is not None:
            mask = mask.unsqueeze(1)  # (b, 1, 1, seq_len2)
            scores = scores.masked_fill(mask == 0, -1e9)

        scores = F.softmax(scores, dim=-1)

        if self.dropout is not None:
            scores = self.dropout(scores)

        scores = torch.matmul(scores, v)  # (b,h,seq_len1,seq_len2) * (b,h,seq_len2,d_k) -> (b,h,seq_len1,d_k)
        return scores  # (b,h,seq_len1,d_k)

class EncoderLayer(nn.Module):
    """
    -> norm1 -> multi-head attention -> dropout -> res
    -> norm2 -> feedForward ->          dropout -> res
    """
    def __init__(self, d_model, heads ,dropout):
        super().__init__()
        self.norm1 = FeatureNorm(d_model)
        self.norm2 = FeatureNorm(d_model)

        self.multi_attention = MultiHeadAttention(d_model, heads, dropout)
        self.ff = FeedForward(d_model, 2048, dropout)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, src_mask):
        x2 = self.norm1(x)
        x = x + self.dropout1(self.multi_attention(x2, x2, x2, src_mask))

        x2 = self.norm2(x)
        x = x + self.dropout2(self.ff(x2))
        return x  # (b,seq_len1,d_model)

class Encoder(nn.Module):
    """
        token-embedding: (b,seq_len) -> (b,seq_len,d_model)
        position-embedding: (b,seq_len,d_model) -> (b,seq_len,d_model)
        encoderLayers: (b,seq_len,d_model) -> (b,seq_len,d_model)
        norm: featureNorm
    """
    def __init__(self, vocab_size, d_model, n_layers, heads, dropout):
        super().__init__()
        self.token_embed = TokenEmbedding(vocab_size, d_model)
        self.position_embed = PositionEmbedding(200, d_model, dropout)
        self.layers = get_clones(EncoderLayer(d_model, heads, dropout), n_layers)
        self.norm = FeatureNorm(d_model)

    def forward(self, x, src_mask):
        x = self.token_embed(x)
        x = self.position_embed(x)
        for layer in self.layers:
            x = layer(x, src_mask)
        x = self.norm(x)
        return x


class DecoderLayer(nn.Module):
    """
    -> norm1 -> multi-head attention ->       dropout -> res
    -> norm2 -> cross multi-head attention -> dropout -> res
    -> norm3 -> feedForward                -> dropout -> res
    """
    def __init__(self, d_model, heads, dropout):
        super().__init__()
        self.norm1 = FeatureNorm(d_model)
        self.norm2 = FeatureNorm(d_model)
        self.norm3 = FeatureNorm(d_model)

        self.multi_attention = MultiHeadAttention(d_model, heads, dropout)
        self.cross_multi_attention = MultiHeadAttention(d_model, heads, dropout)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.ff = FeedForward(d_model, dropout)

    def forward(self, trg_input, e_outputs, src_mask, trg_mask):
        x2 = self.norm1(trg_input)
        trg_input = trg_input + self.dropout1(self.multi_attention(x2, x2, x2, trg_mask))

        x2 = self.norm2(trg_input)
        trg_input = trg_input + self.dropout2(self.cross_multi_attention(x2, e_outputs, e_outputs, src_mask))

        x2 = self.norm3(trg_input)
        trg_input = trg_input + self.dropout3(self.ff(x2))
        return trg_input

class Decoder(nn.Module):
    """
    token-embedding:
    position-embedding:
    decoderLayer:
    norm:
    """
    def __init__(self, vocab_size, d_model, n_layers, heads, dropout):
        super().__init__()
        self.token_embed = TokenEmbedding(vocab_size, d_model)
        self.position_embed = PositionEmbedding(200, d_model, dropout)
        self.layers = get_clones(DecoderLayer(d_model, heads, dropout))
        self.norm = FeatureNorm(d_model)

    def forward(self, trg_input, e_outputs, src_mask, trg_mask):
        x = self.token_embed(trg_input)
        x = self.position_embed(x)
        for layer in self.layers:
            x = layer(x, e_outputs, src_mask, trg_mask)

        x = self.norm(x)
        return x


class Transformer(nn.Module):
    """
    encoder: (b,seq_len1) -> (b,seq_len1,d_model)
    decoder: (b,seq_len2), (b,seq_len1,d_model) -> (b,seq_len2,d_model)
    linear: (b,seq_len2,d_model) -> (b,seq_len2,trg_vocab_size)
    """
    def __init__(self, src_vocab_size, trg_vocab_size, d_model, n_layers, heads, dropout):
        super().__init__()
        self.encoder = Encoder(src_vocab_size, d_model, n_layers, heads, dropout)
        self.decoder = Decoder(trg_vocab_size, d_model, n_layers, heads, dropout)
        self.fn = nn.Linear(d_model, trg_vocab_size)
        self._init_weights_()

    def forward(self, x):
        encoder_output = self.encoder(x, src_mask)
        decoder_output = self.decoder(trg_input, encoder_output, src_mask, trg_mask)
        output = self.fn(decoder_output)
        return output  # (b,seq_len2,trg_vocab_size)

if __name__ == '__main__':
    src_vocab_size = 13724
    trg_vocab_size = 29324
    src_input = torch.randint(0, src_vocab_size, (65, 17))
    trg = torch.randint(0, trg_vocab_size, (65, 23))
    trg_input = trg[:, :-1]
    label = trg[:, 1:]

    src_mask, trg_mask = create_masks(src_input, trg_input)

    model = Transformer(src_vocab_size, trg_vocab_size, d_model, n_layers, heads, dropout)
