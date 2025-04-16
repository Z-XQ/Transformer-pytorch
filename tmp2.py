import copy
import math

import numpy as np
import torch

from simple import MultiHeadAttention


def lower_triangular_mask(seq_len):
    upper_mask = np.triu(np.ones(shape=(1, seq_len, seq_len)), k=1)
    lower_mask = torch.from_numpy(upper_mask == 0)
    return lower_mask
def create_masks(src, trg, padding_num=1):
    """src: (b,seq_len1); trg: (b,seq_len2)"""
    src_mask = (src != padding_num).unsqueeze(-2)  # (b,1,seq_len1)
    if trg is not None:
        trg_mask = (trg != padding_num).unsqueeze(-2)  # (b,1,seq_len2)

        seq_len2 = trg.size(1)
        lower_mask = lower_triangular_mask(seq_len2)
        # (b,1,seq_len2) & (1,seq_len2,seq_len2)
        trg_mask = trg_mask & lower_mask  # (b,seq_len2,seq_len2)

    else:
        trg_mask = None

    return src_mask, trg_mask

import torch.nn as nn


class FeatureNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        self.alpha = nn.Parameter(torch.ones(size=(d_model,)))  # (512,)
        self.bias = nn.Parameter(torch.zeros(size=(d_model, )))  # (512,)
        self.eps = eps

    def forward(self, x):
        """x: (b,seq_len,d_model)"""
        # (d_model,) * ((b,seq_len,d_model) - (b,seq_len,1)) / (b,seq_len,1)
        norm_v = self.alpha * (x - x.mean(dim=-1, keepdim=True)) / (self.eps + x.std(dim=-1, keepdim=True)) + self.bias
        return norm_v

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        self.embed = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embed(x)

# scale token-embedding -> position-embedding -> add -> dropout
class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_seq_len=200, dropout=0.1):
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        self.pe = torch.zeros(size=(max_seq_len, d_model))
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                self.pe[pos, i] = math.sin(pos / (10000 **(2*i/d_model)))
                self.pe[pos, i+1] = math.cos(pos / (10000 ** (2*i/d_model)))
        self.pe = self.pe.unsqueeze(0)  # (1,max_seq_len,d_model)

    def forward(self, x):
        """(b,seq_len,d_model)"""
        x = x * math.sqrt(self.d_model)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

def get_clones(module, n_layers):
    return nn.ModuleList([copy.deepcopy(module) for i in range(n_layers)])


# fn1,fn2,fn3: q,k,v
# split:
# transpose:

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        pass

# norm1 -> multi-attention -> dropout -> add -> norm2 -> fn -> dropout2 -> add
class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        self.norm1 = FeatureNorm(d_model)
        self.multi_attention = MultiHeadAttention(d_model, heads, dropout)
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = FeatureNorm(d_model)
        self.fn = nn.Linear(d_model, d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, src_mask):
        x2 = self.norm1(x)
        x2 = self.dropout1(self.multi_attention(x2, x2, x2, src_mask))
        x = x + x2

        x2 = self.norm2(x)
        x2 = self.dropout2(self.fn(x2))
        x = x + x2
        return x

# token-embedding -> position-embedding -> encoderLayer -> norm
class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, heads, dropout):
        self.n_layers = n_layers
        self.token_embed = TokenEmbedding(vocab_size, d_model)
        self.position_embed = PositionalEmbedding(d_model, 200, dropout)
        self.layers = get_clones(EncoderLayer(d_model, heads, dropout), n_layers)
        self.norm = FeatureNorm(d_model)

    def forward(self, src, src_mask):
        pass

class Decoder(nn.Module):
    pass

# encoder -> decoder -> fn
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, trg_vocab_size, d_model, n_layers, heads, dropout):
        super().__init__()
        pass


if __name__ == '__main__':
    src_vocab_size = 13724
    trg_vocab_size = 23469
    src = torch.randint(0, src_vocab_size, (65, 17))
    trg = torch.randint(0, trg_vocab_size, (65, 23))
    trg_input = trg[:, :-1]

    # src_mask: (b,1,seq_len1); trg_mask: (b,seq_len2,seq_len2)
    src_mask, trg_mask = create_masks(src, trg_input, 1)
    print(src_mask.shape, trg_mask.shape)

    d_model = 512
    n_layers = 6
    heads = 8
    dropout = 0.1
    model = Transformer(src_vocab_size, trg_vocab_size, d_model, n_layers, heads, dropout)