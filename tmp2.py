import math

import numpy as np
import torch
import torch.nn as nn

def lower_triangular_mask(seq_len):
    tri_upper_mask = np.triu(np.ones(shape=(1, seq_len, seq_len)), k=1)
    tri_upper_mask = torch.from_numpy(tri_upper_mask == 0)
    return tri_upper_mask

def create_masks(src, trg, padding_num=1):
    src_mask = (src != padding_num).unsqueeze(-2)  # (b,1,seq_len1)
    if trg is not None:
        trg_mask = (trg != padding_num).unsqueeze(-2)  # (b,1,seq_len2)

        seq_len2 = trg.size(1)  #
        np_mask = lower_triangular_mask(seq_len2)  # (1,seq_len2,seq_len2)
        trg_mask = trg_mask & np_mask  # (b,1,seq_len2) & (1,seq_len2,seq_len2) -> (b,seq_len2,seq_len2)
    else:
        trg_mask = None
    return src_mask, trg_mask


# alpha * (x-m) / std + bias
class FeatureNorm(nn.Module):
    def __init__(self, d_model):
        self.alpha = nn.Parameter(torch.ones(size=d_model))


class EncoderLayer(nn.Module):
    pass

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        self.embed = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        """x: (b,seq_len)"""
        return self.embed(x)  # (b,seq_len,d_model)


# PositionalEmbedding: scale token-embedding -> position-embedding -> add -> dropout
class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_seq_len=200, dropout=0.1):
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        self.pe = torch.zeros(size=(max_seq_len, d_model))  # (max_seq_len,d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                self.pe[pos, i] = math.sin(pos / (10000**(2*i/d_model)))
                self.pe[pos, i+1] = math.cos(pos / (10000**(2*i/d_model)))
        self.pe = self.pe.unsqueeze(0)  # (1,max_seq_len,d_model)

    def forward(self, x):
        """x: (b,seq_len,d_model)"""
        x = x * math.sqrt(self.d_model)
        x = x + self.pe[:, :x.size[1]]  # (b,seq_len,d_model) + (1,seq_len,d_model) -> (b,seq_len,d_model)
        return self.dropout(x)

class Encoder(nn.Module):
    pass


class Decoder(nn.Module):
    pass

class Transformer(nn.Module):
    pass

if __name__ == '__main__':
    src_vocab_size = 13724
    trg_vocab_size = 23469
    src = torch.randint(0, src_vocab_size, (65, 17))  # (b,seq_len1)
    trg = torch.randint(0, trg_vocab_size, (65, 23))
    trg_input = trg[:, :-1]  # (b,seq_len2)

    src_mask, trg_mask = create_masks(src, trg_input, 1)
    print(src_mask.shape, trg_mask.shape)

    src = src.to(torch.device('cuda'))
    trg = trg.to(torch.device('cuda'))
    trg_input = trg_input.to(torch.device('cuda'))
    src_mask = src_mask.to(torch.device('cuda'))
    trg_mask = trg_mask.to(torch.device('cuda'))

    d_model = 512
    n_layers = 6
    heads = 8
    dropout = 0.1
    model = Transformer(src_vocab_size, trg_vocab_size, d_model, n_layers, heads, dropout)