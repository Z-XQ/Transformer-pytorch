import math

import torch
import numpy as np
from Batch import nopeak_mask
import torch.nn as nn

def lower_mask(seq_len):
    """
    seq_len: int. seq_len

    Returns: tensor. (1, seq_len, seq_len)

    """
    np_mask = np.triu(np.ones(shape=(1, seq_len, seq_len)), k=1).astype('uint8')
    np_mask = torch.from_numpy(np_mask == 0)
    return np_mask

def create_masks(src, trg, padding_num=1):
    """
    src: (b, seq_len1)
    trg: (b, seq_len2)
    padding_num: 1

    Returns
    src_mask: (b, 1, seq_len1)
    trg_mask: (b, seq_len2, seq_len2)
    -------

    """
    # 1, src mask, 去掉padding部分 (b,1,seq_len1)
    src_mask = (src != padding_num).unsqueeze(-2)

    # 2, trg mask, 去掉padding部分和后面的词。
    if trg is not None:
        # padding 部分  (b,1,seq_len2)
        trg_mask = (trg != padding_num).unsqueeze(-2)
        # 后面的词
        seq_len2 = trg.size(1)
        np_mask = lower_mask(seq_len2)  # (1, seq_len2, seq_len2)
        trg_mask = trg_mask & np_mask  # (b,seq_len2,seq_len2)
    else:
        trg_mask = None

    return src_mask, trg_mask


# 1 token-embedding
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embed(x)

# 2 position-embedding
class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_seq_len=200, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)

        # (max_seq_len, d_model)
        self.pe = torch.zeros(size=(max_seq_len, d_model))
        for pos in range(max_seq_len):  # 每个位置都对应一个位置编码向量
            for i in range(0, d_model, 2):  # 向量的不同维度值
                self.pe[pos, i] = math.sin(pos / (10000**(2*i/d_model)))
                self.pe[pos, i+1] = math.cos(pos / (10000**(2*i/d_model)))
        # (1, max_seq_len, d_model)
        self.pe = self.pe.unsqueeze(0)

    def forward(self, x):
        """x: token-embedding, (b, seq_len, d_model)"""
        # scale token-embedding: 此操作的目的是让词嵌入与位置编码在量级上相匹配，从而保证二者在相加时权重能够均衡。
        x = x * math.sqrt(self.d_model)

        seq_len = x.size(1)
        cur_pe = self.pe[:, :seq_len]  # (1, seq_len, d_model)
        if x.is_cuda:
            cur_pe = cur_pe.cuda()

        x = x + cur_pe  #
        return self.dropout(x)

class FeatureNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(size=d_model))
        self.bias = nn.Parameter(torch.zeros(size=d_model))
        self.eps = eps

    def forward(self, x):
        """x: (b,seq_len,d_model)"""
        # alpha*(x-m)/std(x) + bias
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm

class MultiHeadAttention(nn.Module):
    pass

class FeedForward(nn.Module):
    pass

# 3 encoder-layers
class EncoderLayer(nn.Module):
    pass

class Encoder(nn.Module):
    """包含了token-embedding, position-embedding, encoder-layers"""
    def __init__(self, vocab_size, d_model, n_layers, heads, dropout):
        super().__init__()
        self.token_embed = TokenEmbedding(vocab_size, d_model)
        self.position_embed = PositionalEmbedding(d_model)

class Decoder(nn.Module):
    pass

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, trg_vocab_size, d_model, n_layers, heads, dropout):
        pass


if __name__ == '__main__':
    src_vocab_size = 13724
    trg_vocab_size = 23469
    d_model = 512
    heads = 8
    dropout = 0.1
    n_layers = 6
    # 1, get input data
    src = torch.randint(0, src_vocab_size, (65, 17))  # 原文
    trg = torch.randint(0, trg_vocab_size, (65, 23))  # 翻译
    trg_input = trg[:, :-1]  # (65,22) 翻译的词，去掉最后一个词，因为最后一个词是<eos>，我们不需要预测<eos>。
    # 2, 掩码
    src_mask, trg_mask = create_masks(src, trg_input, padding_num=1)  # (b,1,seq_len1), (b,seq_len2,seq_len2)

    # 3, 初始化模型
    model = Transformer(src_vocab_size, trg_vocab_size, d_model, n_layers, heads, dropout)

    src = src.to(torch.device("cuda"))
    trg = trg.to(torch.device("cuda"))
    trg_input = trg_input.to(torch.device("cuda"))
    src_mask = src_mask.to(torch.device("cuda"))
    trg_mask = trg_mask.to(torch.device("cuda"))

    # 测试TokenEmbedding和PositionalEmbedding功能
    te = TokenEmbedding(src_vocab_size, d_model)
    pe = PositionalEmbedding(d_model)
    te.cuda()
    pe.cuda()
    test_token_embed = te(src)  # (b,seq_len) -> (b,seq_len,d_model)
    test_position_embed = pe(test_token_embed)  # (b,seq_len,d_model)-> (b,seq_len,d_model)
    print(test_position_embed.shape)


