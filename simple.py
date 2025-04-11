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
    np_mask = torch.from_numpy(np_mask == 0).to(torch.device("cuda"))
    return np_mask

def create_masks(src, trg, padding_num=1):
    """
    src: (b, seq_len1)
    trg: (b, seq_len2)
    padding_num: 1

    Returns
    -------

    """
    # 1, src mask, 去掉padding部分 (b,1,seq_len1)
    src_mask = (src != padding_num).unsqueeze(-2).to(torch.device("cuda"))

    # 2, trg mask, 去掉padding部分和后面的词。
    if trg is not None:
        # padding 部分  (b,1,seq_len2)
        trg_mask = (trg != padding_num).unsqueeze(-2).to(torch.device("cuda"))
        # 后面的词
        seq_len2 = trg.size(1)
        np_mask = lower_mask(seq_len2).to(torch.device("cuda"))  # (1, seq_len2, seq_len2)
        trg_mask = trg_mask & np_mask
    else:
        trg_mask = None

    return src_mask, trg_mask


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embed(x)

class PositionalEmbedding(nn.Module):


class Encoder(nn.Module):
    """包含了token-embedding, position-embedding, encoder-layers"""
    def __init__(self, vocab_size, d_model, n_layers, heads, dropout):
        super().__init__()
        self.token_embed = TokenEmbedding(vocab_size, d_model)
        self.position_embed = PositionalEmbedding(d_model)

class Decoder(nn.Module):
    pass

class Transformer(nn.Module):
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
    trg_input = trg[:, :-1]  # 翻译的词，去掉最后一个词，因为最后一个词是<eos>，我们不需要预测<eos>。
    # 2, 掩码
    src_mask, trg_mask = create_masks(src, trg_input, padding_num=1)
    # 3, 初始化模型
    model = Transformer(src_vocab_size, trg_vocab_size, d_model, n_layers, heads, dropout)
