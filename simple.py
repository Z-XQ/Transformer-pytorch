import copy
import math

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

def lower_triangular_mask(seq_len):
    """
    seq_len: int. seq_len

    Returns: tensor. (1, seq_len, seq_len)

    """
    upper_mask = np.triu(np.ones(shape=(1, seq_len, seq_len)), k=1).astype('uint8')
    lower_mask = torch.from_numpy(upper_mask == 0)
    return lower_mask

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
        np_mask = lower_triangular_mask(seq_len2)  # (1, seq_len2, seq_len2)
        trg_mask = trg_mask & np_mask  # (b,seq_len2,seq_len2)
    else:
        trg_mask = None

    return src_mask, trg_mask


# 对特征维度进行，alpha * (x-m) / std + bias
class FeatureNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(size=(d_model,)))  # 每个维度都有各自的权重
        self.bias = nn.Parameter(torch.zeros(size=(d_model,)))
        self.eps = eps

    def forward(self, x):
        """x: (b,seq_len,d_model)"""
        # alpha*(x-m)/std(x) + bias
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm


# 1 token-embedding
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embed(x)

# 2 scale token-embedding -> position-embedding -> add -> dropout
class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_seq_len=200, dropout=0.1):
        super().__init__()
        self.d_model = d_model  # for scale token-embedding
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


"""
1, fn1, fn2, fn3: (b,seq_len,d_model) -> (b,seq_len,d_model)
2, split q,k,v: view+transpose, (b,seq_len,d_model) -> (b,h,seq_len,d_k)
3, multi-attention: (b,h,seq_len,d_k) -> (b,h,seq_len,d_k)
4, concat: transpose+view, (b,h,seq_len,d_k) -> (b,seq_len,h*d_k)
5, fn: (b,seq_len,h*d_k) -> (b,seq_len,h*d_k=d_model)
"""
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.fn1 = nn.Linear(d_model, d_model)
        self.fn2 = nn.Linear(d_model, d_model)
        self.fn3 = nn.Linear(d_model, d_model)

        self.d_k = d_model // heads
        self.heads = heads

        self.dropout = nn.Dropout(dropout)
        self.fn = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, src_mask):
        """
        q: (b, seq_len1, d_model);
        k, v: (b, seq_len2, d_model);
        src_mask: (b,1,seq_len2). 用于处理源序列（src）中的填充（padding）部分。
        return: (b,seq_len1,d_model)"""

        # 1, fn+split: (b, seq_len, d_model) -> # (b, seq_len, heads, d_k)
        b = q.size(0)
        q = self.fn1(q).view(b, -1, self.heads, self.d_k)
        k = self.fn2(k).view(b, -1, self.heads, self.d_k)
        v = self.fn3(v).view(b, -1, self.heads, self.d_k)

        # 2, transpose: (b, seq_len, heads, d_k) -> (b, heads, seq_len, d_k)
        q = q.transpose(1, 2)  # (b, heads, seq_len1, d_k)
        k = k.transpose(1, 2)  # (b, heads, seq_len2, d_k)
        v = v.transpose(1, 2)

        # 3, attention
        scores = self.multi_attention(q, k, v, src_mask)  #  (b,h,seq_len1,d_k)

        # 4, concat: (b, heads, seq_len1, d_k) -> (b, seq_len1, heads * d_k)
        scores = scores.transpose(1, 2).contiguous().view(b, -1, self.d_k * self.heads)

        # 5, output
        return self.fn(scores)  # (b, seq_len1, d_model)

    def multi_attention(self, q, k, v, src_mask):
        """
        q:  (b,h,seq_len1,d_k);
        k,v: (b,h,seq_len2,d_k);
        mask: (b,1,seq_len2)
        return: (b,h,seq_len1,d_k)

        q*k^T -> mask -> softmax -> dropout -> scores * v
        """
        #  (b,h,seq_len1,d_k) * (b,h,d_k,seq_len2)  -> (b,h,seq_len1,seq_len2)
        # 可以将点积结果的量级进行缩放，使其保持在一个合理的范围内。这样，softmax 函数的输入不会过大，输出的概率分布在中间区域，避免了梯度消失或爆炸问题，同时也提高了模型的学习效率和稳定性。
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.d_k)

        # 屏蔽位置的分数设置为很小值，在后续的 softmax 操作中把这些位置的权重置为接近 0 的值，进而屏蔽掉这些位置。
        if src_mask is not None:
            mask = src_mask.unsqueeze(1)  # (b,1,1,seq_len)
            scores = scores.masked_fill(mask == 0, -1e9)
            # scores_where = torch.where(mask == 0, -1e9, scores)
        scores = torch.softmax(scores, dim=-1)

        scores = self.dropout(scores)

        # 2, (b,h,seq_len1,seq_len2) * (b,h,seq_len2,d_k) -> (b,h,seq_len1,d_k)
        v = torch.matmul(scores, v)

        return v

# 两层 全连接层：ln1 -> ReLU -> dropout -> ln2
class FeedForward(nn.Module):
    """
    linear -> relue -> dropout: 第一层扩展维度引入非线性
    linear: 第二层恢复原始维度保持网络深度
    """
    def __init__(self, d_model, hidden_dim=2048, dropout=0.1):
        super().__init__()
        self.fn1 = nn.Linear(d_model, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.fn2 = nn.Linear(hidden_dim, d_model)

    def forward(self, x):
        x = self.dropout(self.relu(self.fn1(x)))
        x = self.fn2(x)
        return x

"""
-> norm1->multi-head attention->dropout1 -> res
-> norm2->feed-forward->        dropout2
"""
class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm1 = FeatureNorm(d_model)
        self.norm2 = FeatureNorm(d_model)
        self.attention = MultiHeadAttention(d_model, heads, dropout)
        self.fn = FeedForward(d_model, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, src_mask):
        """
        x: (b,seq_len,d_model)
        src_mask: (b,1,seq_len). 用于处理源序列（src）中的填充（padding）部分。
        Returns
        -------

        """
        tmp = self.norm1(x)
        tmp = self.dropout1(self.attention(tmp, tmp, tmp, src_mask))  # (b,seq_len,d_model)
        x = x + tmp

        tmp = self.norm2(x)
        tmp = self.dropout2(self.fn(tmp))
        x = x + tmp
        return x

def get_clones(module, n_layers):
    return nn.ModuleList([copy.deepcopy(module) for i in range(n_layers)])

"""token-embed -> position-embed -> encoderLayers -> norm"""
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
        self.position_embed = PositionalEmbedding(d_model, 200, dropout)
        self.layers = get_clones(EncoderLayer(d_model, heads, dropout), n_layers)
        self.norm = FeatureNorm(d_model)

    def forward(self, x, src_mask):
        """
        x: (b, seq_len)
        src_mask: (b, 1, seq_len). 用于处理源序列（src）中的填充（padding）部分。
        """
        x = self.token_embed(x)  # (b,seq_len) -> (b,seq_len,d_model)
        x = self.position_embed(x)  # (b,seq_len,d_model)
        for layer in self.layers:
            x = layer(x, src_mask)  # (b,seq_len,d_model)
        x = self.norm(x)
        return x  # (b,seq_len,d_model)

"""
-> norm1 -> multi-head attention -> dropout1 -> res
-> norm2 -> multi-head attention -> dropout2 -> res
-> norm2 -> feedForward ->          dropout3 -> res
"""
class DecoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm1 = FeatureNorm(d_model)
        self.norm2 = FeatureNorm(d_model)
        self.norm3 = FeatureNorm(d_model)

        self.multi_attention = MultiHeadAttention(d_model, heads, dropout)
        self.cross_multi_attention = MultiHeadAttention(d_model, heads, dropout)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.fn = FeedForward(d_model, dropout=dropout)

    def forward(self, trg_embed, e_outputs, src_mask, trg_mask):
        """
        x: trg-embed, (b,seq_len2,d_model)
        e_outputs: (b,seq_len1,d_model)
        src_mask: (b,1,seq_len1)
        trg_mask: (b,1,seq_len2)

        Returns
        -------

        """
        # 1, trg_embed(b,seq_len2,d_model) trg_mask(b,1,seq_len2) -> (b,seq_len2,d_model)
        tmp = self.norm1(trg_embed)
        tmp = self.dropout1(self.multi_attention(tmp, tmp, tmp, trg_mask))
        trg_embed = trg_embed + tmp

        # 2, trg_embed(b,seq_len2,d_model) src_mask(b,1,seq_len1) -> (b,seq_len2,d_model)
        tmp = self.norm2(trg_embed)
        tmp = self.dropout2(self.cross_multi_attention(tmp, e_outputs, e_outputs, src_mask))
        trg_embed = trg_embed + tmp

        tmp = self.norm3(trg_embed)
        tmp = self.dropout3(self.fn(tmp))
        trg_embed = trg_embed + tmp
        return trg_embed

"""token-embed -> position-embed -> decoderLayers -> norm"""
class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, heads, dropout):
        super().__init__()
        self.embed = TokenEmbedding(vocab_size, d_model)
        self.position_embed = PositionalEmbedding(d_model, 200, dropout)
        self.layers = get_clones(DecoderLayer(d_model, heads, dropout), n_layers)
        self.norm = FeatureNorm(d_model)

    def forward(self, trg_input, e_outputs, src_mask, trg_mask):
        """
        trg_input: (b, seq_len2). trg_input代表前面词序列，据此预测后面词
        e_outputs: (b,seq_len1,d_model). 编码特征
        src_mask: (b,1,seq_len1). 屏蔽padding位置
        trg_mask: (b,1,seq_len2,seq_len2). 遮住前面的词和填充（padding）部分。

        Returns
        -------

        """
        # 1 (b,seq_len2) -> (b,seq_len2,d_model)
        token_embed = self.embed(trg_input)

        # 2 (b,seq_len2,d_model)
        embed = self.position_embed(token_embed)

        # 3 (b,seq_len2,d_model)
        for layer in self.layers:
            embed = layer(embed, e_outputs, src_mask, trg_mask)
        embed = self.norm(embed)
        return embed

class Transformer(nn.Module):
    """
    encoder: (b,seq_len1) -> (b,seq_len1,d_model)
    decoder: (b,seq_len2), (b,seq_len1,d_model) -> (b,seq_len2,d_model)
    linear: (b,seq_len2,d_model) -> (b,seq_len2,trg_vocab_size)
    """
    def __init__(self, src_vocab_size, trg_vocab_size, d_model, n_layers, heads, dropout):
        super().__init__()

        # check params
        assert d_model % heads == 0
        assert dropout < 1

        self.encoder = Encoder(src_vocab_size, d_model, n_layers, heads, dropout)
        self.decoder = Decoder(trg_vocab_size, d_model, n_layers, heads, dropout)
        self.fn = nn.Linear(d_model, trg_vocab_size)

        self._init_weights_()

    def forward(self, src_input, trg_input, src_mask, trg_mask):
        # 1 (b,seq_len1) -> (b,seq_len1,d_model)
        encoder_output = self.encoder(src_input, src_mask)

        # 2 (b,seq_len2), (b,seq_len1,d_model) -> (b,seq_len2,d_model)
        decoder_output = self.decoder(trg_input, encoder_output, src_mask, trg_mask)
        return self.fn(decoder_output)  # (b,seq_len2,d_model) -> (b,seq_len2,trg_vocab_size)

    def _init_weights_(self):
        for p in self.parameters():
            if p.dim() > 1:  # 矩阵维度大于1，则初始化权重，用于跳过偏置向量等一维参数，只对二维及以上的权重矩阵进行Xavier初始化。
                nn.init.xavier_uniform_(p)

if __name__ == '__main__':
    src_vocab_size = 13724
    trg_vocab_size = 23469
    d_model = 512
    heads = 8
    dropout = 0.1
    n_layers = 6
    # 1, get input data
    src_input = torch.randint(0, src_vocab_size, (65, 17))  # 原文
    trg = torch.randint(0, trg_vocab_size, (65, 23))  # 翻译
    trg_input = trg[:, :-1]  # (65,22) 翻译的词，去掉最后一个词，作为模型的输入
    label = trg[:, 1:].contiguous()  # (65,22) 翻译的词，去掉第一个词，作为模型的标签。这样trg_input和label一一对应。

    # 2, 掩码
    src_mask, trg_mask = create_masks(src_input, trg_input, padding_num=1)  # (b,1,seq_len1), (b,seq_len2,seq_len2)

    # 3, 初始化模型
    model = Transformer(src_vocab_size, trg_vocab_size, d_model, n_layers, heads, dropout)
    model.to(torch.device("cuda"))
    model.train()

    src_input = src_input.to(torch.device("cuda"))
    trg = trg.to(torch.device("cuda"))
    trg_input = trg_input.to(torch.device("cuda"))
    label = label.to(torch.device("cuda"))
    src_mask = src_mask.to(torch.device("cuda"))
    trg_mask = trg_mask.to(torch.device("cuda"))

    pred = model(src_input, trg_input, src_mask, trg_mask)  # (b,seq_len2-1,trg_vocab_size)
    pred = pred.view(-1, pred.size(-1))  # (b*(seq_len2-1),trg_vocab_size)

    label = label.view(-1)  # (b*(seq_len2-1),)
    print("input: ", pred.shape)
    print("label: ", label.shape)

    loss = F.cross_entropy(pred, label, ignore_index=1)

    # # 测试TokenEmbedding和PositionalEmbedding功能
    # te = TokenEmbedding(src_vocab_size, d_model)
    # pe = PositionalEmbedding(d_model)
    # te.cuda()
    # pe.cuda()
    # test_token_embed = te(src)  # (b,seq_len) -> (b,seq_len,d_model)
    # test_position_embed = pe(test_token_embed)  # (b,seq_len,d_model)-> (b,seq_len,d_model)
    # print(test_position_embed.shape)

    # # 测试encoder
    # encoder = Encoder(src_vocab_size, d_model, n_layers, heads, dropout)
    # encoder.cuda()
    # print("input: ", src.shape)
    # encoder_output = encoder(src, src_mask)
    # print("output: ", encoder_output.shape)
    #
    # # 测试decoder
    # decoder = Decoder(trg_vocab_size, d_model, n_layers, heads, dropout)
    # decoder.cuda()
    # print("input: ", trg_input.shape)
    # decoder_output = decoder(trg_input, encoder_output, src_mask, trg_mask)
    # print("output: ", decoder_output.shape)