import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Norm(nn.Module):
    def __init__(self, d_model, eps = 1e-6):
        super().__init__()
    
        self.size = d_model
        
        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        
        self.eps = eps
    
    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
        / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm

def attention(q, k, v, d_k, mask=None, dropout=None):
    """
    q: (b,h,seq_len,d_k)
    k: (b,h,seq_len,d_k)
    v: (b,h,seq_len,d_k)
    d_k: int.
    src_mask: (b,1,seq_len): 主要用于处理源序列（src）中的填充（padding）部分。
    dropout: 0.1

    Returns
    -------

    """
    # (b,h,seq_len,d_k) mul (b,h,d_k,seq_len) -> (b,h,seq_len,seq_len)
    # q*k^T -> attention分数值(b,h,seq_len,seq_len)
    scores = torch.matmul(q, k.transpose(-2, -1)) /  math.sqrt(d_k)  # 分母的作用：避免过大的权重值，过大的权重值会导致梯度消失，从而影响模型的训练效果。

    # 屏蔽padding part，一直设置为0，避免影响计算结果
    if mask is not None:
        mask = mask.unsqueeze(1)  # (b,1,seq_len)->(b,1,1,seq_len)
        scores = scores.masked_fill(mask == 0, -1e9)  # (b,h,seq_len,seq_len) * (b,1,1,seq_len) -> (b,h,seq_len,seq_len)

    # softmax，获得权重。使其权重值范围在0-1之间
    # 最后维度的值是0-1之间，表示每个位置的权重值。比如(b,h,2,seq_len)，索引2位置token对其他token的关注度。
    scores = F.softmax(scores, dim=-1)
    
    if dropout is not None:
        scores = dropout(scores)

    # 对v进行加权，分数值乘以value向量，对其进行加权，再把所有向量求和，得到加权后的新向量。
    # scores每一行该位置的权重序列，v每一列是该位置的token在「特征维度」上的具体值
    # 相乘求和：「按我的权重，把他们的特征「揉」成我的新特征」
    output = torch.matmul(scores, v)  # (b,h,seq_len,seq_len) * (b,h,seq_len,d_k) -> (b,h,seq_len,d_k)
    return output

class MultiHeadAttention(nn.Module):
    def __init__(self, heads=8, d_model=512, dropout = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.d_k = d_model // heads  # 512/8=64
        self.h = heads
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)
    
    def forward(self, q, k, v, src_mask=None):
        """
        q,k,v: 来自同一个输入。(b,seq_len,d_model)
        src_mask: (b,1,seq_len): 主要用于处理源序列（src）中的填充（padding）部分。

        Returns:

        """
        bs = q.size(0)
        
        # 1，获取q,k,v，并分成多个头
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)  # (b,seq_len,d_model) -> (b,seq_len,h,d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)  # (b,seq_len,d_model) -> (b,seq_len,h,d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)  # (b,seq_len,d_model) -> (b,seq_len,h,d_k)

        # 2，(b,seq_len,h,d_k) -> (b,h,seq_len,d_k)
        k = k.transpose(1,2)
        q = q.transpose(1,2)
        v = v.transpose(1,2)

        # 3，多头注意力模块
        scores = attention(q, k, v, self.d_k, src_mask, self.dropout)  # scores：(b,h,seq_len,d_k)

        # 4，concat所有头结果，并接最后一个全连接层
        concat = scores.transpose(1,2).contiguous()\
        .view(bs, -1, self.d_model)  # (b,h,seq_len,d_k) -> (b,seq_len,h*d_k)
        output = self.out(concat)  # (b,seq_len,h*d_k) -> (b,seq_len,d_model)
    
        return output  # (b,seq_len,d_model)

# 两层 全连接层：ln1 -> ReLU -> dropout -> ln2
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout = 0.1):
        super().__init__()
        # We set d_ff as a default to 2048
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)
    
    def forward(self, x):
        # ReLU激活函数之后使用dropout
        x = self.dropout(F.relu(self.linear_1(x)))  # (bs, seq_len, d_model) -> (bs, seq_len, d_ff)
        x = self.linear_2(x)  # (bs, seq_len, d_ff) -> (bs, seq_len, d_model)
        return x
