import torch
import torch.nn as nn
import torch.nn.functional as F
import math


"""
类似LayerNorm的功能，与BatchNorm1d的主要区别在于：
1. 归一化维度不同（LayerNorm对特征维度，BatchNorm对批量维度）
2. 统计量计算方式不同（LayerNorm每个样本独立计算，BatchNorm跨样本计算）
3. 适用场景不同（LayerNorm更适合序列模型，BatchNorm适合图像等固定维度数据）
"""
class Norm(nn.Module):
    def __init__(self, d_model, eps = 1e-6):
        super().__init__()
    
        self.size = d_model
        
        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(size=(self.size,)))  # (512,). 可学习的参数。# 每个维度都有各自的权重
        self.bias = nn.Parameter(torch.zeros(size=(self.size,)))
        self.eps = eps
    
    def forward(self, x):  # x: (b,seq_len,d_model)
        # 公式：alpha（x-m)/std(x) + bias
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        # x.mean(dim=-1, keepdim=True): (b,seq_len,1)
        # x.std(dim=-1, keepdim=True): (b,seq_len,1)
        return norm

"""
matmul(q,k^T) -> scores(b,h,seq_len,seq_len) -> mask -> softmax最后维度 -> (b,h,seq_len,seq_len) -> 
matmul(scores,v) -> (b,h,seq_len,d_k) 
"""
def multi_attention(q, k, v, d_k, mask=None, dropout=None):
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
    # 可以将点积结果的量级进行缩放，使其保持在一个合理的范围内。这样，softmax 函数的输入不会过大，输出的概率分布更加平滑，避免了梯度消失或爆炸问题，同时也提高了模型的学习效率和稳定性。
    scores = torch.matmul(q, k.transpose(-2, -1)) /  math.sqrt(d_k)

    # 屏蔽padding part，将score值一直设置为很小值，在后续的 softmax 操作中把这些位置的权重置为接近 0 的值，进而屏蔽掉这些位置。
    if mask is not None:
        mask = mask.unsqueeze(1)  # (b,1,seq_len)->(b,1,1,seq_len)
        scores = scores.masked_fill(mask == 0, -1e9)  # (b,h,seq_len,seq_len) * (b,1,1,seq_len) -> (b,h,seq_len,seq_len)

    # softmax特征维度，获得权重。使其权重值范围在0-1之间
    # 最后维度的值是0-1之间，表示每个位置的权重值。比如(b,h,2,seq_len)，索引2位置token对其他token的关注度。
    scores = F.softmax(scores, dim=-1)
    
    if dropout is not None:
        scores = dropout(scores)

    # 对v进行加权，分数值乘以value向量，对其进行加权，再把所有向量求和，得到加权后的新向量。
    # scores每一行该位置的权重序列，v每一列是该位置的token在「特征维度」上的具体值
    # 相乘求和：「按我的权重，把他们的特征「揉」成我的新特征」
    output = torch.matmul(scores, v)  # (b,h,seq_len,seq_len) * (b,h,seq_len,d_k) -> (b,h,seq_len,d_k)
    return output

"""
fn1, fn2, fn3: (b,seq_len,d_model) -> (b,seq_len,d_model) 
split q,k,v: (b,seq_len,d_model) -> (b,seq_len,h,d_k)
transpose q,k,v:  (b,seq_len,h,d_k) ->  (b,h,seq_len,d_k)
attention: (b,h,seq_len,d_k) -> (b,h,seq_len,d_k)
concat: (b,h,seq_len,d_k) -> (b,seq_len,h*d_k)
fn: (b,seq_len,h*d_k) -> (b,seq_len,h*d_k=d_model)
"""
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
        scores = multi_attention(q, k, v, self.d_k, src_mask, self.dropout)  # scores：(b,h,seq_len,d_k)

        # 4，concat所有头结果，并接最后一个全连接层
        concat = scores.transpose(1,2).contiguous().view(bs, -1, self.d_model)  # (b,h,seq_len,d_k) -> (b,seq_len,h*d_k)
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
