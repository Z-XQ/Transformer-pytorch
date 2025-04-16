import torch
import torch.nn as nn
from Sublayers import FeedForward, MultiHeadAttention, Norm


"""
norm1->multi-head attention->dropout1 -> res
-> norm2->feed-forward->dropout2
"""
class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.attn = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.ff = FeedForward(d_model, dropout=dropout)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        
    def forward(self, x, src_mask):
        """
        x: (b,seq_len,d_model)
        src_mask: (b,1,seq_len): 主要用于处理源序列（src）中的填充（padding）部分。

        Returns
        -------

        """
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn(x2,x2,x2,src_mask))  # (b,seq_len,d_model)->(b,seq_len,d_model)

        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.ff(x2))
        return x
    
"""
-> norm1->multi-head attention->dropout1 -> res
-> norm2->multi-head attention->dropout2 -> res
-> norm3->feed forward->        dropout3 -> res
"""
class DecoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.norm_3 = Norm(d_model)
        
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)
        
        self.attn_1 = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.attn_2 = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.ff = FeedForward(d_model, dropout=dropout)

    def forward(self, x, e_outputs, src_mask, trg_mask):
        """
        x: (b, seq_len2). x代表目标序列（trg_input）
        e_outputs: (b,seq_len1,d_model). 编码特征
        src_mask: (b,1,seq_len1). 屏蔽padding位置
        trg_mask: (b,1,seq_len2,seq_len2). 遮住前面的词和填充（padding）部分。

        Returns
        -------

        """
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn_1(x2, x2, x2, trg_mask))  # (b,seq_len2,d_model)->(b,seq_len2,d_model)

        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.attn_2(x2, e_outputs, e_outputs, src_mask))  # (b,seq_len2,d_model)->(b,seq_len2,d_model)

        x2 = self.norm_3(x)
        x = x + self.dropout_3(self.ff(x2))  # (b,seq_len2,d_model)->(b,seq_len2,d_model)
        return x