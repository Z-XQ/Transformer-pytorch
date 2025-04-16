import torch
import torch.nn as nn 
from Layers import EncoderLayer, DecoderLayer
from Embed import Embedder, PositionalEncoder
from Sublayers import Norm
import copy

def get_clones(module, n_layers):
    return nn.ModuleList([copy.deepcopy(module) for i in range(n_layers)])

"""token-embed -> position-embed -> encoderLayer -> norm"""
class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, heads, dropout):
        """
        vocab_size: int, 词表大小, 13724
        d_model: int, 词嵌入维度, 512
        n_layers: int, transformer中编码器层数和decoder层数, 6
        heads: int, 多头注意力头个数, 8
        dropout: float, dropout比率, 0.1
        """
        super().__init__()
        self.n_layers = n_layers
        self.embed = Embedder(vocab_size, d_model)  # 词嵌入层，将词转化为词嵌入向量
        self.pe = PositionalEncoder(d_model, dropout=dropout)
        self.layers = get_clones(EncoderLayer(d_model, heads, dropout), n_layers)
        self.norm = Norm(d_model)
    def forward(self, src, src_mask):
        """
        src: (b,seq_len)
        src_mask: (b,1,seq_len): 用于处理源序列（src）中的填充（padding）部分。

        Returns
        -------

        """
        # 1，词嵌入层，将词转化为词嵌入向量
        x = self.embed(src)  # (b,seq_len)->(b,seq_len,d_model)

        # 2，加上位置编码
        x = self.pe(x)  # (b,seq_len,d_model)

        # 3，多头注意力层
        for i in range(self.n_layers):  # 串行encoder_layer
            x = self.layers[i](x, src_mask)
        return self.norm(x)
    
class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, heads, dropout):
        super().__init__()
        self.n_layers = n_layers
        self.embed = Embedder(vocab_size, d_model)
        self.pe = PositionalEncoder(d_model, dropout=dropout)
        self.layers = get_clones(DecoderLayer(d_model, heads, dropout), n_layers)
        self.norm = Norm(d_model)
    def forward(self, trg, e_outputs, src_mask, trg_mask):
        x = self.embed(trg)
        x = self.pe(x)
        for i in range(self.n_layers):
            x = self.layers[i](x, e_outputs, src_mask, trg_mask)
        return self.norm(x)

# encoder -> decoder -> fn
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, trg_vocab_size, d_model, n_layers, heads, dropout):
        super().__init__()
        assert d_model % heads == 0  # 嵌入向量维度必须能被head头数整除
        assert dropout < 1

        self.encoder = Encoder(src_vocab_size, d_model, n_layers, heads, dropout)  #
        self.decoder = Decoder(trg_vocab_size, d_model, n_layers, heads, dropout)
        self.out = nn.Linear(d_model, trg_vocab_size)  # (b,seq_len2,d_model) -> (b,seq_len2,vocab_size)
        self._init_weights_()

    def forward(self, src, trg, src_mask, trg_mask):
        """
        输入原文src，前面翻译的词trg_input，还有对应的mask。得到预测结果。
        src: (b, seq_len1)
        trg: (b, seq_len2)
        src_mask: (b,1,seq_len): (b, 1, seq_len1) src_mask 主要用于处理源序列（src）中的填充（padding）部分。
        trg_mask: (b, seq_len2, seq_len2) trg_mask 遮住前面的词和填充（padding）部分。

        Returns:
        """
        # src.shape=(b,seq_len1); src_mask.shape=(b,1,seq_len1); e_outputs.shape=(b,seq_len1,d_model)
        e_outputs = self.encoder(src, src_mask)  # 提取原文的编码向量。

        #print("DECODER")
        # trg:(b,seq_len2); e_outputs:(b,seq_len1,d_model); src_mask:(b,1,seq_len1); d_output: (b,seq_len2,seq_len2)
        d_output = self.decoder(trg, e_outputs, src_mask, trg_mask)  # (b,seq_len2,d_model)
        output = self.out(d_output)  # (b,seq_len2,d_model) -> (b,seq_len2,vocab_size)
        return output

    def _init_weights_(self):
        for p in self.parameters():
            if p.dim() > 1:  # 矩阵维度大于1，则初始化权重，用于跳过偏置向量等一维参数，只对二维及以上的权重矩阵进行Xavier初始化。
                nn.init.xavier_uniform_(p)

def get_model(opt, src_vocab_size, trg_vocab_size):
    model = Transformer(src_vocab_size, trg_vocab_size, opt.d_model, opt.n_layers, opt.heads, opt.dropout)
    model.to(opt.device)
       
    if opt.load_weights is not None:
        print("loading pretrained weights...")
        model.load_state_dict(torch.load(f'{opt.load_weights}/model_weights'))
    # else:  # 训练的权重不存在，则初始化权重
    #     for p in model.parameters():
    #         if p.dim() > 1:  # 矩阵维度大于1，则初始化权重，用于跳过偏置向量等一维参数，只对二维及以上的权重矩阵进行Xavier初始化。
    #             nn.init.xavier_uniform_(p)
    
    return model
    
