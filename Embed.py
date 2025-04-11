import torch
import torch.nn as nn
import math
from torch.autograd import Variable

class Embedder(nn.Module):
    def __init__(self, vocab_size, d_model=512):
        super().__init__()
        self.d_model = d_model
        # 定义词嵌入层：接受一个词汇表大小和嵌入维度作为参数，并创建一个可以学习的权重矩阵。
        self.embed = nn.Embedding(vocab_size, d_model)
    def forward(self, x):
        return self.embed(x)  # (b,seq_len)->(b,seq_len,d_model)

class PositionalEncoder(nn.Module):
    def __init__(self, d_model=512, max_seq_len = 200, dropout = 0.1):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)

        # create constant 'pe' matrix with values dependant on pos and i： 维度i，对应的分子是2i
        # pe = torch.zeros(max_seq_len, d_model)
        # for pos in range(max_seq_len):  # 遍历每一个token位置，每个token位置上都生成一个位置embedding向量
        #     for i in range(0, d_model, 2):
        #         pe[pos, i] = math.sin(pos / (10000 ** ((2 * i)/d_model)))
        #         pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))  # 位置是i+1，分子是2(i+1)。
        # pe = pe.unsqueeze(0)
        # self.register_buffer('pe', pe)

        # 2，高效率实现方式：矩阵实现
        self.pe = torch.zeros(max_seq_len, d_model)  # shape=(5000, 512)
        pos = torch.arange(0, max_seq_len).unsqueeze(1)  # shape=(5000, 1), 每个位置对应一个嵌入向量
        # 指数计算: 2i*(-ln(10000)/512). i的值范围是[0,256]
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)).unsqueeze(0)

        # self.pe[:, 0::2].shape=(5000,256) (5000,1) * (1,256) = (5000,256)
        self.pe[:, 0::2] = torch.sin(pos * div_term)
        self.pe[:, 1::2] = torch.cos(pos * div_term)
        self.pe = self.pe.unsqueeze(0)  # (5000,256)->(1,5000,256)

    def forward(self, x):
        """
        x: token_embedding. (b,seq_len,d_model)
        Returns: token_embedding + position_embedding
        """
        # 1, make embeddings relatively larger
        # 乘以 √d_model 后，嵌入向量的模长变为√(d_model * 1) = √d_model，
        # 与位置编码的模长量级一致（位置编码模长约为 √(d_model/2)，确保两者在加法中权重均衡。
        x = x * math.sqrt(self.d_model)
        # 2, 获取位置编码向量
        seq_len = x.size(1)
        # cur_pe = Variable(self.pe[:,:seq_len], requires_grad=False)  # 显示说明这是一个常量位置编码向量，不需要求梯度
        cur_pe = self.pe[:, :seq_len].requires_grad_(False)  # 显示说明这是一个常量位置编码向量，不需要求梯度
        # cur_pe = self.pe[:, :seq_len]  # 直接切片，自动继承 requires_grad=False

        if x.is_cuda:
            cur_pe = cur_pe.cuda()
        # 3，# add constant to embedding，即常量位置编码加到词嵌入向量上
        x = x + cur_pe
        return self.dropout(x)


if __name__ == '__main__':
    # 测试Embedder类
    vocab_size = 100
    d_model = 512
    embedder = Embedder(vocab_size, d_model)
    x = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    print(embedder(x).shape)
    print(embedder(x))
    print(embedder(x).size())
