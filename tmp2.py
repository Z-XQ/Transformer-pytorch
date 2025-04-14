import numpy as np
import torch
import torch.nn as nn

def lower_mask(seq_len):
    """seq_len: int.
    return: (1,seq_len,seq_len)
    """
    np_mask = np.triu(np.ones(shape=(1, seq_len, seq_len)), k=1).astype("uint8")
    np_mask = torch.from_numpy(np_mask == 0)  # (1,seq_len2,seq_len2)
    return np_mask

def create_masks(src, trg, padding_num=1):
    src_mask = (src != padding_num).unsqueeze(-2)  # (b,1,seq_len1)
    if trg is not None:
        trg_mask = (trg != padding_num).unsqueeze(-2)  # (b,1,seq_len2)
        seq_len2 = trg.size(1)
        np_mask = lower_mask(seq_len2)  # (1, seq_len2, seq_len2)
        # (b,1,seq_len2) & (1,seq_len2,seq_len2) -> (b,seq_len2,seq_len2) & (1,seq_len2,seq_len2)
        # -> (b,seq_len2,seq_len2) & (b,seq_len2,seq_len2) -> (b,seq_len2,seq_len2)
        trg_mask = trg_mask & np_mask
    else:
        trg_mask = None

    return src_mask, trg_mask

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.embed(x)




class Encoder(nn.Module):
    pass


class Decoder(nn.Module):
    pass

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, trg_vocab_size, d_model, n_layers, heads, dropout):
        super().__init__()
        pass

if __name__ == '__main__':
    src_vocab_size = 13724
    trg_vocab_size = 19109
    src = torch.randint(0, src_vocab_size, (65, 17))
    trg = torch.randint(0, trg_vocab_size, (65, 23))
    trg_input = trg[:, :-1]  # (65,22). 翻译的词，去掉最后一个词，因为最后一个词是<eos>，我们不需要预测<eos>。 在 Python 中，切片操作遵循 “左闭右开” 原则

    src_mask, trg_mask = create_masks(src, trg_input)
    src_mask = src_mask.to(torch.device("cuda"))
    trg_mask = trg_mask.to(torch.deivce("cuda"))
    src = src.to(torch.device("cuda"))
    trg = trg.to(torch.device("cuda"))
    trg_input = trg_input.to(torch.device("cuda"))

    d_model = 512
    n_layers = 6
    heads = 8
    dropout = 0.1
    model = Transformer(src_vocab_size, trg_vocab_size, d_model, n_layers, heads, dropout)


