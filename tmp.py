import torch
import numpy as np
from Batch import nopeak_mask


def nopeak_mask(size):
    """
    size: int. seq_len

    Returns: tensor. (1, seq_len, seq_len)

    """
    np_mask = np.triu(np.ones(1, size, size), k=1).astype('uint8')
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
    # 1, src mask, 去掉padding部分 (b,1,seq_len)
    src_mask = (src != padding_num).unsqueeze(-2).to(torch.device("cuda"))

    # 2, trg mask, 去掉padding部分和后面的词。
    if trg is not None:
        # padding 部分
        trg_mask = (trg != padding_num).unsqueeze(-2).to(torch.device("cuda"))
        # 后面的词
        size = trg.size(1)
        np_mask = nopeak_mask(size).to(torch.device("cuda"))  # (1, seq_len, seq_len)
        np_mask = np_mask & np_mask

if __name__ == '__main__':
    src = torch.randint(0, 13724, (65, 17))  # 原文
    trg = torch.randint(0, 23469, (65, 23))  # 翻译
    trg_input = trg[:, :-1]  # 翻译的词，去掉最后一个词，因为最后一个词是<eos>，我们不需要预测<eos>。

