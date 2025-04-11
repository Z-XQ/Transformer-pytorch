import torch
import numpy as np
from Batch import nopeak_mask


def nopeak_mask(seq_len):
    np_mask = np.triu(np.ones(1, seq_len, seq_len), k=1).astype("uint8")
    np_mask = torch.from_numpy(np_mask==0).to(torch.device("cuda"))
    return np_mask

def create_masks(src, trg, padding_num=1):
    # src mask
    src_mask = (src != padding_num).unsqueeze(-2).to(torch.device("cuda"))

    # trg mask
    if trg is not None:
        # (b,1,seq_len2)
        trg_mask = (trg != padding_num).unsqueeze(-2).to(torch.device("cuda"))
        # 后面的词 (b,seq_len2,seq_len2)
        seq_lens2 = trg.size(1)
        np_mask = nopeak_mask(seq_lens2)  # (1,seq_len2,seq_len2）
        # (b,1,seq_len2) & (1,seq_len2,seq_len2) -> (b,seq_len2,seq_len2)
        trg_mask = trg_mask & np_mask
    else:
        trg_mask = None

    return src_mask, trg_mask


if __name__ == '__main__':
    create_masks()