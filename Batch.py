import torch
from torchtext.legacy import data
import numpy as np
from torch.autograd import Variable


def nopeak_mask(size, opt):
    np_mask = np.triu(np.ones((1, size, size)), k=1).astype('uint8')
    np_mask = Variable(torch.from_numpy(np_mask == 0).to(opt.device))
    return np_mask

def create_masks(src, trg, opt):
    
    src_mask = (src != opt.src_pad).unsqueeze(-2).to(opt.device)

    if trg is not None:
        trg_mask = (trg != opt.trg_pad).unsqueeze(-2).to(opt.device)
        size = trg.size(1)  # get seq_len for matrix
        np_mask = nopeak_mask(size, opt).to(opt.device)
        trg_mask = trg_mask & np_mask

    else:
        trg_mask = None
    return src_mask, trg_mask

# patch on Torchtext's batching process that makes it more efficient
# from http://nlp.seas.harvard.edu/2018/04/03/attention.html#position-wise-feed-forward-networks

class MyIterator(data.Iterator):
    """
    自定义迭代器类，用于生成批量数据

    功能：
        根据训练/非训练模式创建不同的数据批处理方式，支持动态批处理大小和排序策略，
        训练模式下增加数据随机性，非训练模式保持数据顺序

    继承：
        data.Iterator: 基础迭代器类
    """

    def create_batches(self):
        """
        创建数据批次的主方法

        根据self.train标志位选择不同的批处理策略：
        - 训练模式：通过pool方法生成动态随机批次
        - 非训练模式：生成有序的固定批次

        参数说明：
            无显式参数，依赖以下实例属性：
            self.train: bool 标识是否为训练模式
            self.batch_size: int 单个批次大小
            self.batch_size_fn: function 动态计算批次大小的函数
            self.sort_key: function 数据排序的键函数
            self.data(): function 获取原始数据的方法
            self.random_shuffler: function 随机洗牌函数

        返回值：
            None，结果存储在self.batches属性中
        """
        if self.train:
            # 训练模式批处理流程：
            # 1. 创建大容量缓冲池（100倍批次大小）
            # 2. 对缓冲池数据排序后划分标准批次
            # 3. 对批次进行随机洗牌增加数据随机性
            def pool(d, random_shuffler):
                for p in data.batch(data=d, batch_size=self.batch_size * 100):  # d: list of torchtext.legacy.data.Example
                    p_batch = data.batch(  # p_batch: generator of list of torchtext.legacy.data.Example
                        sorted(p, key=self.sort_key),  # p: the number of example is 100*batch_size.
                        self.batch_size, self.batch_size_fn)
                    for b in random_shuffler(list(p_batch)):  # b: list of torchtext.legacy.data.Example.
                        yield b

            # 创建大容量缓冲池，并对缓冲池数据进行随机洗牌，减少数据排序时间
            # self.data(): 获取原始数据，返回迭代器
            self.batches = pool(self.data(), self.random_shuffler)

        else:
            # 非训练模式批处理流程：
            # 1. 直接划分标准批次
            # 2. 保持每个批次内部的有序性
            self.batches = []
            for b in data.batch(self.data(), self.batch_size,
                                self.batch_size_fn):
                self.batches.append(sorted(b, key=self.sort_key))


global max_src_in_batch, max_tgt_in_batch


def batch_size_fn(new, count, sofar):
    """动态计算当前批次的实际token总量（包含padding填充部分）

    通过跟踪当前批次中最长的源语言序列和目标语言序列长度，计算当前批次的总token数。
    该函数适用于动态批处理策略，在保证序列长度相近的同时最大化GPU利用率。

    Args:
        new: 新加入批次的样本对象，需包含.src和.trg两个长度属性
        count: 当前批次已包含的样本数量
        sofar: 已处理的样本总数（当前未使用该参数）

    Returns:
        int: 当前批次的实际token总量，取源语言侧和目标语言侧的最大值

    实现策略：
    1. 每个新样本加入时更新批次最大长度
    2. 目标语言长度+2是考虑添加句首和句尾标记
    3. 总token数 = 样本数 × 当前最长序列长度
    """
    global max_src_in_batch, max_tgt_in_batch
    # 初始化批次时重置最大长度记录
    if count == 1:
        max_src_in_batch = 0
        max_tgt_in_batch = 0

    # 更新当前批次最大序列长度
    max_src_in_batch = max(max_src_in_batch, len(new.src))
    max_tgt_in_batch = max(max_tgt_in_batch, len(new.trg) + 2)

    # 计算当前批次的总token数（含padding）
    src_elements = count * max_src_in_batch
    tgt_elements = count * max_tgt_in_batch

    # 返回两种语言方向的最大token数作为实际批次大小
    return max(src_elements, tgt_elements)

