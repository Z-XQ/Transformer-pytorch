import torch
from torchtext.legacy import data
import numpy as np

def nopeak_mask(size, opt):
    """
    size: seq_len2. int. 8
    opt
    Returns: (1,seq_len2,seq_len2). 生成一个下三角布尔矩阵（对角线及下方为True，上方为False）
    -------

    """
    # k=1 意味着从主对角线右上方第一条对角线开始取上三角部分(第一行只有一个0），主对角线及其下方的元素都会被置为 0。
    np_mask = np.triu(np.ones((1, size, size)), k=1).astype('uint8')  # 上方为1，下方为0(包括主对角线）。(1,seq_len2,seq_len2)
    np_mask = torch.from_numpy(np_mask == 0).to(opt.device)  # 下方为True. (1,seq_len2,seq_len2)
    return np_mask

def create_masks(src, trg, opt):
    """
    src: (b,seq_len1)
    trg: (b,seq_len2)
    opt
    Returns:
        src_mask: (b,1,seq_len1). false is padding location. 用于屏蔽padding部分
        trg_mask: (b,seq_len2,seq_len2). 屏蔽padding部分，且遮住前面的词。下三角+屏蔽padding

    """
    # 倒数第二的位置添加一个新维度
    src_mask = (src != opt.src_pad).unsqueeze(-2).to(opt.device)  # 创建一个mask，用于屏蔽padding部分
    # tmp = src_mask.view(src_mask.shape[0], -1).cpu().numpy()  # 查看对齐情况
    if trg is not None:  # 已经有翻译文本，则是训练模式
        # trg_mask: (b,1,seq_len2)
        trg_mask = (trg != opt.trg_pad).unsqueeze(-2).to(opt.device)  # 用于屏蔽padding部分
        # tmp = trg_mask.view(trg_mask.shape[0], -1).cpu().numpy()  # 查看对齐情况

        # np_mask: (1,seq_len2,seq_len2).
        seq_lens2 = trg.size(1)
        np_mask = nopeak_mask(seq_lens2, opt).to(opt.device)  # (1,seq_len2,seq_len2)
        # tmp = np_mask[0].cpu().numpy()  # 查看对齐情况

        # 屏蔽padding部分，且遮住前面的词 (b,1,seq_len2) & (1,seq_len2,seq_len2) -> (b,seq_len2,seq_len2)
        # 最后维度直接匹配，无需扩展；
        # 第二维度，trg_mask复制seq_len2份：(b,1,seq_len2) -> (b,seq_len2,seq_len2)
        # 第一维度，np_mask复制b份：(1,seq_len2,seq_len2) -> (b,seq_len2,seq_len2)
        # 相乘后的每一行代表：预测当前词时的可见的词（屏蔽后面的词和padding词）。
        trg_mask = trg_mask & np_mask

    else:
        trg_mask = None  # 测试模型，则没有翻译文本，需要逐个生成，也就不需要遮住后面的词
    return src_mask, trg_mask

# patch on Torchtext's batching process that makes it more efficient
# from http://nlp.seas.harvard.edu/2018/04/03/attention.html#position-wise-feed-forward-networks

class MyIterator(data.Iterator):  # 继承基础迭代器类
    """
    自定义迭代器类，用于生成批量数据
    功能：
        根据训练/非训练模式创建不同的数据批处理方式，支持动态批处理大小和排序策略，
        训练模式下增加数据随机性，非训练模式保持数据顺序

    继承并使用父类的__init__方法：
    显式参数，依赖以下实例属性：
        self.train: bool 标识是否为训练模式
        self.batch_size: int 单个批次大小
        self.batch_size_fn: function 动态计算批次大小的函数
        self.sort_key: function 数据排序的键函数
        self.data(): function 获取原始数据的方法
        self.random_shuffler: function 随机洗牌函数
    """

    def create_batches(self):  # 复写基类方法：生成数据批次
        """
        创建数据批次的主方法
        - 训练模式：通过pool方法生成动态随机批次
        - 非训练模式：生成有序的固定批次

        返回值：
            None，结果存储在self.batches属性中
        """
        if self.train:
            def pool(data_set, random_shuffler):
                # 通过torchtext.legacy.data.batch划分成多个大batch_size的数据块，循环遍历每个超大批次数据
                for big_batch_data in data.batch(data=data_set, batch_size=self.batch_size * 100):  # d: list of torchtext.legacy.data.Example
                    # 100*batch_size超大批次数据排序，big_batch_generator是一个generator, 每次迭代返回一个批次，每个批次使用batch_size_fn整理批次内数据
                    big_batch_generator = data.batch(sorted(big_batch_data, key=self.sort_key), self.batch_size, self.batch_size_fn)
                    for batch_data in random_shuffler(list(big_batch_generator)):  # 打乱list（存放多个batch_data），每次迭代随机返回一个批次
                        yield batch_data

            # 创建大容量缓冲池，并对缓冲池数据进行随机洗牌，减少数据排序时间
            # self.data(): 全部的数据，进行了分词的数据集
            self.batches = pool(self.data(), self.random_shuffler)

        else:
            # 直接划分标准批次
            self.batches = []
            for b in data.batch(self.data(), self.batch_size, self.batch_size_fn):
                self.batches.append(sorted(b, key=self.sort_key))  # 保持每个批次内部的有序性


# 同一个batch内部，下次进入batch_size_fn时，要更新批次最大序列长度记录
global max_src_in_batch, max_tgt_in_batch


# 需要知道最大允许总token数量（GPU显存限制），根据GPU显存设置（如12GB≈12,000,000 token）
def batch_size_fn(new_token, current_batch_size, current_total_tokens):
    """动态计算当前批次的实际token总量（包含padding填充部分）（GPU现存限制）
        根据 GPU 显存上限，按 token 总数量动态打包样本，实现动态的batch_size
    Args:
        new_token: 新加入批次的样本对象，这里需包含.src和.trg属性 list。存放分词内容
        current_batch_size: 当前批次已包含的样本数量
        current_total_tokens: 已处理的样本总数（当前未使用该参数）

    Returns:
        int: 当前批次的实际token总量
    """
    global max_src_in_batch, max_tgt_in_batch
    # 初始化批次时重置最大长度记录
    if current_batch_size == 1:
        max_src_in_batch = 0
        max_tgt_in_batch = 0

    # 更新当前批次最大序列长度
    max_src_in_batch = max(max_src_in_batch, len(new_token.src))
    max_tgt_in_batch = max(max_tgt_in_batch, len(new_token.trg) + 2)  # 增加2个标记：句首和句尾标记

    # 计算当前批次的总token数（含padding）
    src_elements = current_batch_size * max_src_in_batch
    tgt_elements = current_batch_size * max_tgt_in_batch

    # 返回两种语言方向的最大token数作为实际批次大小
    return max(src_elements, tgt_elements)

