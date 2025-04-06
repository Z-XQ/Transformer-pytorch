import pandas as pd
from torchtext.legacy import data
from Tokenize import CustomerTokenizer
from Batch import MyIterator, batch_size_fn
import os
import dill as pickle

import en_core_web_sm
import fr_core_news_sm

def read_data(opt):
    """
    输入：文本文件的路径
    Returns:
        src_data: [text1, text2, ...];
        trg_data: [text1',text2', ...];
    """
    if opt.src_data is not None:
        try:
            # 打开文件路径→读取全部内容→去除首尾空白→按换行符分割为列表。
            # 去除首尾空白不仅指去除空格，还包括去除换行符（\n）、制表符（\t）等其他空白字符。strip()方法会移除字符串开头和结尾的所有空白字符。
            opt.src_data = open(opt.src_data, encoding='utf-8').read().strip().split('\n')
        except:
            print("error: '" + opt.src_data + "' file not found")
            quit()
    
    if opt.trg_data is not None:
        try:
            # 打开文件路径→读取全部内容→去除首尾空白→按换行符分割为列表。
            opt.trg_data = open(opt.trg_data, encoding='utf-8').read().strip().split('\n')
        except:
            print("error: '" + opt.trg_data + "' file not found")
            quit()

def create_fields(opt):
    spacy_langs = ['en_core_web_sm', 'fr_core_news_sm', 'de', 'es', 'pt', 'it', 'nl']
    if opt.src_lang not in spacy_langs:
        print('invalid src language: ' + opt.src_lang + 'supported languages : ' + str(spacy_langs))
    if opt.trg_lang not in spacy_langs:
        print('invalid trg language: ' + opt.trg_lang + 'supported languages : ' + str(spacy_langs))
    
    print("loading spacy tokenizers...")

    # 加载分词器
    t_src = CustomerTokenizer(opt.src_lang)  # 加载分词模型，创建src 分词器
    t_trg = CustomerTokenizer(opt.trg_lang)  # 加载分词模型，创建trg 分词器

    # 创建field，用来对数据进行预处理，比如分词、去停用词、添加特殊标记（如起始标记 <sos> 和结束标记 <eos>）、词干化等。
    # tokenize：data.Field 会使用自定义的分词器来处理输入文本，从而实现文本的分词和预处理。
    # lower: 不会影响模型的理解能力，因为模型主要关注词汇的语义而非大小写。如果需要保留大小写信息，可以在预处理阶段添加额外的标记或特征。
    TRG = data.Field(lower=True, tokenize=t_trg.tokenize, init_token='<sos>', eos_token='<eos>')
    SRC = data.Field(lower=True, tokenize=t_src.tokenize)

    # 加载预训练的vocab（vocab是字典，里面包含了单词和单词的索引）
    # SRC.pkl文件是Python 的 pickle 模块序列化后保存的 Field 对象
    """
    复用词汇表：在自然语言处理任务中，词汇表的构建是基于训练数据的。当你训练好一个模型后，如果后续需要重新训练或者进行推理，为了保证输入数据的处理方式一致，
    需要使用相同的词汇表。保存 Field 对象可以保存其内部的词汇表信息，这样在后续使用时就不需要重新构建词汇表。
    """
    if opt.load_weights is not None:
        try:
            print("loading presaved fields...")
            # 该文件保存了Field对象的所有属性（包括词汇表、分词器、预处理配置等），通过Python的类实例序列化机制重建对象内存结构，确保与原始对象完全一致。
            SRC = pickle.load(open(f'{opt.load_weights}/SRC.pkl', 'rb'))  # rb: read binary
            TRG = pickle.load(open(f'{opt.load_weights}/TRG.pkl', 'rb'))
        except:
            print("error opening SRC.pkl and TXT.pkl field files, please ensure they are in " + opt.load_weights + "/")
            quit()
        
    return(SRC, TRG)

def create_dataset(opt, SRC, TRG):
    """
    SRC: src data.Field
    TRG: trg data.Field
    Returns: dataset and iterator
    -------
    """
    print("creating dataset and iterator... ")

    # 1，过滤掉超长文本，并保存成本地csv文件
    # {src: list of raw text, trg: list of raw text}
    raw_data = {'src': opt.src_data, 'trg': opt.trg_data}
    # 创建dataframe: 两列，分别对应src和trg；每一行分别是输入句子，翻译句子
    df = pd.DataFrame(raw_data, columns=["src", "trg"])

    # 过滤src和trg长度都超长文本的数据
    # 句子中的空格数就是单词数，筛选出源文本和译文单词数均小于max_strlen的数据行（空格数+1≈单词数），用于过滤超长文本。
    mask = (df['src'].str.count(' ') < opt.max_strlen) & (df['trg'].str.count(' ') < opt.max_strlen)
    df = df.loc[mask]  # 将满足条件的数据行保留在DataFrame中，并删除不满足条件的数据行。

    # src raw text and trg raw text保存到本地csv文件
    df.to_csv("translate_transformer_temp.csv", index=False)

    # 2，读取csv文件数据集，创建dataset
    data_fields = [('src', SRC), ('trg', TRG)]  # 设置Field，每个Field对应一个列，包括列名、列对应的Field对象。
    train_dataset = data.TabularDataset('./translate_transformer_temp.csv', format='csv', fields=data_fields)

    # 3，创建batch iterator
    # 双长度排序的机制是：先按源语句长度排序，长度相同时再按目标语句长度排序。这种双重排序可有效减少同一batch内不同样本的填充量，提升GPU计算效率。
    train_dataset_iter = MyIterator(
        dataset=train_dataset,
        batch_size=opt.batchsize,  # batch_size=1500
        device=opt.device,
        repeat=False,
        sort_key=lambda x: (len(x.src), len(x.trg)),  # sort_key：对数据集进行排序，排序规则为按照src和trg的长度进行排序。这可能用于动态批处理，将相似长度的样本放在同一批次，减少填充。
        batch_size_fn=batch_size_fn, train=True, shuffle=True)  # batch_size_fn：自定义batch大小函数，该函数用于动态调整batch大小，根据每个样本的长度进行计算。

    os.remove('translate_transformer_temp.csv')

    # 如果没有预训练权重，则构建词汇表，保存分词器
    if opt.load_weights is None:
        SRC.build_vocab(train_dataset)  # 会自动抽取train_dataset的src列文本构建源语言词汇表
        TRG.build_vocab(train_dataset)  # 会自动抽取train_dataset的trg列文本构建目标语言词汇表
        if opt.checkpoint > 0:  # 大于0，说明需要保存权重，创建文件夹
            try:
                os.mkdir("weights")
            except:
                print("weights folder already exists, run program with -load_weights weights to load them")
                quit()
            # 先保存词汇表，后面加载的时候需要用到
            pickle.dump(SRC, open('weights/SRC.pkl', 'wb'))
            pickle.dump(TRG, open('weights/TRG.pkl', 'wb'))

    # 获取pad的索引
    # 设置源语言与目标语言的填充符索引
    # 从SRC/TRG词汇表中获取'<pad>'符号对应的索引值
    # 用于后续数据处理或模型中的序列对齐及掩码生成
    opt.src_pad = SRC.vocab.stoi['<pad>']
    opt.trg_pad = TRG.vocab.stoi['<pad>']

    opt.train_len = get_len(train_dataset_iter)  # 遍历一遍dataset，获取最后一个batch_data索引，作为train_len

    return train_dataset_iter

def get_len(train_iter):
    """get train_dataset len: 多少个batch"""
    # 只是遍历train_dataset_iter，没有前向和反向推理，所以很快。i为batch索引，b为batch数据
    for i, b in enumerate(train_iter):
        pass
    
    return i  # 返回最后一个batch索引
