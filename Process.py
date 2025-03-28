import pandas as pd
import torchtext
from torchtext.legacy import data
from Tokenize import tokenize
from Batch import MyIterator, batch_size_fn
import os
import dill as pickle

import en_core_web_sm
import fr_core_news_sm

def read_data(opt):
    """

    Parameters
    ----------
    opt
    输入：文本文件的路径
    Returns:
    src_data: [text1, text2, ...];
    trg_data: [text1',text2', ...];
    -------

    """
    if opt.src_data is not None:
        try:
            # 打开文件路径→读取全部内容→去除首尾空白→按换行符分割为列表。
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
    # spacy支持多种语言模型，可以通过windows的命令来下载安装，
    # 可以在https://spacy.io/models/zh找到支持的语言模型
    # 英文模型：spacy download en_core_web_sm
    spacy_langs = ['en_core_web_sm', 'fr_core_news_sm', 'de', 'es', 'pt', 'it', 'nl']
    if opt.src_lang not in spacy_langs:
        print('invalid src language: ' + opt.src_lang + 'supported languages : ' + str(spacy_langs))
    if opt.trg_lang not in spacy_langs:
        print('invalid trg language: ' + opt.trg_lang + 'supported languages : ' + str(spacy_langs))
    
    print("loading spacy tokenizers...")

    # 加载分词器
    t_src = tokenize(opt.src_lang)  # 加载分词模型，创建src 分词器
    t_trg = tokenize(opt.trg_lang)  # 加载分词模型，创建trg 分词器

    # 创建field，用来对数据进行预处理，比如分词、去停用词、词干化等。
    TRG = data.Field(lower=True, tokenize=t_trg.tokenizer, init_token='<sos>', eos_token='<eos>')
    SRC = data.Field(lower=True, tokenize=t_src.tokenizer)

    # 加载预训练的vocab（vocab是字典，里面包含了单词和单词的索引）
    # SRC.pkl文件是Python 的 pickle 模块序列化后保存的 Field 对象
    """
    复用词汇表：在自然语言处理任务中，词汇表的构建是基于训练数据的。当你训练好一个模型后，如果后续需要重新训练或者进行推理，为了保证输入数据的处理方式一致，
    需要使用相同的词汇表。保存 Field 对象可以保存其内部的词汇表信息，这样在后续使用时就不需要重新构建词汇表。
    """
    if opt.load_weights is not None:
        try:
            print("loading presaved fields...")
            SRC = pickle.load(open(f'{opt.load_weights}/SRC.pkl', 'rb'))
            TRG = pickle.load(open(f'{opt.load_weights}/TRG.pkl', 'rb'))
        except:
            print("error opening SRC.pkl and TXT.pkl field files, please ensure they are in " + opt.load_weights + "/")
            quit()
        
    return(SRC, TRG)

def create_dataset(opt, SRC, TRG):

    print("creating dataset and iterator... ")

    raw_data = {'src' : [line for line in opt.src_data], 'trg': [line for line in opt.trg_data]}
    df = pd.DataFrame(raw_data, columns=["src", "trg"])
    
    mask = (df['src'].str.count(' ') < opt.max_strlen) & (df['trg'].str.count(' ') < opt.max_strlen)
    df = df.loc[mask]

    df.to_csv("translate_transformer_temp.csv", index=False)
    
    data_fields = [('src', SRC), ('trg', TRG)]
    train = data.TabularDataset('./translate_transformer_temp.csv', format='csv', fields=data_fields)

    train_iter = MyIterator(train, batch_size=opt.batchsize, device=opt.device,
                        repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
                        batch_size_fn=batch_size_fn, train=True, shuffle=True)
    
    os.remove('translate_transformer_temp.csv')

    if opt.load_weights is None:
        SRC.build_vocab(train)  # 构建词汇表
        TRG.build_vocab(train)
        if opt.checkpoint > 0:  # 大于0，说明需要保存权重，创建文件夹
            try:
                os.mkdir("weights")
            except:
                print("weights folder already exists, run program with -load_weights weights to load them")
                quit()
            # 先保存词汇表，后面加载的时候需要用到
            pickle.dump(SRC, open('weights/SRC.pkl', 'wb'))
            pickle.dump(TRG, open('weights/TRG.pkl', 'wb'))

    opt.src_pad = SRC.vocab.stoi['<pad>']
    opt.trg_pad = TRG.vocab.stoi['<pad>']

    opt.train_len = get_len(train_iter)

    return train_iter

def get_len(train):

    for i, b in enumerate(train):
        pass
    
    return i
