from dill import pickle
from torchtext.legacy import data

from Tokenize import CustomerTokenizer


def read_data(opt):
    """input: dataPath
        returns: [], []"""
    if opt.src_data is not None:
        opt.src_data = open(opt.src_data, encoding="utf-8").read().strip().split("\n")

    if opt.trg_data is not None:
        opt.trg_data = open(opt.trg_data, encoding="utf-8").read().strip().split("\n")


def create_fields(opt):
    # 分词函数
    spacy_langs = ['en_core_web_sm', 'fr_core_news_sm', 'de', 'es', 'pt', 'it', 'nl']
    if opt.src_lang not in spacy_langs:
        print('invalid src language: ' + opt.src_lang + 'supported languages : ' + str(spacy_langs))
    if opt.trg_lang not in spacy_langs:
        print('invalid trg language: ' + opt.trg_lang + 'supported languages : ' + str(spacy_langs))

    # 加载分词器
    t_src = CustomerTokenizer(opt.src_lang)
    t_trt = CustomerTokenizer(opt.trg_lang)

    # 创建field
    SRC = data.Field(lower=True, tokenize=t_src.tokenize, init_token="<sos>", eos_token="<eos>")
    TRG = data.Field(lower=True, tokenize=t_trt.tokenize, init_token="<sos>", eos_token="<eos>")

    if opt.load_weights is not None:
        SRC = pickle.load(open(f'{opt.load_weights}/SRC.pkl', 'rb'))
        TRG = pickle.load(open(f'{opt.load_weights}/TRG.pkl', 'rb'))

    return SRC, TRG