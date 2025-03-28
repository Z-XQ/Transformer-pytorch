import spacy
import re

class tokenize(object):
    """
    一个用于分词的类，支持对输入句子进行预处理和分词操作。

    Attributes
    ----------
    nlp : spacy.Language
        用于分词的spacy语言模型对象。

    Methods
    -------
    __init__(lang)
        初始化tokenize类，加载指定语言的spacy模型。
    tokenizer(sentence)
        对输入的句子进行预处理和分词，返回分词后的列表。
    """
    
    def __init__(self, lang):
        """
        初始化tokenize类，加载指定语言的spacy模型。

        Parameters
        ----------
        lang : str
            代表的是语言，可以是英文，也可以是中文。该变量类型是字符串。
        """
        # 创建一个spacy对象，用于分词
        self.nlp = spacy.load(lang)  # Package name or model path.
            
    def tokenizer(self, sentence):
        """
        对输入的句子进行预处理和分词，返回分词后的列表。

        Parameters
        ----------
        sentence : str
            需要分词的句子。

        Returns
        -------
        list
            分词后的单词列表，去除空格。eg. ['sentence', 'dl', 'hello', 'how', 'are', 'you', '?', 'I', 'am', 'fine', '.']
        """
        # 使用正则表达式去除句子中的特殊字符和多余的空格
        # 包括：包括星号(*)、双引号(")、中文引号(“”)、换行符(\n)、反斜杠(\)、省略号(…)、加号(+)、
        # 减号(-)、斜杠(/)、等号(=)、括号(())、
        # 中文单引号(‘’)、冒号(:)、方括号([])、竖线(|)、感叹号(!)和分号(;)。
        # 将这些特殊字符替换为空格。
        sentence = re.sub(
        r"[\*\"“”\n\\…\+\-\/\=\(\)‘•:\[\]\|’\!;]", " ", str(sentence))
        sentence = re.sub(r"[ ]+", " ", sentence)
        sentence = re.sub(r"\!+", "!", sentence)
        sentence = re.sub(r"\,+", ",", sentence)
        sentence = re.sub(r"\?+", "?", sentence)
        sentence = sentence.lower()
        
        # 使用spacy进行分词，并过滤掉空格
        return [tok.text for tok in self.nlp.tokenizer(sentence) if tok.text != " "]
