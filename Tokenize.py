import spacy  # 分词库
import re  # 正则表达式库，用于替换字符

class CustomerTokenizer(object):
    """
    一个用于分词的类，支持对输入句子进行预处理和分词操作。
    """
    
    def __init__(self, lang):
        """
        初始化tokenize类，加载指定语言的spacy模型。
        lang : str, 'en_core_web_sm', 'fr_core_news_sm', 'de', 'es', 'pt', 'it', 'nl'
        """
        # 创建一个spacy对象，用于分词
        self.nlp = spacy.load(lang)  # Package name or model path.
            
    def tokenize(self, sentence):
        """
        对输入的句子进行预处理和分词，返回分词后的列表。
        sentence : str 需要分词的句子。
        Returns:  list 分词后的单词列表，去除空格。eg. ['sentence', 'dl', 'hello', 'how', 'are', 'you', '?', 'I', 'am', 'fine', '.']
        """

        # 1，使用正则表达式库re，将句子中的特殊字符（比如星号(*)、双引号(")、中文引号(“”)、换行符(\n)、反斜杠(\)、省略号(…)等），替换成空格。
        # 在机器翻译中，特殊字符不重要，甚至干扰模型（这些符号如果出现频率低）。
        sentence = re.sub(
        r"[\*\"“”\n\\…\+\-\/\=\(\)‘•:\[\]\|’\!;]", " ", str(sentence))
        sentence = re.sub(r"[ ]+", " ", sentence)  # 去除多余的空格。
        sentence = re.sub(r"\!+", "!", sentence)  # 去除多余的感叹号。
        sentence = re.sub(r"\,+", ",", sentence)
        sentence = re.sub(r"\?+", "?", sentence)
        sentence = sentence.lower()
        
        # 2，使用spacy的tokenizer方法进行分词，并过滤掉空格
        doc = self.nlp.tokenizer(sentence)
        return [tok.text for tok in doc if tok.text != " "]
