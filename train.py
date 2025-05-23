import argparse
import time
import torch
from Models import get_model
from Process import *
import torch.nn.functional as F
from Optim import CosineWithRestarts
from Batch import create_masks
import dill as pickle

from simple import Transformer


def train_model(model, opt):
    print("training model...")
    model.train()
    start = time.time()
    if opt.checkpoint > 0:
        cptime = time.time()
                 
    for epoch in range(opt.epochs):

        total_loss = 0
        if opt.floyd is False:
            print("   %dm: epoch %d [%s]  %d%%  loss = %s" %\
            ((time.time() - start)//60, epoch + 1, "".join(' '*20), 0, '...'), end='\r')
        
        if opt.checkpoint > 0:
            torch.save(model.state_dict(), 'weights/model_weights')
                    
        for i, batch in enumerate(opt.train_dataset_iter):

            src = batch.src.transpose(0, 1).to(opt.device)  # (seq_len1, b) -> (b, seq_len1)
            trg = batch.trg.transpose(0, 1).to(opt.device)  # (seq_len2, b) -> (b, seq_len2)
            trg_input = trg[:, :-1]  # (b, seq_len2-1)。去掉最后一个词。

            src_mask, trg_mask = create_masks(src, trg_input, opt)
            src_mask = src_mask.to(opt.device)  # (b, 1, seq_len1) src_mask 主要用于处理源序列（src）中的填充（padding）部分。
            trg_mask = trg_mask.to(opt.device)  # (b, seq_len2, seq_len2) trg_mask 遮住前面的词和填充（padding）部分。

            # 输入原文src，前面翻译的词trg_input，还有对应的mask。得到预测结果。
            preds = model(src, trg_input, src_mask, trg_mask)  # preds.shape=(b, seq_len2-1, vocab_size)
            preds = preds.view(-1, preds.size(-1))  # (b, seq_len2-1, vocab_size) -> (b*(seq_len2-1), vocab_size)
            ys = trg[:, 1:].contiguous().view(-1)  # (b, seq_len2-1, vocab_size) -> (b*(seq_len2-1),). ys代表预测的词，即翻译的词。 ys.shape=

            # preds: (b*(seq_len2-1),vocab_size); ys.shape=(b*(seq_len2-1), )
            loss = F.cross_entropy(preds, ys, ignore_index=opt.trg_pad)
            opt.optimizer.zero_grad()
            loss.backward()
            opt.optimizer.step()
            if opt.SGDR == True: 
                opt.sched.step()
            
            total_loss += loss.item()
            
            if (i + 1) % opt.printevery == 0:
                 p = int(100 * (i + 1) / opt.train_len)
                 avg_loss = total_loss/opt.printevery
                 if opt.floyd is False:
                    print("   %dm: epoch %d [%s%s]  %d%%  loss = %.3f" %\
                    ((time.time() - start)//60, epoch + 1, "".join('#'*(p//5)), "".join(' '*(20-(p//5))), p, avg_loss), end='\r')
                 else:
                    print("   %dm: epoch %d [%s%s]  %d%%  loss = %.3f" %\
                    ((time.time() - start)//60, epoch + 1, "".join('#'*(p//5)), "".join(' '*(20-(p//5))), p, avg_loss))
                 total_loss = 0
            
            if opt.checkpoint > 0 and ((time.time()-cptime)//60) // opt.checkpoint >= 1:
                torch.save(model.state_dict(), 'weights/model_weights')
                cptime = time.time()
   
   
        print("%dm: epoch %d [%s%s]  %d%%  loss = %.3f\nepoch %d complete, loss = %.03f" %\
        ((time.time() - start)//60, epoch + 1, "".join('#'*(100//5)), "".join(' '*(20-(100//5))), 100, avg_loss, epoch + 1, avg_loss))

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-src_data', required=True)  # src数据集路径
    parser.add_argument('-trg_data', required=True)  # dst数据集路径
    parser.add_argument('-src_lang', required=True)  # src语言模型，eg. en_core_web_sm
    parser.add_argument('-trg_lang', required=True)  # dst语言模型，eg. en_core_web_sm
    parser.add_argument('-no_cuda', action='store_true')
    parser.add_argument('-SGDR', action='store_true')
    parser.add_argument('-epochs', type=int, default=2)
    parser.add_argument('-d_model', type=int, default=512)  # 嵌入向量维度
    parser.add_argument('-n_layers', type=int, default=6)   # transformer中编码器层数和decoder层数
    parser.add_argument('-heads', type=int, default=8)      # 多头注意力头个数
    parser.add_argument('-dropout', type=int, default=0.1)
    parser.add_argument('-batchsize', type=int, default=1500)  # batchsize
    parser.add_argument('-printevery', type=int, default=100)  # 每隔多少步打印一次loss
    parser.add_argument('-lr', type=int, default=0.0001)  # 学习率
    parser.add_argument('-load_weights')
    parser.add_argument('-create_valset', action='store_true')  # 是否创建验证集
    parser.add_argument('-max_strlen', type=int, default=80)  # 最大句子长度
    parser.add_argument('-floyd', action='store_true')  # floyd平台
    parser.add_argument('-checkpoint', type=int, default=0)  # 保存模型权重的间隔

    opt = parser.parse_args()

    opt.device = 'cuda' if opt.no_cuda is False else 'cpu'
    if opt.device == 'cuda':
        assert torch.cuda.is_available()

    # 1, 读取数据：读取src和dst文本文件，转成list数据
    read_data(opt)

    # 2, 创建field：创建src和dst的field，并创建vocab
    SRC, TRG = create_fields(opt)

    # 3, 创建dataset：创建src和dst的dataset
    opt.train_dataset_iter = create_dataset(opt, SRC, TRG)

    # 4, 创建model：创建transformer模型
    # model = get_model(opt, len(SRC.vocab), len(TRG.vocab))
    model = Transformer(
        src_vocab_size=len(SRC.vocab),
        trg_vocab_size=len(TRG.vocab),
        d_model=opt.d_model,
        n_layers=opt.n_layers,
        heads=opt.heads,
        dropout=opt.dropout
    )
    model.to(torch.device(opt.device))

    opt.optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.98), eps=1e-9)
    if opt.SGDR == True:
        opt.sched = CosineWithRestarts(opt.optimizer, T_max=opt.train_len)

    if opt.checkpoint > 0:
        print("model weights will be saved every %d minutes and at end of epoch to directory weights/"%(opt.checkpoint))
    
    if opt.load_weights is not None and opt.floyd is not None:
        os.mkdir('weights')
        pickle.dump(SRC, open('weights/SRC.pkl', 'wb'))
        pickle.dump(TRG, open('weights/TRG.pkl', 'wb'))
    
    train_model(model, opt)

    if opt.floyd is False:
        promptNextAction(model, opt, SRC, TRG)

def yesno(response):
    while True:
        if response != 'y' and response != 'n':
            response = input('command not recognised, enter y or n : ')
        else:
            return response

def promptNextAction(model, opt, SRC, TRG):

    saved_once = 1 if opt.load_weights is not None or opt.checkpoint > 0 else 0
    
    if opt.load_weights is not None:
        dst = opt.load_weights
    if opt.checkpoint > 0:
        dst = 'weights'

    while True:
        save = yesno(input('training complete, save results? [y/n] : '))
        if save == 'y':
            while True:
                if saved_once != 0:
                    res = yesno("save to same folder? [y/n] : ")
                    if res == 'y':
                        break
                dst = input('enter folder name to create for weights (no spaces) : ')
                if ' ' in dst or len(dst) < 1 or len(dst) > 30:
                    dst = input("name must not contain spaces and be between 1 and 30 characters length, enter again : ")
                else:
                    try:
                        os.mkdir(dst)
                    except:
                        res= yesno(input(dst + " already exists, use anyway? [y/n] : "))
                        if res == 'n':
                            continue
                    break
            
            print("saving weights to " + dst + "/...")
            torch.save(model.state_dict(), f'{dst}/model_weights')
            if saved_once == 0:
                pickle.dump(SRC, open(f'{dst}/SRC.pkl', 'wb'))
                pickle.dump(TRG, open(f'{dst}/TRG.pkl', 'wb'))
                saved_once = 1
            
            print("weights and field pickles saved to " + dst)

        res = yesno(input("train for more epochs? [y/n] : "))
        if res == 'y':
            while True:
                epochs = input("type number of epochs to train for : ")
                try:
                    epochs = int(epochs)
                except:
                    print("input not a number")
                    continue
                if epochs < 1:
                    print("epochs must be at least 1")
                    continue
                else:
                    break
            opt.epochs = epochs
            train_model(model, opt)
        else:
            print("exiting program...")
            break

    # for asking about further training use while true loop, and return
if __name__ == "__main__":
    main()
