import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import jieba
import torch.utils.data as Data
from matplotlib import pyplot

from gensim.models import KeyedVectors,word2vec,Word2Vec
import jieba
import multiprocessing
import gensim
import os
dtype = torch.FloatTensor
torch.manual_seed(1)

target = []
n_class = 4
batch_size = 50
n_hidden = 128
# valid_batch_size = 73
acc_lis = []
loss_lis = []
word_len = 200


def read_file():
    res = []
    # target = []
    file_cla = ['快乐','悲伤','轻松']
    for cla in file_cla:
        # filepath = 'lyrics_text/lyrics_new_'+cla+'_0.2.txt'
        filepath = 'data/lyrics_' + cla + '.txt'
        with open(filepath,'r',encoding='utf-8') as f:
            contents = f.readlines()
        for con in contents:
            if cla == '快乐':
                target.append(0)
            elif cla == '悲伤':
                target.append(1)
            elif cla == '宁静':
                target.append(2)
            con = con.strip().replace('\n','')
            res.append(con)
    with open('data/lyrics_new_愤怒_0.2.txt',encoding='utf-8') as f:
        contents = f.readlines()
    # print(res)
    for con in contents:
        con = con.strip().replace('\n', '')
        res.append(con)
        target.append(3)
        # target.append(2)
    print(len(res))
    return res



def make_batch_new(sentences):
    """

    :param sentences: read_lyrics生成的句子列表
    :return: 转换为word2vec向量的torch shape:[162,90,256]
    """
    sen_vect = []
    eos_vec = [0 for _ in range(256)]
    model = gensim.models.Word2Vec.load('model/word2vec_opencc.model')
    for sen in sentences:
        sen = sen.split(' ')
        for s in sen:
            if '' in sen:
                sen.remove('')
        if len(sen) > word_len:
            sen = sen[:word_len]
        else:
            for i in range(word_len-len(sen)):
                sen.append(sen[i])
        res = []
        for k in sen:
            try:
                res.append(model.wv[k].tolist())
            except:
                res.append(eos_vec)
        sen_vect.append(res)
    # print('input_shape',sen_vect.shape)
    return Variable(torch.Tensor(sen_vect))

def make_loader():
    bath_path = os.getcwd()
    lyrics_path = os.path.join(bath_path, 'lyrics_text')
    sentences = read_file()
    input_batch = make_batch_new(sentences)
    print(input_batch.shape)
    target_batch = Variable(torch.LongTensor(target))

    torch_dataset = Data.TensorDataset(input_batch,target_batch)
    loader = Data.DataLoader(
        dataset = torch_dataset,
        batch_size = batch_size,
        shuffle=True,
        # num_workers=2,
    )
    return loader


import torch
from torch import nn
from torch.nn import functional as F
import time


class FocalLoss(nn.Module):
    def __init__(self, alpha,gamma=2.5, num_classes = 4, size_average=True):
        """
        focal_loss损失函数, -α(1-yi)**γ *ce_loss(xi,yi)
        步骤详细的实现了 focal_loss损失函数.
        :param alpha:   阿尔法α,类别权重.      当α是列表时,为各类别权重,当α为常数时,类别权重为[α, 1-α, 1-α, ....],常用于 目标检测算法中抑制背景类 , retainnet中设置为0.25
        :param gamma:   伽马γ,难易样本调节参数. retainnet中设置为2
        :param num_classes:     类别数量
        :param size_average:    损失计算方式,默认取均值
        """
        super(FocalLoss,self).__init__()
        self.size_average = size_average
        if isinstance(alpha,list):
            assert len(alpha)==num_classes   # α可以以list方式输入,size:[num_classes] 用于对不同类别精细地赋予权重
            # print(" --- Focal_loss alpha = {}, 将对每一类权重进行精细化赋值 --- ".format(alpha))
            self.alpha = torch.Tensor(alpha)
            print('alpha', self.alpha)
        else:
            assert alpha<1   #如果α为一个常数,则降低第一类的影响,在目标检测中为第一类
            # print(" --- Focal_loss alpha = {} ,将对背景类进行衰减,请在目标检测任务中使用 --- ".format(alpha))
            self.alpha = torch.zeros(num_classes)
            self.alpha[0] += alpha
            self.alpha[1:] += (1-alpha) # α 最终为 [ α, 1-α, 1-α, 1-α, 1-α, ...] size:[num_classes]

        self.gamma = gamma

    def forward(self, preds, labels):
        """
        focal_loss损失计算
        :param preds:   预测类别. size:[B,N,C] or [B,C]    分别对应与检测与分类任务, B 批次, N检测框数, C类别数
        :param labels:  实际类别. size:[B,N] or [B]
        :return:
        """
        # assert preds.dim()==2 and labels.dim()==1
        preds = preds.view(-1,preds.size(-1))
        self.alpha = self.alpha.to(preds.device)
        preds_logsoft = F.log_softmax(preds, dim=1) # log_softmax
        preds_softmax = torch.exp(preds_logsoft)    # softmax

        preds_softmax = preds_softmax.gather(1,labels.view(-1,1))   # 这部分实现nll_loss ( crossempty = log_softmax + nll )
        preds_logsoft = preds_logsoft.gather(1,labels.view(-1,1))
        self.alpha = self.alpha.gather(0,labels.view(-1))
        loss = -torch.mul(torch.pow((1-preds_softmax), self.gamma), preds_logsoft)  # torch.pow((1-preds_softmax), self.gamma) 为focal loss中 (1-pt)**γ

        loss = torch.mul(self.alpha, loss.t())
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss




class BiLSTM_Attention(nn.Module):
    def __init__(self):
        super(BiLSTM_Attention, self).__init__()
        self.lstm = nn.LSTM(input_size=256, hidden_size=n_hidden, bidirectional=True)
        self.out = nn.Linear(n_hidden * 2, n_class)

    # lstm_output : [batch_size, n_step, n_hidden * num_directions(=2)], F matrix
    def attention_net(self, lstm_output, final_state):
        # tanh函数
        lstm_output = torch.tanh(lstm_output)
        hidden = final_state.view(-1, n_hidden * 2, 1)   # hidden : [batch_size, n_hidden * num_directions(=2), 1(=n_layer)] [40,128*2,1]
        attn_weights = torch.bmm(lstm_output, hidden).squeeze(2) # attn_weights : [batch_size, n_step] [40,200]
        soft_attn_weights = F.softmax(attn_weights, 1) # [40,200]
        # [batch_size, n_hidden * num_directions(=2), n_step] * [batch_size, n_step, 1] = [batch_size, n_hidden * num_directions(=2), 1]
        context = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2) # [40,256]
        return context, soft_attn_weights.data.numpy() # context : [batch_size, n_hidden * num_directions(=2)]

    def forward(self, input):
        # input:[batch_size,n_step,256]
        input = input.permute(1, 0, 2) # input : [len_seq, batch_size, embedding_dim] [200,40,256]

        hidden_state = Variable(torch.zeros(1*2, batch_size, n_hidden)) # [num_layers(=1) * num_directions(=2), batch_size, n_hidden] [2,40,128]
        cell_state = Variable(torch.zeros(1*2, batch_size, n_hidden)) # [num_layers(=1) * num_directions(=2), batch_size, n_hidden] [2,40,128]
        # final_hidden_state, final_cell_state : [num_layers(=1) * num_directions(=2), batch_size, n_hidden]
        output, (final_hidden_state, final_cell_state) = self.lstm(input, (hidden_state, cell_state))
        output = output.permute(1, 0, 2) # output : [batch_size, len_seq, n_hidden]
        attn_output, attention = self.attention_net(output, final_hidden_state)
        # attn_output: [40,256] attention: [40,200]
        return self.out(attn_output), attention # model : [batch_size, num_classes], attention : [batch_size, n_step]

model = BiLSTM_Attention()
# criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


def train():
    loader = make_loader()
    loss_sum_min = 10000
    for epoch in range(200):
        loss_sum = 0
        print('-------------------------' + 'epoch:' + str(epoch) + '------------------------------')
        sum_acc = 0
        for i, data in enumerate(loader):
            prec_1, prec_2, prec_3, prec_4 = 0, 0, 0, 0
            sum_1, sum_2, sum_3, sum_4 = 0, 0, 0, 0
            input_batch, target_batch = data
            # print('input size',input_batch.shape)
            optimizer.zero_grad()
            output, attention = model(input_batch)
            output_pre = output.data.max(1, keepdim=True)[1].reshape(1, -1)
            output_target = torch.tensor(target_batch)
            for j in range(batch_size):
                if output_pre[0][j].equal(output_target[j]):
                    sum_acc += 1
                if output_target[j] == 0:
                    if output_pre[0][j].equal(output_target[j]):
                        prec_1 += 1
                elif output_target[j] == 1:
                    if output_pre[0][j].equal(output_target[j]):
                        prec_2 += 1
                elif output_target[j] == 2:
                    if output_pre[0][j].equal(output_target[j]):
                        prec_3 += 1
                else:
                    if output_pre[0][j].equal(output_target[j]):
                        prec_4 += 1
            # print('output size',output.shape)
            # loss = criterion(output, target_batch)
            # print('sum',prec_1/sum_1,prec_2/sum_2,prec_3/sum_3,prec_4/sum_4)
            try:
                alpha_1 = 1-(prec_1/100)
                alpha_2 = 1-(prec_2/100)
                alpha_3 = 1-(prec_3/100)
                alpha_4 = 1-(prec_4/100)
            except:
                alpha_1, alpha_2, alpha_3, alpha_4 = 0.25, 0.25, 0.25, 0.25
            # alpha_list = F.softmax(torch.tensor([alpha_1,alpha_2,alpha_3,alpha_4]),dim=0).tolist()
            # print(alpha_list)
            alpha_list = [0.15,0.25,0.25,0.35]
            focal_loss = FocalLoss(alpha=0.25)
            # focal_loss = FocalLoss(alpha=alpha_list.tolist())
            loss = focal_loss(output, target_batch)
            # print(loss)
            loss_sum += loss
            loss.backward()
            optimizer.step()

        acc = sum_acc/1750
        print('sum_acc',sum_acc)
        # print('batch_size',batch_size)
        if epoch % 10 == 0:
            acc_lis.append(acc)

        if epoch%10 == 0:
            loss_lis.append(loss_sum/batch_size)

        print('Epoch:', '%04d' % (epoch + 1),'cost =', '{:.6f}'.format(loss_sum/1750/batch_size))
        print('Accuracy = {:.6f}'.format(acc))
        if loss_sum<loss_sum_min:
            torch.save(model.state_dict(),'new_lyrics-attention-0.2.pkl')
            loss_sum_min = loss_sum
            print('save ',loss_sum_min)

train()
# x = [i*10 for i in range(20)]
# pyplot.plot(x, loss_lis)
# # pyplot.xlabel('epoch')
# # pyplot.ylabel('loss')
# # pyplot.show()
# pyplot.plot(x,acc_lis)
# pyplot.xlabel('epoch')
# pyplot.ylabel('acc')
# pyplot.show()