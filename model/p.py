import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import jieba
import torch.utils.data as Data
# from model.utils import TextRNN
from gensim.models import KeyedVectors,word2vec,Word2Vec
import jieba
import multiprocessing
import gensim
import os
dtype = torch.FloatTensor
torch.manual_seed(1)

batch_size = 70
# n_step = 2 # number of cells(= number of Step)
n_hidden = 128 # number of hidden units in one cell
n_class = 4
# filepath = '../data/train.txt'
target = []
word_len = 200


def read_lyrics():
    res = []
    # target = []
    file_cla = ['快乐','悲伤','宁静']
    for cla in file_cla:
        filepath = 'lyrics_text/测试集/lyrics_test_'+cla+'.txt'
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
    with open('lyrics_text/测试集/lyrics_test_愤怒.txt','r',encoding='utf-8') as f:
        contents = f.readlines()
    for con in contents:
        con = con.strip().replace('\n','')
        res.append(con)
        target.append(3)
    print(res)
    print(len(res))
    return res


def make_batch(sentences):
    """

    :param sentences: read_lyrics生成的句子列表
    :return: 转换为word2vec向量的torch shape:[162,90,256]
    """
    sen_vect = []
    eos_vec = [0 for _ in range(256)]
    model = gensim.models.Word2Vec.load('word2vec_opencc.model')
    for sen in sentences:
        sen = sen.split(' ')
        res = []
        for k in sen:
            if k == '' or k == ' ':
                continue
            try:
                res.append(model.wv[k].tolist())
            except:
                continue
        if len(res) < word_len:
            vec_num = len(res)
            for i in range(word_len-vec_num):
                res.append(eos_vec)
        elif len(res) > word_len:
            res = res[:word_len]
        sen_vect.append(res)

    return Variable(torch.Tensor(sen_vect))


# def make_batch_new(sentences):
#     """
#
#     :param sentences: read_lyrics生成的句子列表
#     :return: 转换为word2vec向量的torch shape:[162,90,256]
#     """
#     sen_vect = []
#     eos_vec = [0 for _ in range(256)]
#     model = gensim.models.Word2Vec.load('word2vec_opencc.model')
#     for sen in sentences:
#         sen = sen.split(' ')
#         for s in sen:
#             if '' in sen:
#                 sen.remove('')
#         if len(sen) > word_len:
#             sen = sen[:word_len]
#         else:
#             for i in range(word_len-len(sen)):
#                 sen.append(sen[i])
#         res = []
#         for k in sen:
#             try:
#                 res.append(model.wv[k].tolist())
#             except:
#                 res.append(eos_vec)
#         sen_vect.append(res)
#     # print('input_shape',sen_vect.shape)
#     return Variable(torch.Tensor(sen_vect))

class BiLSTM_Attention(nn.Module):
    def __init__(self):
        super(BiLSTM_Attention, self).__init__()
        self.lstm = nn.LSTM(input_size=256, hidden_size=n_hidden, bidirectional=True)
        self.out = nn.Linear(n_hidden * 2, n_class)

    # lstm_output : [batch_size, n_step, n_hidden * num_directions(=2)], F matrix
    def attention_net(self, lstm_output, final_state):
        hidden = final_state.view(-1, n_hidden * 2, 1)   # hidden : [batch_size, n_hidden * num_directions(=2), 1(=n_layer)]
        attn_weights = torch.bmm(lstm_output, hidden).squeeze(2) # attn_weights : [batch_size, n_step]
        soft_attn_weights = F.softmax(attn_weights, 1)
        # [batch_size, n_hidden * num_directions(=2), n_step] * [batch_size, n_step, 1] = [batch_size, n_hidden * num_directions(=2), 1]
        context = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)
        return context, soft_attn_weights.data.numpy() # context : [batch_size, n_hidden * num_directions(=2)]

    def forward(self, input):
        # input:[batch_size,n_step,256]
        input = input.permute(1, 0, 2) # input : [len_seq, batch_size, embedding_dim]

        hidden_state = Variable(torch.zeros(1*2, batch_size, n_hidden)) # [num_layers(=1) * num_directions(=2), batch_size, n_hidden]
        cell_state = Variable(torch.zeros(1*2, batch_size, n_hidden)) # [num_layers(=1) * num_directions(=2), batch_size, n_hidden]

        # final_hidden_state, final_cell_state : [num_layers(=1) * num_directions(=2), batch_size, n_hidden]
        output, (final_hidden_state, final_cell_state) = self.lstm(input, (hidden_state, cell_state))
        output = output.permute(1, 0, 2) # output : [batch_size, len_seq, n_hidden]
        attn_output, attention = self.attention_net(output, final_hidden_state)
        return self.out(attn_output), attention # model : [batch_size, num_classes], attention : [batch_size, n_step]


# bath_path = os.getcwd()
# lyrics_path = os.path.join(bath_path, 'lyrics_text')
# lyrics_lis = os.listdir(lyrics_path)
sentences = read_lyrics()
input_batch = make_batch(sentences)
target_batch = Variable(torch.LongTensor(target))


model = BiLSTM_Attention()
state_dict = torch.load('new_lyrics-attention-0.2.pkl')
model.load_state_dict(state_dict)

# print(model)
# hidden = Variable(torch.zeros(1, batch_size, n_hidden))

predict,_ = model(input_batch)
predict = predict.data.max(1,keepdim=True)[1]
target = torch.tensor(target)
predict = predict.reshape(1,-1)
print(predict)

sum = 0
prec_1, prec_2, prec_3, prec_4 = 0,0,0,0
for i in range(batch_size):
    if predict[0][i].equal(target[i]):
        sum += 1
    if target[i] == 0:
        if predict[0][i].equal(target[i]):
            prec_1 += 1
    elif target[i] == 1:
        if predict[0][i].equal(target[i]):
            prec_2 += 1
    elif target[i] == 2:
        if predict[0][i].equal(target[i]):
            prec_3 += 1
    else:
        if predict[0][i].equal(target[i]):
            prec_4 += 1

# print(sum)
print(sum/batch_size)
print('prec_1',prec_1/20)
print('prec_2',prec_2/20)
print('prec_3',prec_3/20)
print('prec_4',prec_4/10)
# _*_ coding: utf-8 _*_
#  __author__:duxiao1
#  2021/2/23
