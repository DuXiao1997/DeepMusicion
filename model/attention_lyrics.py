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
batch_size = 250
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
        filepath = 'lyrics_text/lyrics_new_'+cla+'_0.2.txt'
        with open(filepath,'r',encoding='utf-8') as f:
            contents = f.readlines()
        for con in contents:
            if cla == '快乐':
                target.append(0)
            elif cla == '悲伤':
                target.append(1)
            elif cla == '轻松':
                target.append(2)
            con = con.strip().replace('\n','')
            res.append(con)
    with open('lyrics_text/lyrics_new_愤怒_0.2.txt',encoding='utf-8') as f:
        contents = f.readlines()
    # print(res)
    for con in contents:
        con = con.strip().replace('\n', '')
        res.append(con)
        target.append(3)
        # target.append(2)
    print(len(res))
    return res

# def make_batch(sentences):
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
#         res = []
#         for k in sen:
#             try:
#                 res.append(model.wv[k].tolist())
#             except:
#                 res.append(eos_vec)
#         if len(res) < word_len:
#             vec_num = len(res)
#             for i in range(word_len-vec_num):
#                 res.append(eos_vec)
#         elif len(res) > word_len:
#             res = res[:word_len]
#         sen_vect.append(res)
#     # print('input_shape',sen_vect.shape)
#     return Variable(torch.Tensor(sen_vect))

def make_batch_new(sentences):
    """

    :param sentences: read_lyrics生成的句子列表
    :return: 转换为word2vec向量的torch shape:[162,90,256]
    """
    sen_vect = []
    eos_vec = [0 for _ in range(256)]
    model = gensim.models.Word2Vec.load('word2vec_opencc.model')
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
#
# def make_valid_data():
#     res_valid = []
#     target_valid = []
#     file_cla = ['快乐', '悲伤', '轻松']
#     for cla in file_cla:
#         filepath = 'lyrics_text/lyrics_' + cla + '.txt'
#         with open(filepath, 'r', encoding='utf-8') as f:
#             contents = f.readlines()
#         for con in contents[-10:]:
#             if cla == '快乐':
#                 target_valid.append(0)
#             elif cla == '悲伤':
#                 target_valid.append(1)
#             elif cla == '轻松':
#                 target_valid.append(2)
#             con = con.strip().replace('\n', '')
#             res_valid.append(con)
#     with open('lyrics_text/lyrics_愤怒.txt', encoding='utf-8') as f:
#         contents = f.readlines()
#     # print(res)
#     for con in contents[-10:]:
#         con = con.strip().replace('\n', '')
#         res_valid.append(con)
#         target_valid.append(3)
#         # target.append(2)
#     res_valid = make_batch_new(res_valid)
#     return res_valid,target_valid

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
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


def train():
    loader = make_loader()
    loss_sum_min = 10000
    for epoch in range(200):
        loss_sum = 0
        print('-------------------------' + 'epoch:' + str(epoch) + '------------------------------')
        sum_acc = 0
        for i, data in enumerate(loader):
            input_batch, target_batch = data
            # print('input size',input_batch.shape)
            optimizer.zero_grad()
            output, attention = model(input_batch)
            output_pre = output.data.max(1, keepdim=True)[1].reshape(1, -1)
            output_target = torch.tensor(target_batch)
            for j in range(batch_size):
                if output_pre[0][j].equal(output_target[j]):
                    sum_acc += 1
            # print('output size',output.shape)
            loss = criterion(output, target_batch)
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