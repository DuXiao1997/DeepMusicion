import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import jieba
import torch.utils.data as Data

from gensim.models import KeyedVectors,word2vec,Word2Vec
import jieba
import multiprocessing
import gensim
import os
dtype = torch.FloatTensor
torch.manual_seed(1)

target = []
n_class = 4
batch_size = 40
n_hidden = 128

# def read_lyrics(path,pat):
#     """
#     :param path: 文件夹的地址：快乐文件夹或悲伤文件夹
#     :return: 分词后的歌词列表，每个文件里的歌词是列表的一个值
#     """
#     res = []
#     for i in range(1,81):
#         filename = str(i)+'.txt'
#         file_name = os.path.join(path,filename)
#         # print(file_name)
#         lyric = ''
#         if os.path.getsize(file_name)==0:
#             continue
#         target.append(pat)
#         with open(file_name,encoding='utf-8') as f:
#             contents = f.readlines()
#         for con in contents:
#             lyric += con.strip()
#         lyric = lyric.replace('\t', '')
#         lyric = lyric.replace('\n','')
#         lyric = lyric.replace(' ','')
#         lyric = lyric.replace(',', '')
#         lyric = lyric.replace('。', '')
#         res.append(' '.join(jieba.cut(lyric)))
#     # print(res)
#     return res

def read_file():
    res = []
    # target = []
    file_cla = ['快乐','悲伤','轻松']
    for cla in file_cla:
        filepath = 'data/lyrics_'+cla+'.txt'
        with open(filepath,'r',encoding='utf-8') as f:
            contents = f.readlines()[:80]
        for con in contents:
            if cla == '快乐':
                target.append(0)
            elif cla == '悲伤':
                target.append(1)
            elif cla == '轻松':
                target.append(2)
            con = con.strip().replace('\n','')
            res.append(con)
    with open('data/lyrics_愤怒.txt',encoding='utf-8') as f:
        contents = f.readlines()[:40]
    # print(res)
    for con in contents:
        con = con.strip().replace('\n', '')
        res.append(con)
        target.append(3)
        # target.append(2)

    print(len(res))
    print(target)
    return res

def make_batch(sentences):
    """

    :param sentences: read_lyrics生成的句子列表
    :return: 转换为word2vec向量的torch shape:[162,90,256]
    """
    sen_vect = []
    eos_vec = [0 for _ in range(256)]
    model = gensim.models.Word2Vec.load('model/word2vec_opencc.model')
    for sen in sentences:
        sen = sen.split(' ')
        # print(sen)
        # print(len(sen))
        res = []
        for k in sen:
            if k==' ' or k=='':
                continue
            try:
                res.append(model.wv[k].tolist())
            except:
                continue
        if len(res) < 80:
            vec_num = len(res)
            for i in range(80-vec_num):
                res.append(eos_vec)
        elif len(res) > 80:
            res = res[:80]
        sen_vect.append(res)
    # print('input_shape',sen_vect.shape)
    return Variable(torch.Tensor(sen_vect))

def make_loader():
    bath_path = os.getcwd()
    lyrics_path = os.path.join(bath_path, 'lyrics_text')
    sentences = read_file()
    input_batch = make_batch(sentences)
    # print(input_batch.shape)
    target_batch = Variable(torch.LongTensor(target))

    torch_dataset = Data.TensorDataset(input_batch,target_batch)
    loader = Data.DataLoader(
        dataset = torch_dataset,
        batch_size = batch_size,
        shuffle=True,
        # num_workers=2,
    )
    return loader


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

model = BiLSTM_Attention()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train():
    loader = make_loader()
    for epoch in range(500):
        print('-------------------------' + 'epoch:' + str(epoch) + '------------------------------')
        loss_sum = 0
        sum = 0
        for i,data in enumerate(loader):
            input_batch,target_batch = data
            # print(input_batch.shape)
            optimizer.zero_grad()
            output = model(input_batch)
            # print('output size',output.shape)
            loss = criterion(output, target_batch)
            loss_sum += loss
            loss.backward()
            optimizer.step()

        print('Epoch:', '%04d' % (epoch + 1),'cost =', '{:.6f}'.format(loss_sum/40))
        torch.save(model.state_dict(),'old_lyrics-textlstm-1000.pkl')

train()