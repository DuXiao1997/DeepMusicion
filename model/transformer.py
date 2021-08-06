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
import math
import os
dtype = torch.FloatTensor
torch.manual_seed(1)

batch_size = 250
n_hidden = 128
acc_lis = []
loss_lis = []
word_len = 200
# transformer新增参数
d_model = 256
d_ff = 64 # FeedForward dimension
d_k = d_v = 64  # dimension of K(=Q), V
n_layers = 6  # number of Encoder of Decoder Layer
n_heads = 8  # number of heads in Multi-Head Attention
src_len = word_len
n_class = 4


def read_file():
    res = []
    target = []
    file_cla = ['快乐','悲伤','轻松']
    for cla in file_cla:
        filepath = 'data/lyrics_new_'+cla+'_0.2.txt'
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
    with open('data/lyrics_new_愤怒_0.2.txt',encoding='utf-8') as f:
        contents = f.readlines()
    # print(res)
    for con in contents:
        con = con.strip().replace('\n', '')
        res.append(con)
        target.append(3)
        # target.append(2)
    print(len(res))
    return res,target

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

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):

        super(PositionalEncoding, self).__init__()

        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)

        postion = torch.arange(0, max_len).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(postion * div_term)
        pe[:, 1::2] = torch.cos(postion * div_term)

        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)],
                         requires_grad=False)
        return self.dropout(x)

def make_position_encode():
    res,targets = read_file()
    inputs = make_batch_new(res)
    pe = PositionalEncoding(d_model,dropout=0.1,max_len=2000)
    enc_inputs = pe(inputs) + inputs
    return enc_inputs,targets

# enc_inputs = make_position_encode()
def attention(query, key, value, mask=None, dropout=None): # q=k=v=enc_inputs [batch_size,seq_len,256]

    d_k = query.size(-1) #256
    scores = torch.matmul(query, key.transpose(-2,-1) / math.sqrt(d_k)) #[batch_size,seq_len,seq_len] [1750,200,200]

    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    p_attn = F.softmax(scores, dim=-1) #[1750,200,200]

    if dropout is not None:
        p_attn = dropout(p_attn)

    return torch.matmul(p_attn, value), p_attn #[1750,200,256],[1750,200,200]

# enc_inputs = make_position_encode()
# attn, p_attn = attention(enc_inputs,enc_inputs,enc_inputs)
# print('attn shape',attn.shape)
# print('p_attn shape',p_attn.shape)

import copy

def clones(module, N):

    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def make_loader():
    bath_path = os.getcwd()
    # lyrics_path = os.path.join(bath_path, 'lyrics_text')
    # sentences = read_file()
    # input_batch = make_batch_new(sentences)
    input_batch,target_batch = make_position_encode()
    input_batch = Variable(torch.Tensor(input_batch))
    print('input-shape',input_batch.shape)
    target_batch = Variable(torch.LongTensor(target_batch))
    print('target-shape',target_batch.shape)
    torch_dataset = Data.TensorDataset(input_batch,target_batch)
    loader = Data.DataLoader(
        dataset = torch_dataset,
        batch_size = batch_size,
        shuffle=True,
        # num_workers=2,
    )
    return loader

class MultiHeadeAttention(nn.Module):

    def __init__(self, head, embedding_dim, dropout=0.1):

        super(MultiHeadeAttention, self).__init__()

        assert embedding_dim % head == 0

        self.d_k = embedding_dim // head

        self.head = head

        self.linears = clones(nn.Linear(embedding_dim, embedding_dim), 4)

        self.attn = None

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):

        if mask is not None:

            mask = mask.unsqueeze(0)

        batch_size = query.size(0)


        query, key, value = \
        [model(x).view(batch_size, -1, self.head, self.d_k).transpose(1, 2)
         for model, x in zip(self.linears, (query, key, value))]

        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.head * self.d_k)

        return self.linears[-1](x)

# query = key = value = enc_inputs
# mha = MultiHeadeAttention(head=8, embedding_dim=d_model, dropout=0.2)
# mha_result = mha(query, key, value)
# print(mha_result.shape)
# print(mha_result)

class PositionwiseFeedForward(nn.Module):

    def __init__(self, d_model, d_ff, dropout=0.1):

        super(PositionwiseFeedForward, self).__init__()

        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        return self.w2(self.dropout(F.relu(self.w1(x))))


# ff = PositionalwiseForward(d_model,d_ff,dropout=0.2)
# ff = PositionwiseFeedForward(d_model,d_ff,dropout=0.2)
# ff_result = ff(mha_result)
# print(ff_result.shape)

class LayerNorm(nn.Module):

    def __init__(self, features, eps = 1e6):

        super(LayerNorm, self).__init__()

        self.a2 = nn.Parameter(torch.ones(features))
        self.b2 = nn.Parameter(torch.zeros(features))

        self.eps = eps

    def forward(self, x):

        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a2 * (x - mean) / (std + self.eps) + self.b2

# ln = LayerNorm(d_model)
# ln_result = ln(ff_result)
# print(ln_result.shape)

class SublayerConnection(nn.Module):
    def __init__(self, size, dropout=0.1):

        super(SublayerConnection, self).__init__()

        self.norm = LayerNorm(size)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, sublayer):

        return x + self.dropout(sublayer(self.norm(x)))

# enc_inputs = make_position_encode()
# attn, p_attn = attention(enc_inputs,enc_inputs,enc_inputs)
# query = key = value = enc_inputs
# mha = MultiHeadeAttention(head=8, embedding_dim=d_model)
# mha_result = mha(query, key, value)
# ff = PositionwiseFeedForward(d_model,d_ff,dropout=0.2)
# ff_result = ff(mha_result)
# ln = LayerNorm(d_model)
# ln_result = ln(ff_result)
# sublayer = lambda enc_inputs: mha(enc_inputs,enc_inputs,enc_inputs)
# sc = SublayerConnection(d_model,dropout=0.2)
# sc_result = sc(enc_inputs,sublayer)
# print(sc_result.shape)

class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):

        super(EncoderLayer, self).__init__()

        self.self_attn = self_attn
        self.feed_forward = feed_forward

        self.sublayer = clones(SublayerConnection(size, dropout), 2)

        self.size = size

    def forward(self, x, mask=None):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)

class Transformer(nn.Module):
    def __init__(self):
        super(Transformer,self).__init__()

        self.W = nn.Parameter(torch.randn([d_model, n_class]).type(dtype))
        self.b = nn.Parameter(torch.randn([n_class]).type(dtype))

    def forward(self, X):
        input = X.transpose(0, 1)[-1]
        model = torch.mm(input, self.W) + self.b  # model : [batch_size, n_class]
        # print('model_shape',model.shape)
        return model


# enc_inputs = make_position_encode()
# mask = Variable(torch.zeros(8,4,4))
# self_attn = MultiHeadeAttention(head=8,embedding_dim=d_model)
# ff = PositionwiseFeedForward(d_model,d_ff,dropout=0.2)
# el = EncoderLayer(d_model,self_attn,ff,dropout=0.2)
# el_result = el(enc_inputs)
# print(el_result.shape)

model = Transformer()
# result = model(el_result)
# print('result',result.shape)


# model = get_sinusoid_encoding_table()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

#
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


            # mask = Variable(torch.zeros(8, 4, 4))
            self_attn = MultiHeadeAttention(head=8, embedding_dim=d_model)
            ff = PositionwiseFeedForward(d_model, d_ff, dropout=0.2)
            el = EncoderLayer(d_model, self_attn, ff, dropout=0.2)
            el_result = el(input_batch)


            output = model(el_result)
            print('output shape',output.shape)
            # output = model(200,d_model)
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
            torch.save(model.state_dict(),'transformer.pkl')
            loss_sum_min = loss_sum
            print('save ',loss_sum_min)

train()
