import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
from torch.autograd import Variable

class HighwayNetwork(nn.Module):

    def __init__(self, input_size,activation='ReLU'):

       super(HighwayNetwork, self).__init__()
       #transform gate
       self.trans_gate = nn.Sequential(
                    nn.Linear(input_size,input_size),
                    nn.Sigmoid())
       #highway
       if activation== 'ReLU':
           self.activation = nn.ReLU()

       self.h_layer = nn.Sequential(
                           nn.Linear(input_size,input_size),
                           self.activation)
       #self.trans_gate[0].weight.data.uniform_(-0.05,0.05)
       #self.h_layer[0].weight.data.uniform_(-0.05,0.05)
       self.trans_gate[0].bias.data.fill_(-2)
       #self.h_layer[0].bias.data.fill_(0)

    def forward(self,x):

        t = self.trans_gate(x)
        h = self.h_layer(x)

        z = torch.mul(t,h)+torch.mul(1-t,x)

        return z

class LM(nn.Module):

    def __init__(self,word_vocab,char_vocab,max_len,embed_dim,out_channels,kernels,hidden_size,batch_size):

        super(LM, self).__init__()
        self.word_vocab = word_vocab
        self.char_vocab = char_vocab
        #Embedding layer
        self.embed = nn.Embedding(len(char_vocab)+1, embed_dim,padding_idx=0)
        #CNN layer
        self.cnns = []
        for kernel in kernels:
            self.cnns.append(nn.Sequential(
                    nn.Conv2d(1,out_channels,kernel_size=(kernel,embed_dim)),
                    nn.Tanh(),
                    nn.MaxPool2d((max_len-kernel+1,1))))

        self.cnns = nn.ModuleList(self.cnns)

        #highway layer
        self.highway = HighwayNetwork(out_channels*len(kernels))
        self.highway2 = HighwayNetwork(out_channels*len(kernels))

        #lstm layer
        self.lstm = nn.LSTM(out_channels*len(kernels),hidden_size,2,batch_first=True,dropout=0.5)

        #output layer
        self.linear = nn.Sequential(
                            nn.Dropout(0.5),
                            nn.Linear(hidden_size,len(word_vocab)))

        #self.init_weight()

    def init_weight(self):
        #self.embed.weight.data.uniform_(-0.05,0.05)
        for cnn in self.cnns:
            cnn[0].weight.data.uniform_(-0.05,0.05)
            cnn[0].bias.data.fill_(0)       
        self.linear[1].weight.data.uniform_(-0.05,0.05)
        self.linear[1].bias.data.fill_(0)
        self.lstm.weight_hh_l0.data.uniform_(-0.05,0.05)
        self.lstm.weight_hh_l1.data.uniform_(-0.05,0.05)
        self.lstm.weight_ih_l0.data.uniform_(-0.05,0.05)
        self.lstm.weight_ih_l1.data.uniform_(-0.05,0.05)
        self.lstm.bias_hh_l0.data.fill_(0)
        self.lstm.bias_hh_l1.data.fill_(0)
        self.lstm.bias_ih_l0.data.fill_(0)
        self.lstm.bias_ih_l1.data.fill_(0)

    def forward(self,x,h):

        batch_size = x.shape[0]
        seq_len = x.shape[1]

        
        x = x.contiguous().view(-1,x.shape[2])
        
        x = self.embed(x)

        x = x.contiguous().view(x.shape[0],1,x.shape[1],x.shape[2])
        
        y = [cnn(x).squeeze() for cnn in self.cnns]     
        w = torch.cat(y,1)
        w = self.highway(w)
        w = self.highway2(w)

        w = w.contiguous().view(batch_size,seq_len,-1)

        out, h = self.lstm(w,h)

        out = out.contiguous().view(batch_size*seq_len,-1)

        out = self.linear(out)

        
        return out,h

