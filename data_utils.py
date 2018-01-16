import torch
import numpy as np
from torch.autograd import Variable

def word_to_idx(words,word_vocab):

    idx = []
    for word in words:
        
        idx += [word_vocab[word]]
    
    return np.array(idx)

def to_char(x,char_vocab,max_len):

    
    for i, word in enumerate(x):
            chars  = [char_vocab[c] for c in list(word)]
            chars.insert(0,char_vocab['<start>'])
            chars.append(char_vocab['<end>'])
            for k in range(0,max_len-len(chars)):
                chars.append(0)

            x[i] = chars

def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
        return Variable(x, volatile=volatile)


def txt_to_word(path,batch_size):
    
    data = []

    with open(path,'r') as f:
        for line in f:
            words = line.split() + ['+']      
            data+= words

    total_len = len(data)
    num_batches = total_len//batch_size

    data = data[:num_batches*batch_size]

    return data,num_batches

def txt_to_word_total(path1,path2,batch_size):

    data = []

    with open(path1,'r') as f:
        for line in f:
            words = line.split() + ['+']
            data+= words

    with open(path2,'r') as f:
        for line in f:
            words = line.split() + ['+']
            data+= words

    total_len = len(data)
    num_batches = total_len//batch_size

    data = data[:num_batches*batch_size]

    return data,num_batches

