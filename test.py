import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
from torch.autograd import Variable
import pickle
from data_utils import word_to_idx,to_char,to_var,txt_to_word
from model import HighwayNetwork,LM

dic = pickle.load(open('data/dic.pickle','rb'))
word_vocab = dic['word_vocab']
char_vocab = dic['char_vocab']


#Hyper Parameters
batch_size = 20
max_len = dic['max_len']+2
embed_dim = 15
kernels = [1,2,3,4,5,6]
out_channels = 25
seq_len = 35
hidden_size = 500
learning_rate = 1.0
num_epochs = 25

#load test set
test_data,_ = txt_to_word('data/test.txt',batch_size)
    
#test_label
test_label = word_to_idx(test_data,word_vocab)
test_label = test_label
test_label = torch.from_numpy(test_label)
test_label = test_label.view(batch_size,-1)

#test_input_data
to_char(test_data,char_vocab,max_len)
test_data = np.array(test_data)
test_data = torch.from_numpy(test_data)
test_data = test_data.view(batch_size,-1,max_len)

   
def eval(seq_len,val_data,val_label,model,h):
    
    model.eval()
    val_loss = 0
    count = 0
    for j in range(0,val_data.size(1)-seq_len,seq_len):

        val_inputs = to_var(val_data[:,j:j+seq_len,:])
        val_targets = to_var(val_label[:,(j+1):(j+1)+seq_len])

        model.zero_grad()

        h = [state.detach() for state in h]

        output,h = model(val_inputs,h)
        loss = criterion(output,val_targets.view(-1))
        val_loss+=loss.data[0]
        count+=1
    print ('Test  Loss: %.3f, Perplexity: %5.2f' %
            (val_loss/count, np.exp(val_loss/count)))
    
    return val_loss/count




model=LM(word_vocab,char_vocab,max_len,embed_dim,out_channels,kernels,hidden_size,batch_size)



if torch.cuda.is_available():
    model.cuda()


model.load_state_dict(torch.load('model.pkl'))

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',factor=0.5,patience=1,verbose=True)

hidden_state = (to_var(torch.zeros(2,batch_size,hidden_size)),to_var(torch.zeros(2,batch_size,hidden_size)))

       
#validate
test_loss = eval(seq_len,test_data,test_label,model,hidden_state)
test_loss = np.exp(test_loss)
            
                  







