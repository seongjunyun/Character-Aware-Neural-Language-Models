import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
from torch.autograd import Variable
import pickle
from data_utils import word_to_idx,to_char,to_var,txt_to_word,txt_to_word_total
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
num_epochs = 35

#load train set
data,num_batches = txt_to_word('data/train.txt',batch_size)

#train_label
label = word_to_idx(data,word_vocab)
label = label
label = torch.from_numpy(label)
label = label.view(batch_size,-1)

#train_input_data
to_char(data,char_vocab,max_len)
data = np.array(data)
data = torch.from_numpy(data)
data = data.view(batch_size,-1,max_len)

#load validation set
val_data,_ = txt_to_word('data/valid.txt',batch_size)
    
#val_label
val_label = word_to_idx(val_data,word_vocab)
val_label = val_label
val_label = torch.from_numpy(val_label)
val_label = val_label.view(batch_size,-1)

#val_input_data
to_char(val_data,char_vocab,max_len)
val_data = np.array(val_data)
val_data = torch.from_numpy(val_data)
val_data = val_data.view(batch_size,-1,max_len)




   
def validate(seq_len,val_data,val_label,model,h):
    
    
    val_loss = 0
    count = 0
    for j in range(0,val_data.size(1)-seq_len,seq_len):

        val_inputs = to_var(val_data[:,j:j+seq_len,:])
        val_targets = to_var(val_label[:,(j+1):(j+1)+seq_len]).contiguous()

        output,h = model(val_inputs,h)

        loss = criterion(output,val_targets.view(-1))
        val_loss+=loss.data[0]
        count+=1
    print ('Test  Loss: %.3f, Perplexity: %5.2f' %
            (val_loss/count, np.exp(val_loss/count)))
    
    return val_loss/count

best_ppl = 10000

final_ppl = 100

num_trial = 20

for trial in range(num_trial): 


    pivot=100000



    model=LM(word_vocab,char_vocab,max_len,embed_dim,out_channels,kernels,hidden_size,batch_size)



    if torch.cuda.is_available():
        model.cuda()

    
    learning_rate = 1.0

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',factor=0.5,patience=1,verbose=True)



    
    for epoch in range(num_epochs):

        hidden_state = (to_var(torch.zeros(2,batch_size,hidden_size)),to_var(torch.zeros(2,batch_size,hidden_size)))
        
        model.train(True)
        for i in range(0, data.size(1)-seq_len,seq_len):

            
            inputs = to_var(data[:,i:i+seq_len,:])
            targets = to_var(label[:,(i+1):(i+1)+seq_len]).contiguous()
            
            model.zero_grad()
            
            hidden_state = [state.detach() for state in hidden_state]
            
            
            
            output, hidden_state = model(inputs,hidden_state)
            
           
                     
            loss = criterion(output, targets.view(-1))

                    
            loss.backward()
            
            nn.utils.clip_grad_norm(model.parameters(),5)
            optimizer.step()
            
            step = (i+1) // seq_len
            if step % 100 == 0:
                    
                print ('Epoch [%d/%d], Step[%d/%d], Loss: %.3f, Perplexity: %5.2f' %
                    (epoch+1, num_epochs, step, num_batches//seq_len,
                    loss.data[0], np.exp(loss.data[0])))
        
        
        model.eval() 
        #validate
        val_loss = validate(seq_len,val_data,val_label,model,hidden_state)
        val_loss = np.exp(val_loss)
       
        
        if pivot-val_loss < 0.8  :
           
            if learning_rate > 0.03: 
                learning_rate = learning_rate * 0.5
                print(learning_rate)
                optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

        pivot = val_loss

        if val_loss < best_ppl:
            print(val_loss)
            best_ppl = val_loss
            # Save the Model
            torch.save(model.state_dict(), 'model.pkl')

            
    if best_ppl < final_ppl:
        
        print('best ppl')
        print(best_ppl)
        #Save the final_model
        torch.save(model.state_dict(),'model_best.pkl')
        final_ppl = best_ppl







