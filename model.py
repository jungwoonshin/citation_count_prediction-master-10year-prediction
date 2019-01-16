import torch.nn as nn
import torch

class basicLSTM(nn.Module): 
    def __init__(self,args):
        super(basicLSTM, self).__init__()
        self.rnn = nn.LSTM(1,100,1,batch_first=True) # dim_input, dim_hidden, num_layer
        self.regressor = nn.Sequential(nn.Linear(100, 10), nn.ReLU() ,nn.Linear(10,1))

    def forward(self, x, hidden):
        # print(x.size())
        output, (_, _) = self.rnn(x,hidden) # x = [1, seq, 1], output = [1, seq, 100]
        # print(output.size())
        # print(output[:,-1,:].size())
        output = self.regressor(output[:,-1,:].squeeze(0)) # output[:,-1,:] = [1 , 100]
        # print(output.size())
        return output # [1]
    
    def init_hidden(self,device,bsz=1):
        return (torch.zeros((1,bsz, 100),device=device),torch.zeros((1,bsz, 100),device=device))
