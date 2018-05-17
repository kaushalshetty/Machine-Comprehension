import numpy as np
import torch
from torch.autograd import Variable
import  torch.nn.functional as F
from torch import optim

#Encodes the paragraph
class ParagraphEncoder(torch.nn.Module):
    def __init__(self,hidden_dim,c_maxlen):
        super(ParagraphEncoder,self).__init__()
        self.hidden_dim = hidden_dim
        self.c_enc = torch.nn.RNN(hidden_dim,hidden_dim,1,batch_first=True)
        self.maxlen = c_maxlen
            
    def init_hidden(self):
        return Variable(torch.zeros(1,1,self.hidden_dim))
    
    def forward(self,p):
        p_emb = p.view(1,self.maxlen,-1)
        #print(p_emb)
        output,self.hidden_state = self.c_enc(p_emb,self.hidden_state)
        return output
 
#Encodes the question 
class QuestionEncoder(torch.nn.Module):
    def __init__(self,hidden_dim,q_maxlen):
        super(QuestionEncoder,self).__init__()
        self.hidden_dim = hidden_dim
        self.q_enc = torch.nn.RNN(hidden_dim,hidden_dim,1,batch_first=True)
        self.maxlen = q_maxlen
        self.linear = torch.nn.Linear(hidden_dim,1)
        
    def init_hidden(self):
        return Variable(torch.zeros(1,1,self.hidden_dim))
    
    def forward(self,q):
        q_emb = q.view(1,self.maxlen,-1)
        output,self.hidden_state = self.q_enc(q_emb,self.hidden_state)
        
        lin_out = self.linear(output)
        
        b = F.softmax(lin_out,dim=1)
        b = b.transpose(1,2)
        q_out = b@output
        
        
        return q_out,output
        
        
#SOFT ATTENTION IS VERY IMPORATANT FOR ACCURACY
class SoftAlignment(torch.nn.Module):
    def __init__(self,hidden_dim):
        super(SoftAlignment,self).__init__()
        self.hidden_dim = hidden_dim
        self.linear = torch.nn.Linear(hidden_dim,hidden_dim)
        
    def forward(self,q,p):
        q_lin_out = F.relu(self.linear(q))
        p_lin_out = F.relu(self.linear(p))
        q_lin_out_trans = q_lin_out.transpose(1,2)
        soft_attention = p_lin_out@q_lin_out_trans
        att_distribution = F.softmax(soft_attention,dim=2)
        context_aware_query = att_distribution @ q_lin_out
        return context_aware_query
        
            
#Network that predicts the start/end position       
class StartEndPointPredictor(torch.nn.Module):
    def __init__(self,hidden_dim):
        super(StartEndPointPredictor,self).__init__()
        self.start_p = torch.nn.Linear(hidden_dim,hidden_dim*2)
    
    def forward(self,p,q,train=True):
        q_lin_s = self.start_p(q)
        
        if train:
            
            return F.log_softmax(p@q_lin_s.transpose(1,2),dim=1)
        else:
            return F.softmax(p@q_lin_s.transpose(1,2),dim=1)
        
