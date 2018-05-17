import layers
import numpy as np
import torch
from torch.autograd import Variable
import  torch.nn.functional as F
from torch import optim

hidden_dim=100
#Machine comprehension model that encapsulates all the components(encoders and answer predictors)
class MachineComprehension(torch.nn.Module):
    def __init__(self,c_maxlen,q_maxlen,embeddings):
        super(MachineComprehension,self).__init__()
        word_embeddings = torch.Tensor(embeddings)
        emb_dim = word_embeddings.size(1)
        vocab_size = word_embeddings.size(0)
        self.embedder = torch.nn.Embedding(vocab_size,emb_dim,padding_idx=0)
        
        self.embedder.weight = torch.nn.Parameter(word_embeddings)
        
        self.hidden_dim = emb_dim
        self.vocab_size = vocab_size
        self.c_maxlen = c_maxlen
        self.q_maxlen = q_maxlen
        

        self.passage_encoder = layers.ParagraphEncoder(self.hidden_dim,c_maxlen)
        self.question_encoder = layers.QuestionEncoder(self.hidden_dim,q_maxlen)
        self.soft_aligner = layers.SoftAlignment(self.hidden_dim)
        self.start_point_predictor = layers.StartEndPointPredictor(self.hidden_dim)
        self.end_point_predictor = layers.StartEndPointPredictor(self.hidden_dim)

    def forward(self,p_var,q_var,train=True):
        p_emb = self.embedder(p_var)
        q_emb = self.embedder(q_var)
        p_enc = self.passage_encoder(p_emb)
        q_enc,q_out = self.question_encoder(q_emb)
        query_aligned = self.soft_aligner(q_out,p_enc)
        
        p_enc = torch.cat([p_enc,query_aligned],dim=2)
        start_pos = self.start_point_predictor(p_enc,q_enc,train)
        end_pos = self.end_point_predictor(p_enc,q_enc,train)
        return start_pos,end_pos


