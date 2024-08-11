# coding: utf-8
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from .utils import ut_mask

BERT_EMB_DIM = 768

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ATKT(nn.Module):
    def __init__(self, num_c, skill_dim, answer_dim, hidden_dim, attention_dim=80, epsilon=10, beta=0.2, dropout=0.2, emb_type="qid", emb_path="", fix=True):
        super(ATKT, self).__init__()
        print("Using local file")
        self.model_name = "atkt"
        self.fix = fix
        print(f"fix: {fix}")
        if self.fix == True:
            self.model_name = "atktfix"
        self.emb_type = emb_type
        self.skill_dim=skill_dim
        self.answer_dim=answer_dim
        self.hidden_dim = hidden_dim
        self.num_c = num_c
        self.epsilon = epsilon
        self.beta = beta
        
        self.dropout_layer = nn.Dropout(dropout)
        self.fc = nn.Linear(self.hidden_dim*2, self.num_c)
        self.sig = nn.Sigmoid()
        
        self.skill_emb = nn.Embedding(self.num_c+1, self.skill_dim)
        self.skill_emb.weight.data[-1]= 0
        
        self.answer_emb = nn.Embedding(2+1, self.answer_dim)
        self.answer_emb.weight.data[-1]= 0
        
        self.attention_dim = attention_dim
        self.mlp = nn.Linear(self.hidden_dim, self.attention_dim)
        self.similarity = nn.Linear(self.attention_dim, 1, bias=False)
        
        self.rnn = nn.LSTM(self.skill_dim+self.answer_dim+BERT_EMB_DIM, self.hidden_dim, batch_first=True)

    
    def attention_module(self, lstm_output):
        att_w = self.mlp(lstm_output)
        att_w = torch.tanh(att_w)
        att_w = self.similarity(att_w)

        if self.fix == True:
            attn_mask = ut_mask(lstm_output.shape[1])
            att_w = att_w.transpose(1,2).expand(lstm_output.shape[0], lstm_output.shape[1], lstm_output.shape[1]).clone()
            att_w = att_w.masked_fill_(attn_mask, float("-inf"))
            alphas = torch.nn.functional.softmax(att_w, dim=-1)
            attn_ouput = torch.bmm(alphas, lstm_output)
        else: # 原来的官方实现
            alphas=nn.Softmax(dim=1)(att_w)
            # print(f"alphas: {alphas.shape}")    
            attn_ouput = alphas*lstm_output # 整个seq的attn之和为1，计算前面的的时候，所有的attn都<<1，不会有问题？做的少的时候，历史作用小，做得多的时候，历史作用变大？

        attn_output_cum=torch.cumsum(attn_ouput, dim=1)
        attn_output_cum_1=attn_output_cum-attn_ouput

        final_output=torch.cat((attn_output_cum_1, lstm_output),2)
        # import sys
        # sys.exit()

        return final_output

    # skill: concept
    # response: answer
    def forward(self, skill, answer, id, emb_dic, question, perturbation=None):
        batch_size, seqlen = question.shape
        bert_embeddings = []
        for i in range(batch_size):
            batch_emb = []
            for pid in question[i]:
                if pid.item() == 0:
                    emb = [0.0] * 768
                else:
                    emb = emb_dic[(id[i].item(), pid.item())]
                batch_emb.append(emb)
            bert_embeddings.append(batch_emb)
        bert_embeddings = torch.tensor(bert_embeddings, dtype=torch.float32).to(skill.device)
        
        emb_type = self.emb_type
        r = answer
        
        skill_embedding=self.skill_emb(skill)
        answer_embedding=self.answer_emb(answer)
        
        skill_answer=torch.cat((skill_embedding,answer_embedding), 2)
        answer_skill=torch.cat((answer_embedding,skill_embedding), 2)
        
        answer=answer.unsqueeze(2).expand_as(skill_answer)
        
        skill_answer_embedding=torch.where(answer==1, skill_answer, answer_skill)
        
        # skill_answer_embedding1=skill_answer_embedding
        if  perturbation is not None:
            skill_answer_embedding += perturbation
        
        combined_emb = torch.cat([skill_answer_embedding, bert_embeddings], dim=-1)    
        out,_ = self.rnn(combined_emb)
        out=self.attention_module(out)
        res = self.sig(self.fc(self.dropout_layer(out)))
        
        return res, skill_answer_embedding

from torch.autograd import Variable

def _l2_normalize_adv(d):
    if isinstance(d, Variable):
        d = d.data.cpu().numpy()
    elif isinstance(d, torch.FloatTensor) or isinstance(d, torch.cuda.FloatTensor):
        d = d.cpu().numpy()
    d /= (np.sqrt(np.sum(d ** 2, axis=(1, 2))).reshape((-1, 1, 1)) + 1e-16)
    return torch.from_numpy(d)
