import os

import numpy as np
import torch

from torch.nn import Module, Embedding, LSTM, Linear, Dropout
BERT_EMB_DIM = 768

class DKT(Module):
    def __init__(self, num_c, emb_size, dropout=0.1, emb_type='qid', emb_path="", pretrain_dim=768):
        super().__init__()
        self.model_name = "dkt"
        self.num_c = num_c
        self.emb_size = emb_size
        self.hidden_size = emb_size
        self.emb_type = emb_type

        if emb_type.startswith("qid"):
            self.interaction_emb = Embedding(self.num_c * 2, self.emb_size)

        self.lstm_layer = LSTM(self.emb_size + BERT_EMB_DIM, self.hidden_size, batch_first=True)
        self.dropout_layer = Dropout(dropout)
        self.out_layer = Linear(self.hidden_size, self.num_c)
        

    # model input: concept, response
    def forward(self, q, r, id, emb_dic):
        # print(f"q.shape is {q.shape}")
        emb_type = self.emb_type
        if emb_type == "qid":
            x = q + self.num_c * r
            xemb = self.interaction_emb(x)
        # print(f"xemb.shape is {xemb.shape}")
        
        batch_size, seqlen = q.shape
        bert_embeddings = []
        for i in range(batch_size):
            batch_emb = []
            for pid in q[i]:
                if pid.item() == 0:
                    emb = [0.0] * 768
                else:
                    emb = emb_dic[(id[i].item(), pid.item())]
                batch_emb.append(emb)
            bert_embeddings.append(batch_emb)
        bert_embeddings = torch.tensor(bert_embeddings, dtype=torch.float32).to(q.device)

        combined_emb = torch.cat([xemb, bert_embeddings], dim=-1)
        
        h, _ = self.lstm_layer(combined_emb)
        h = self.dropout_layer(h)
        y = self.out_layer(h)
        y = torch.sigmoid(y)

        return y