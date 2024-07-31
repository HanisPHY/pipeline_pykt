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
            print("size of interaction_emb input is ", self.emb_size)
            self.interaction_emb = Embedding(self.num_c * 2 + 20, self.emb_size)

        self.lstm_layer = LSTM(self.emb_size + BERT_EMB_DIM, self.hidden_size, batch_first=True)
        self.dropout_layer = Dropout(dropout)
        self.out_layer = Linear(self.hidden_size, self.num_c)
        

    # model input: concept, response
    def forward(self, q, r, id, emb_dic, question):
        # print(f"q.shape is {q.shape}")
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
        bert_embeddings = torch.tensor(bert_embeddings, dtype=torch.float32).to(q.device)
        
        emb_type = self.emb_type
        if emb_type == "qid":
            x = (q + self.num_c * r).long()  # Ensure x contains integer indices
            print("q is ", q)
            print("x.shape is", x.shape)  # Check the shape of x
            print("x.min():", x.min().item(), "x.max():", x.max().item())  # Check the range of indices
            print("q.min(): ", q.min().item(), "q.max():", q.max().item())
            
            # Ensure the sequence length matches expected size
            if x.shape[1] != 200:
                print("Warning: Unexpected sequence length")

            # Ensure all indices are within the valid range
            print("self.num_c * 2 is ", self.num_c * 2)
            # assert x.min() >= 0 and x.max() < self.num_c * 2, "Indices are out of range for the embedding layer"

            # Pass the valid input tensor to the embedding layer
            xemb = self.interaction_emb(x)
    
        print(f"xemb.shape is {xemb.shape}, device: {xemb.device}")
        print(f"bert_embeddings.shape is {bert_embeddings.shape}, device: {bert_embeddings.device}")
        
        # batch_size, seqlen = q.shape
        # bert_embeddings = []
        # for i in range(batch_size):
        #     batch_emb = []
        #     for pid in q[i]:
        #         print(pid)
        #         if pid.item() == 0:
        #             emb = [0.0] * 768
        #         else:
        #             emb = emb_dic[(id[i].item(), pid.item())]
        #         batch_emb.append(emb)
        #     bert_embeddings.append(batch_emb)
        # bert_embeddings = torch.tensor(bert_embeddings, dtype=torch.float32).to(q.device)

        combined_emb = torch.cat([xemb, bert_embeddings], dim=-1)
        
        h, _ = self.lstm_layer(combined_emb)
        h = self.dropout_layer(h)
        y = self.out_layer(h)
        y = torch.sigmoid(y)

        return y