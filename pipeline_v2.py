import logging
import json
from transformers import logging as transformers_logging
transformers_logging.set_verbosity_error()

from torch.nn.functional import one_hot

from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.data._utils.collate import default_collate
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import os, random, time, shutil, warnings
from sklearn.metrics import accuracy_score, f1_score
from model.akt import AKT
from model.dkt import DKT
from model.atkt import ATKT
from model.saint import SAINT
from model.dkvmn import DKVMN
from model.simplekt import simpleKT

from torch.nn.functional import binary_cross_entropy

from torch.autograd import Variable, grad
from model.atkt import _l2_normalize_adv

import argparse

ATKT_SKILL_DIM=256
ATKT_ANSWER_DIM=96
ATKT_HIDDEN_DIM=80
SEQ_LEN = 200
DROPOUT=0.2
NUM_ATTN_HEADS=8
DKVMN_DIM_S=200
DKVMN_SIZE_M=50

def remove_folder(folder_path):
    try:
        shutil.rmtree(folder_path)
        print(f"Folder '{folder_path}' and all its contents have been removed successfully.")
    except FileNotFoundError:
        print(f"The folder '{folder_path}' does not exist.")
    except Exception as e:
        print(f"An error occurred while removing the folder: {e}")

class CustomDataset(Dataset):
    def __init__(self, dataframe, emb_dict, max_length=20):
        self.dataframe = dataframe
        self.emb_dict = emb_dict
        self.max_length = max_length
        
        self.grouped = self.dataframe.groupby('student_id').apply(lambda x: x).reset_index(drop=True)

    def __len__(self):
        return len(self.grouped['student_id'].unique())

    # q_data: concept
    # response: students' response
    # pid_data: question_id
    # id_data: student_id
    # return seq_len(max_length) each time, then put all together and finally return (batch_size, seq_len)
    def __getitem__(self, idx):
        student_id = self.grouped['student_id'].unique()[idx]
        student_data = self.grouped[self.grouped['student_id'] == student_id]
        
        past_data = student_data[student_data["xy_type"] == "past"]
        future_data = student_data[student_data["xy_type"] == "future"]
        
        if len(past_data) == 0 or len(future_data) == 0:
            print(f"No valid data for student_id: {student_id}")
            return None
            
        # Process past data
        concept_past = past_data['unique_slide_id'].values
        response_past = past_data['correctness'].values
        question_past = past_data['unique_question_id'].values
        id_data_past = past_data['student_id'].values[0]
        uid_data_past = past_data['uid'].values
        
        # Padding
        pad_length_past = self.max_length - len(concept_past)
        mask_past = np.ones(self.max_length)
        if pad_length_past > 0:
            concept_past = np.pad(concept_past, (0, pad_length_past), 'constant', constant_values=0)
            response_past = np.pad(response_past, (0, pad_length_past), 'constant', constant_values=0)
            question_past = np.pad(question_past, (0, pad_length_past), 'constant', constant_values=0)
            uid_data_past = np.pad(uid_data_past, (0, pad_length_past), 'constant', constant_values='0')
            mask_past[len(concept_past)-pad_length_past:] = 0
        else:
            concept_past = concept_past[:self.max_length]
            response_past = response_past[:self.max_length]
            question_past = question_past[:self.max_length]
            uid_data_past = uid_data_past[:self.max_length]
        
        concept_past = torch.tensor(concept_past, dtype=torch.int)
        response_past = torch.tensor(response_past, dtype=torch.int)
        question_past = torch.tensor(question_past, dtype=torch.int)
        mask_past = torch.tensor(mask_past, dtype=torch.bool)
        
        embeddings = []
        for qid in question_past:
            if qid == -1 or qid.item() == 0:
                emb = [0.0] * 768  # Assuming the embedding dimension is 768
            else:
                emb = self.emb_dict.get((id_data_past, qid.item()))
                if emb is None:
                    raise ValueError(f"No embedding found for (id_data_past, question_id): ({id_data_past}, {qid.item()})")
            embeddings.append(emb)
        emb_data_past = torch.tensor(embeddings, dtype=torch.float32)    
        
        
        # Process future(shift) data
        concept_future = future_data['unique_slide_id'].values
        response_future = future_data['correctness'].values
        question_future = future_data['unique_question_id'].values
        id_data_future = future_data['student_id'].values[0]
        uid_data_future = future_data['uid'].values
        
        # Padding
        pad_length_future = self.max_length - len(concept_future)
        mask_future = np.ones(self.max_length)
        if pad_length_future > 0:
            concept_future = np.pad(concept_future, (0, pad_length_future), 'constant', constant_values=0)
            response_future = np.pad(response_future, (0, pad_length_future), 'constant', constant_values=0)
            question_future = np.pad(question_future, (0, pad_length_future), 'constant', constant_values=0)
            uid_data_future = np.pad(uid_data_future, (0, pad_length_future), 'constant', constant_values='0')
            mask_future[len(concept_future)-pad_length_future:] = 0
        else:
            concept_future = concept_future[:self.max_length]
            response_future = response_future[:self.max_length]
            question_future = question_future[:self.max_length]
            uid_data_future = uid_data_future[:self.max_length]
    
        concept_future = torch.tensor(concept_future, dtype=torch.int)
        response_future = torch.tensor(response_future, dtype=torch.int)
        question_future = torch.tensor(question_future, dtype=torch.int)
        mask_future = torch.tensor(mask_future, dtype=torch.bool)
        
        embeddings = []
        for qid in question_future:
            if qid == -1 or qid.item() == 0:
                emb = [0.0] * 768  # Assuming the embedding dimension is 768
            else:
                emb = self.emb_dict.get((id_data_future, qid.item()))
                if emb is None:
                    raise ValueError(f"No embedding found for (id_data_future, question_id): ({id_data_future}, {qid.item()})")
            embeddings.append(emb)
        emb_data_future = torch.tensor(embeddings, dtype=torch.float32)

        return {
            'concept_past': concept_past,
            'response_past': response_past,
            'question_past': question_past,
            'id_data_past': id_data_past,
            'uid_data_past': uid_data_past,
            'emb_data_past': emb_data_past,
            'mask_past': mask_past,
            'concept_future': concept_future,
            'response_future': response_future,
            'question_future': question_future,
            'id_data_future': id_data_future,
            'uid_data_future': uid_data_future,
            'emb_data_future': emb_data_future,
            'mask_future': mask_future
        }

class Experiment_Pipeline():
    def __init__(self, max_length, log_folder, dataset_raw_path, load_model_type, num_q, num_c, d_model, n_blocks, dropout, model_name, emb_size, input_type, lr=1e-5):
        self.max_length = max_length
        self.set_seed(4)

        self.dataframe_raw = pd.read_csv(dataset_raw_path)
        self.log_folder = log_folder
        self.load_model_type = load_model_type

        self.num_q = num_q
        self.num_c = num_c
        self.d_model = d_model
        self.n_blocks = n_blocks
        self.dropout = dropout
        self.emb_size = emb_size
        self.model_name = model_name
        self.input_type = input_type

        self.model_init(load_model_type, model_name, lr)

    def set_seed(self, seed_num):
        np.random.seed(seed_num)
        random.seed(seed_num)
        torch.manual_seed(seed_num)
        torch.cuda.manual_seed(seed_num)
        torch.cuda.manual_seed_all(seed_num)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        self.train_test_seed = seed_num

    def dataset_prepare(self, dataset_path_train, dataset_path_test, batch_size=64):
        dataframe_train = pd.read_csv(dataset_path_train)
        dataframe_test = pd.read_csv(dataset_path_test)
        
        emb_dict_train = {"type": "train"}
        emb_dict_test = {"type": "test"}

        for _, row in dataframe_train.iterrows():
            str_emb = row["embedding_bert"].strip('][').split(',')
            float_emb = [float(emb) for emb in str_emb]
            emb_dict_train[(int(row['student_id']), int(row['unique_question_id']))] = float_emb

        for _, row in dataframe_test.iterrows():
            str_emb = row["embedding_bert"].strip('][').split(',')
            float_emb = [float(emb) for emb in str_emb]
            emb_dict_test[(int(row['student_id']), int(row['unique_question_id']))] = float_emb

        self.emb_dict_train = emb_dict_train
        self.emb_dict_test = emb_dict_test
        
        unique_student_ids = dataframe_train["student_id"].unique()
        train_student_ids, val_student_ids = train_test_split(unique_student_ids, test_size=0.2, random_state=self.train_test_seed)
        
        train_data = dataframe_train[dataframe_train["student_id"].isin(train_student_ids)]
        val_data = dataframe_train[dataframe_train["student_id"].isin(val_student_ids)]

        train_dataset = CustomDataset(train_data, emb_dict_train, self.max_length)
        val_dataset = CustomDataset(val_data, emb_dict_train, self.max_length)
        test_dataset = CustomDataset(dataframe_test, emb_dict_test, self.max_length)

        self.train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, collate_fn=self.custom_collate_fn)
        self.val_dataloader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=self.custom_collate_fn)
        self.test_dataloader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=self.custom_collate_fn)

    def custom_collate_fn(self, batch):
        # Filter out empty dictionaries
        batch = [data for data in batch if data]
        
        uids_past = [item['uid_data_past'] for item in batch]
        uids_future = [item['uid_data_future'] for item in batch]
        
        for item in batch:
            del item['uid_data_past']
            del item['uid_data_future']
            
        batch = default_collate(batch)
        batch['uid_data_past'] = uids_past
        batch['uid_data_future'] = uids_future
        
        return batch

    def model_save(self, checkpoint, checkpoint_path):
        if os.path.exists(checkpoint_path): os.remove(checkpoint_path)
        torch.save(checkpoint, checkpoint_path)

    # num_c: n_question
    # num_q: n_pid
    def model_init(self, load_model_type, model_name, lr=1e-4):
        assert load_model_type in ['best', 'last', 'none']

        self.checkpoint_last_path = os.path.join(self.log_folder, 'model_last.pt')
        self.checkpoint_best_path = os.path.join(self.log_folder, 'model_best.pt')

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.device = torch.device('cpu')

        if load_model_type in ['best', 'last']:
            checkpoint_path = self.checkpoint_last_path if load_model_type == 'last' else self.checkpoint_best_path
            checkpoint = torch.load(checkpoint_path)
            match model_name:
                case "akt":
                    self.model = AKT(self.num_c, self.num_q, self.d_model, self.n_blocks, self.dropout).to(self.device)
                case "dkt":
                    self.model = DKT(self.num_c, self.emb_size).to(self.device)
                case "atkt":
                    self.model = ATKT(self.num_c, ATKT_SKILL_DIM, ATKT_ANSWER_DIM, ATKT_HIDDEN_DIM).to(self.device)
                case "dkvmn":
                    self.model = DKVMN(self.num_c, dim_s=200, size_m=50).to(self.device)
                case "simpltKT":
                    self.model = simpleKT(self.num_c, self.num_q, self.d_model, n_blocks=2, dropout=0.1).to(self.device)
                
                

            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer = optim.AdamW(self.model.parameters(), lr=lr)
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epoch_exist = checkpoint['epoch']
            self.train_losses = checkpoint['train_loss']
            self.val_losses = checkpoint['val_loss']
        else:
            match model_name:
                case "akt":
                    self.model = AKT(self.num_c, self.num_q, self.d_model, self.n_blocks, self.dropout).to(self.device)
                case "dkt":
                    self.model = DKT(self.num_c, self.emb_size).to(self.device)
                case "atkt":
                    self.model = ATKT(self.num_c, ATKT_SKILL_DIM, ATKT_ANSWER_DIM, ATKT_HIDDEN_DIM).to(self.device)
                case "dkvmn":
                    self.model = DKVMN(self.num_c, dim_s=DKVMN_DIM_S, size_m=DKVMN_SIZE_M).to(self.device)
                case "simpleKT":
                    self.model = simpleKT(self.num_c, self.num_q, self.d_model, n_blocks=2, dropout=0.1).to(self.device)
                    
            self.optimizer = optim.AdamW(self.model.parameters(), lr=lr)
            self.epoch_exist = 0
            self.train_losses = []
            self.val_losses = []

        self.criterion = nn.BCELoss()
    
    # sm: mask_past    
    def cal_loss(self, model, ys, r, rshft, mask_past, mask_future, preloss=[]):
        sm = mask_past
        model_name = model.model_name

        if model_name in ["atdkt", "simplekt", "bakt_time", "sparsekt"]:
            y = torch.masked_select(ys[0], mask_future)
            t = torch.masked_select(rshft, mask_future)
            # print(f"loss1: {y.shape}")
            loss1 = binary_cross_entropy(y.double(), t.double())

            if model.emb_type.find("predcurc") != -1:
                if model.emb_type.find("his") != -1:
                    loss = model.l1*loss1+model.l2*ys[1]+model.l3*ys[2]
                else:
                    loss = model.l1*loss1+model.l2*ys[1]
            elif model.emb_type.find("predhis") != -1:
                loss = model.l1*loss1+model.l2*ys[1]
            else:
                loss = loss1

        elif model_name in ["rkt","dimkt","dkt", "dkt_forget", "dkvmn","deep_irt", "kqn", "sakt", "saint", "atkt", "atktfix", "gkt", "skvmn", "hawkes"]:
            y = torch.masked_select(ys[0], mask_future)
            t = torch.masked_select(rshft, mask_future)
            loss = binary_cross_entropy(y.double(), t.double())
        elif model_name == "dkt+":
            y_curr = torch.masked_select(ys[1], mask_future)
            y_next = torch.masked_select(ys[0], mask_future)
            r_curr = torch.masked_select(r, mask_future)
            r_next = torch.masked_select(rshft, mask_future)
            loss = binary_cross_entropy(y_next.double(), r_next.double())

            loss_r = binary_cross_entropy(y_curr.double(), r_curr.double()) # if answered wrong for C in t-1, cur answer for C should be wrong too
            loss_w1 = torch.masked_select(torch.norm(ys[2][:, 1:] - ys[2][:, :-1], p=1, dim=-1), mask_future[:, 1:])
            loss_w1 = loss_w1.mean() / model.num_c
            loss_w2 = torch.masked_select(torch.norm(ys[2][:, 1:] - ys[2][:, :-1], p=2, dim=-1) ** 2, mask_future[:, 1:])
            loss_w2 = loss_w2.mean() / model.num_c

            loss = loss + model.lambda_r * loss_r + model.lambda_w1 * loss_w1 + model.lambda_w2 * loss_w2
        elif model_name in ["akt","folibikt", "akt_vector", "akt_norasch", "akt_mono", "akt_attn", "aktattn_pos", "aktmono_pos", "akt_raschx", "akt_raschy", "aktvec_raschx","dtransformer"]:
            y = torch.masked_select(ys[0], mask_future)
            t = torch.masked_select(rshft, mask_future)
            loss = binary_cross_entropy(y.double(), t.double()) + preloss[0]
        elif model_name == "lpkt":
            y = torch.masked_select(ys[0], mask_future)
            t = torch.masked_select(rshft, mask_future)
            criterion = nn.BCELoss(reduction='none')        
            loss = criterion(y, t).sum()
        
        return loss

    def model_train(self, epochs=3):
        best_epoch = -1
        for epoch in range(epochs):
            if epoch + 1 <= self.epoch_exist: continue
            self.model.train()
            total_loss = 0
            for batch in self.train_dataloader:
                self.optimizer.zero_grad()
                
                print(batch['concept_past'])
                concept_past = batch['concept_past'].to(self.device)
                response_past = batch['response_past'].to(self.device)
                question_past = batch['question_past'].to(self.device)
                id_data_past = batch['id_data_past'].to(self.device)
                mask_past = batch['mask_past'].to(self.device)

                concept_future = batch['concept_future'].to(self.device)
                response_future = batch['response_future'].to(self.device)
                question_future = batch['question_future'].to(self.device)
                id_data_future = batch['id_data_future'].to(self.device)
                mask_future = batch['mask_future'].to(self.device)
                
                ys, preloss = [], []
                
                match self.input_type:
                    case 'past':
                        c = concept_past
                        r = response_past
                        q = question_past
                        id = id_data_past
                    case 'past_future':
                        c = torch.cat((concept_past, concept_future), dim=1)
                        rsp_future_pad = torch.zeros((response_future.size(0), response_future.size(1))).long()
                        r = torch.cat((response_past, rsp_future_pad), dim=1)
                        q = torch.cat((question_past, question_future), dim=1)
                        id = torch.cat((id_data_past, id_data_future), dim=0)
                
                match self.model_name:
                    case 'akt':
                        y, reg_loss = self.model(c, r, id, self.emb_dict_train, q)
                        if self.input_type == "past_future":
                            ys.append(y[:, self.max_length:])
                        elif self.input_type == "past":
                            ys.append(y)
                        preloss.append(reg_loss)
                    case 'dkt':
                        y = self.model(c.long(), r.long(), id, self.emb_dict_train, q)
                        if self.input_type == "past_future":
                            y = (y[:, self.max_length:, :] * one_hot(concept_future.long(), self.model.num_c)).sum(-1)
                        else:
                            y = (y * one_hot(concept_future.long(), self.model.num_c)).sum(-1)
                        ys.append(y)
                    case 'atkt':
                        y, features = self.model(c.long(), r.long(), id, self.emb_dict_train, q)

                        if self.input_type == "past_future":
                            y = (y[:, self.max_length:, :] * one_hot(concept_future.long(), self.model.num_c)).sum(-1)
                        else:
                            y = (y * one_hot(concept_future.long(), self.model.num_c)).sum(-1)
                        loss = self.cal_loss(self.model, [y], response_past, response_future, mask_past, mask_future)
                        # at
                        features_grad = grad(loss, features, retain_graph=True)
                        p_adv = torch.FloatTensor(self.model.epsilon * _l2_normalize_adv(features_grad[0].data))
                        p_adv = Variable(p_adv).to(self.device)
                        pred_res, _ = self.model(c.long(), r.long(), id, self.emb_dict_train, q, p_adv)
                        # second loss
                        if self.input_type == "past_future":
                            pred_res = (pred_res[:, self.max_length:, :] * one_hot(concept_future.long(), self.model.num_c)).sum(-1)
                        else:
                            pred_res = (pred_res * one_hot(concept_future.long(), self.model.num_c)).sum(-1)
                        adv_loss = self.cal_loss(self.model, [pred_res], response_past, response_future, mask_past, mask_future)
                        loss = loss + self.model.beta * adv_loss
                    case 'dkvmn':
                        y = self.model(c.long(), r.long(), id, self.emb_dict_train, q)
                        if self.input_type == "past_future":
                            ys.append(y[:, self.max_length:])
                        elif self.input_type == "past":
                            ys.append(y)
                    case 'simpleKT':
                        y, y2, y3 = self.model(q, c, r, id, self.emb_dict_train, train=True)
                        if self.input_type == "past_future":
                            ys = [y[:, self.max_length:]]
                        elif self.input_type == "past":
                            ys = [y]
                                            
                if self.model_name not in ["atkt", "atktfix"]:
                    loss = self.cal_loss(self.model, ys, response_past, response_future, mask_past, mask_future, preloss)

                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(self.train_dataloader)
            self.train_losses.append(avg_loss)
            print(f'Epoch {epoch + 1}, Training loss: {avg_loss}')

            val_loss, accuracy, f1 = self.model_eval('val')
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'train_loss': self.train_losses,
                'val_loss': self.val_losses
            }
            self.model_save(checkpoint, self.checkpoint_last_path)

            if len(self.val_losses) == 0 or val_loss == min(self.val_losses):
                self.model_save(checkpoint, self.checkpoint_best_path)
                best_epoch = checkpoint['epoch']
                
            if epoch - best_epoch >= 10:
                break

    def model_eval(self, eval_mode):
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []

        dataloader = self.test_dataloader if eval_mode == 'test' else self.val_dataloader
        emb_dic = self.emb_dict_test if eval_mode =='test' else self.emb_dict_train
        with torch.no_grad():
            y_trues = []
            y_scores = []
            for batch in dataloader:        
                concept_past = batch['concept_past'].to(self.device)
                response_past = batch['response_past'].to(self.device)
                question_past = batch['question_past'].to(self.device)
                id_data_past = batch['id_data_past'].to(self.device)
                mask_past = batch['mask_past'].to(self.device)

                concept_future = batch['concept_future'].to(self.device)
                response_future = batch['response_future'].to(self.device)
                question_future = batch['question_future'].to(self.device)
                id_data_future = batch['id_data_future'].to(self.device)
                mask_future = batch['mask_future'].to(self.device)
                
                uid_data_past = batch['uid_data_past']
                uid_data_future = batch['uid_data_future']
                uid = uid_data_future
                
                ys, preloss = [], []
                
                match self.input_type:
                    case 'past':
                        c = concept_past
                        r = response_past
                        q = question_past
                        id = id_data_past
                    case 'past_future':
                        c = torch.cat((concept_past, concept_future), dim=1)
                        rsp_future_pad = torch.zeros((response_future.size(0), response_future.size(1))).long()
                        r = torch.cat((response_past, rsp_future_pad), dim=1)
                        q = torch.cat((question_past, question_future), dim=1)
                        id = torch.cat((id_data_past, id_data_future), dim=0)
             
                match self.model_name:
                    case 'akt':
                        # y, reg_loss = self.model(cc, cr, id_data, self.emb_dict_train, cq)
                        y, reg_loss = self.model(c, r, id, emb_dic, q)
                        if self.input_type == "past_future":
                            ys.append(y[:, self.max_length:])
                        elif self.input_type == "past":
                            ys.append(y)
                        preloss.append(reg_loss)
                    case 'dkt':
                        y = self.model(c.long(), r.long(), id, emb_dic, q)
                        if self.input_type == "past_future":
                            y = (y[:, self.max_length:, :] * one_hot(concept_future.long(), self.model.num_c)).sum(-1)
                        else:
                            y = (y * one_hot(concept_future.long(), self.model.num_c)).sum(-1)

                        ys.append(y)
                    case 'atkt':
                        y, _ = self.model(c.long(), r.long(), id, emb_dic, q)
                        if self.input_type == "past_future":
                            y = (y[:, self.max_length:, :] * one_hot(concept_future.long(), self.model.num_c)).sum(-1)
                        else:
                            y = (y * one_hot(concept_future.long(), self.model.num_c)).sum(-1)
                    case 'dkvmn':
                        y = self.model(c.long(), r.long(), id, emb_dic, q)
                        if self.input_type == "past_future":
                            ys.append(y[:, self.max_length:])
                        elif self.input_type == "past":
                            ys.append(y)
                    case 'simpleKT':
                        y = self.model(q, c, r, id, emb_dic)
                        if self.input_type == "past_future":
                            ys.append(y[:, self.max_length:])
                        elif self.input_type == "past":
                            ys.append(y)
                        
                if self.model_name not in ["atkt", "atktfix"]:        
                    loss = self.cal_loss(self.model, ys, response_past, response_future, mask_past, mask_future, preloss)
                    total_loss += loss.item()
                elif self.model_name in ["atkt"]:
                    loss = self.cal_loss(self.model, [y], response_past, response_future, mask_past, mask_future)
                    total_loss += loss.item()
                else:
                    print("Didn't calculate loss.")
                    total_loss = 0
                
                # outputs of DKT and ATKT have already been processed
                if self.input_type == "past_future" and self.model_name in ["akt", "dkvmn", "simpleKT"]:
                    y = torch.masked_select(y[:, self.max_length:], mask_future).detach().cpu()
                else:
                    y = torch.masked_select(y, mask_future).detach().cpu()
                t = torch.masked_select(response_future, mask_future).detach().cpu()

                y_trues.append(t.numpy())
                y_scores.append(y.numpy())
                
            avg_loss = total_loss / len(self.train_dataloader)
            ts = np.concatenate(y_trues, axis=0)
            ps = np.concatenate(y_scores, axis=0)
            prelabels = [1 if p >= 0.5 else 0 for p in ps]
            
            if eval_mode == 'test':
                self.dataframe_raw['predicted_correctness'] = -1
                uids = []
                
                for j, batch_mask in enumerate(mask_future):
                    for i, mask in enumerate(batch_mask):
                        if mask:
                            uids.append(uid[j][i])
                
                for i, uid in enumerate(uids):
                    self.dataframe_raw.loc[self.dataframe_raw["uid"] == uid, "predicted_correctness"] = prelabels[i]
                # print(uids[0], prelabels[0])
                # print(self.dataframe_raw[self.dataframe_raw["uid"] == 'c6442e41-39f3-4589-81fc-c7162eda3471'])
                
                self.dataframe_raw.to_csv(f"./output/eduAgent/{self.model_name}_{self.input_type}_output_data.csv", index=False)
                # print(self.dataframe_raw["predicted_correctness"])
                print("sum is ", sum(self.dataframe_raw["predicted_correctness"]))

            accuracy = accuracy_score(ts, prelabels)
            f1 = f1_score(ts, prelabels, average='weighted')
            print(f'{eval_mode.capitalize()} loss: {avg_loss}, Accuracy: {accuracy}, F1 Score: {f1}')

        return avg_loss, accuracy, f1

def run_exp(model_name, input_type):
    # GKT
    # dataset_path_train = './data/gkt_new/cogedu_emb_raw_train.csv'
    # dataset_path_test = './data/gkt_new/cogedu_emb_raw_test.csv'
    # dataset_raw_path = './data/gkt_new/cogedu_emb_raw.csv'
    
    # eduAgent
    dataset_path_train = './data/gkt_bert/eduAgent_emb_train.csv'
    dataset_path_test = './data/gkt_bert/eduAgent_emb_test.csv'
    dataset_raw_path = './data/gkt_bert/eduAgent_emb_test.csv'
    log_folder = './data'

    # concept: slide
    # 1-index: +1; 0-index: no need to +1
    # Embedding layer size match
    
    # GKT
    # num_c = 195 + 1  # slide id
    # num_q = 233 + 1  # Number of unique problem IDs
    
    # eduAgent
    num_c = 18 + 1  # slide id
    num_q = 58 + 1  # Number of unique problem IDs
    d_model = 200  # Model dimension
    n_blocks = 4  # Number of blocks
    dropout = 0.2  # Dropout rate
    emb_size = 200

    experiment_pipeline = Experiment_Pipeline(100, log_folder, dataset_raw_path, 'none', num_q, num_c, d_model, n_blocks, dropout, model_name, emb_size, input_type)
    experiment_pipeline.dataset_prepare(dataset_path_train, dataset_path_test)
    print("--------------training--------------")
    experiment_pipeline.model_train(epochs=30)
    print("--------------testing--------------")
    experiment_pipeline.model_eval(eval_mode='test')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="simpleKT")
    parser.add_argument('--input_type', type=str, default="past_future")

    args = parser.parse_args()

    run_exp(model_name=args.model_name, input_type=args.input_type)