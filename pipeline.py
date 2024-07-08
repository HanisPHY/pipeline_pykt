import logging
from transformers import logging as transformers_logging
transformers_logging.set_verbosity_error()

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import os, random, time, shutil, warnings
from sklearn.metrics import accuracy_score, f1_score
from akt import AKT

from torch.nn.functional import binary_cross_entropy

def remove_folder(folder_path):
    try:
        shutil.rmtree(folder_path)
        print(f"Folder '{folder_path}' and all its contents have been removed successfully.")
    except FileNotFoundError:
        print(f"The folder '{folder_path}' does not exist.")
    except Exception as e:
        print(f"An error occurred while removing the folder: {e}")

# class CustomDataset(Dataset):
#     def __init__(self, dataframe, emb_dict, max_length):
#         self.dataframe = dataframe
#         self.emb_dict = emb_dict
#         self.max_length = max_length
        
#         self.grouped = self.dataframe.groupby('student_id').apply(lambda x: x).reset_index(drop=True)

#     def __len__(self):
#         return len(self.grouped['student_id'].unique())

#     def __getitem__(self, idx):
#         student_id = self.grouped['student_id'].unique()[idx]
#         student_data = self.grouped[self.grouped['student_id'] == student_id]

#         q_data = torch.tensor(student_data['course_name'].values, dtype=torch.long)
#         response = torch.tensor(student_data['correctness'].values, dtype=torch.float)
#         pid_data = torch.tensor(student_data['question_id'].values, dtype=torch.long)
#         id_data = torch.tensor(student_data['student_id'].values, dtype=torch.long)[0]  # Same for all rows for this student

#         embeddings = []
#         for qid in pid_data:
#             emb = self.emb_dict.get((student_id, qid.item()))
#             if emb is None:
#                 raise ValueError(f"No embedding found for (student_id, question_id): ({student_id}, {qid.item()})")
#             embeddings.append(emb)
#         emb_data = torch.tensor(embeddings, dtype=torch.float32)

#         return {
#             'q_data': q_data,
#             'response': response,
#             'pid_data': pid_data,
#             'emb_data': emb_data,
#             'id_data': id_data
#         }

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
        # print("student_data is ", student_data)

        q_data = student_data['course_name'].values
        response = student_data['correctness'].values
        pid_data = student_data['question_id'].values
        id_data = student_data['student_id'].values[0]  # Same for all rows for this student
        # print("__getitem__, id_data is ", id_data)

        # Padding
        pad_length = self.max_length - len(q_data)
        mask = np.ones(self.max_length)
        if pad_length > 0:
            q_data = np.pad(q_data, (0, pad_length), 'constant', constant_values=0)
            response = np.pad(response, (0, pad_length), 'constant', constant_values=0)
            pid_data = np.pad(pid_data, (0, pad_length), 'constant', constant_values=0)
            mask[len(q_data)-pad_length:] = 0
        else:
            q_data = q_data[:self.max_length]
            response = response[:self.max_length]
            pid_data = pid_data[:self.max_length]
    
        # processed data
        q_data = torch.tensor(q_data, dtype=torch.int)
        response = torch.tensor(response, dtype=torch.int)
        pid_data = torch.tensor(pid_data, dtype=torch.int)
        id_data = torch.tensor(id_data, dtype=torch.int)
        mask = torch.tensor(mask, dtype=torch.bool)
        # print(q_data.shape, mask.shape)
        
        # pykt processing before training model
        mask = mask[:-1]
        
        shft_q_data = q_data[1:] * mask
        shft_response = response[1:] * mask
        shft_pid_data = pid_data[1:] * mask
        
        q_data = q_data[:-1]
        response = response[:-1]
        pid_data = pid_data[:-1]
        
        
        embeddings = []
        for qid in pid_data:
            if qid == -1 or qid.item() == 0:
                emb = [0.0] * 768  # Assuming the embedding dimension is 768
            else:
                emb = self.emb_dict.get((student_id, qid.item()))
                if emb is None:
                    raise ValueError(f"No embedding found for (student_id, question_id): ({student_id}, {qid.item()})")
            embeddings.append(emb)
        emb_data = torch.tensor(embeddings, dtype=torch.float32)

        return {
            'raw_q_data': q_data,
            'raw_response': response,
            'raw_pid_data': pid_data,
            'shft_q_data': shft_q_data,
            'shft_response': shft_response,
            'shft_pid_data': shft_pid_data,
            'emb_data': emb_data,
            'id_data': id_data,
            'mask': mask
        }

class Experiment_Pipeline():
    def __init__(self, max_length, log_folder, dataset_raw_path, load_model_type, n_question, n_pid, d_model, n_blocks, dropout, lr=1e-5):
        self.max_length = max_length
        self.set_seed(4)

        self.dataframe_raw = pd.read_csv(dataset_raw_path, sep='\t')
        self.log_folder = log_folder
        self.load_model_type = load_model_type

        self.n_question = n_question
        self.n_pid = n_pid
        self.d_model = d_model
        self.n_blocks = n_blocks
        self.dropout = dropout

        self.model_init(load_model_type, lr)

    def set_seed(self, seed_num):
        np.random.seed(seed_num)
        random.seed(seed_num)
        torch.manual_seed(seed_num)
        torch.cuda.manual_seed(seed_num)
        torch.cuda.manual_seed_all(seed_num)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def dataset_prepare(self, dataset_path_train, dataset_path_test, batch_size=64):
        dataframe_train = pd.read_csv(dataset_path_train, sep='\t')
        dataframe_test = pd.read_csv(dataset_path_test, sep='\t')
        
        for i, row in dataframe_test.iterrows():
            if pd.isna(row["correctness"]):
                dataframe_test.at[i, "correctness"] = int(row["student_choice"] == row["correct_choice"])
        
        emb_dict_train = {"type": "train"}
        emb_dict_test = {"type": "test"}

        for _, row in dataframe_train.iterrows():
            str_emb = row["embedding_bert"].strip('][').split(', ')
            float_emb = [float(emb) for emb in str_emb]
            emb_dict_train[(int(row['student_id']), int(row['question_id']))] = float_emb
            # print("(row['student_id'], row['question_id']) is ", (row['student_id'], row['question_id']))
        # print("emb_dict_train[(136,9)] is ", emb_dict_train[(136,9)])
        for _, row in dataframe_test.iterrows():
            str_emb = row["embedding_bert"].strip('][').split(', ')
            float_emb = [float(emb) for emb in str_emb]
            emb_dict_test[(int(row['student_id']), int(row['question_id']))] = float_emb
        # print("type of row['student_id'] is ", type(row['student_id']))
        # print("type of row['question_id'] is ", type(row['question_id']))
        # print("emb_dict_test[(136, 9)] is ", emb_dict_test[(136, 9)])
        self.emb_dict_train = emb_dict_train
        self.emb_dict_test = emb_dict_test

        # dataset = CustomDataset(dataframe_train, emb_dict_train, self.max_length)
        # test_dataset = CustomDataset(dataframe_test, emb_dict_test, self.max_length)

        # train_size = int(0.8 * len(dataset))
        # val_size = len(dataset) - train_size
        # train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        past_data = dataframe_train[dataframe_train["xy_type"] == "past"]
        future_data = dataframe_train[dataframe_train["xy_type"] == "future"]
        print("future_data[future_data[student_id]] is ", future_data["student_id"])
        # test_future_data = dataframe_test[dataframe_test["xy_type"] == "future"]
        
        train_dataset = CustomDataset(past_data, emb_dict_train, self.max_length)
        val_dataset = CustomDataset(future_data, emb_dict_train, self.max_length)
        test_dataset = CustomDataset(dataframe_test, emb_dict_test, self.max_length)

        self.train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
        self.val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
        self.test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    def model_save(self, checkpoint, checkpoint_path):
        if os.path.exists(checkpoint_path): os.remove(checkpoint_path)
        torch.save(checkpoint, checkpoint_path)

    def model_init(self, load_model_type, lr=1e-4):
        assert load_model_type in ['best', 'last', 'none']

        self.checkpoint_last_path = os.path.join(self.log_folder, 'model_last.pt')
        self.checkpoint_best_path = os.path.join(self.log_folder, 'model_best.pt')

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.device = torch.device('cpu')

        if load_model_type in ['best', 'last']:
            checkpoint_path = self.checkpoint_last_path if load_model_type == 'last' else self.checkpoint_best_path
            checkpoint = torch.load(checkpoint_path)
            self.model = AKT(self.n_question, self.n_pid, self.d_model, self.n_blocks, self.dropout).to(self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer = optim.AdamW(self.model.parameters(), lr=lr)
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epoch_exist = checkpoint['epoch']
            self.train_losses = checkpoint['train_loss']
            self.val_losses = checkpoint['val_loss']
        else:
            self.model = AKT(self.n_question, self.n_pid, self.d_model, self.n_blocks, self.dropout).to(self.device)
            self.optimizer = optim.AdamW(self.model.parameters(), lr=lr)
            self.epoch_exist = 0
            self.train_losses = []
            self.val_losses = []

        self.criterion = nn.BCELoss()
        
    def cal_loss(self, model, ys, r, rshft, sm, preloss=[]):
        model_name = model.model_name

        if model_name in ["atdkt", "simplekt", "bakt_time", "sparsekt"]:
            y = torch.masked_select(ys[0], sm)
            t = torch.masked_select(rshft, sm)
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

            y = torch.masked_select(ys[0], sm)
            t = torch.masked_select(rshft, sm)
            loss = binary_cross_entropy(y.double(), t.double())
        elif model_name == "dkt+":
            y_curr = torch.masked_select(ys[1], sm)
            y_next = torch.masked_select(ys[0], sm)
            r_curr = torch.masked_select(r, sm)
            r_next = torch.masked_select(rshft, sm)
            loss = binary_cross_entropy(y_next.double(), r_next.double())

            loss_r = binary_cross_entropy(y_curr.double(), r_curr.double()) # if answered wrong for C in t-1, cur answer for C should be wrong too
            loss_w1 = torch.masked_select(torch.norm(ys[2][:, 1:] - ys[2][:, :-1], p=1, dim=-1), sm[:, 1:])
            loss_w1 = loss_w1.mean() / model.num_c
            loss_w2 = torch.masked_select(torch.norm(ys[2][:, 1:] - ys[2][:, :-1], p=2, dim=-1) ** 2, sm[:, 1:])
            loss_w2 = loss_w2.mean() / model.num_c

            loss = loss + model.lambda_r * loss_r + model.lambda_w1 * loss_w1 + model.lambda_w2 * loss_w2
        elif model_name in ["akt","folibikt", "akt_vector", "akt_norasch", "akt_mono", "akt_attn", "aktattn_pos", "aktmono_pos", "akt_raschx", "akt_raschy", "aktvec_raschx","dtransformer"]:
            y = torch.masked_select(ys[0], sm)
            t = torch.masked_select(rshft, sm)
            loss = binary_cross_entropy(y.double(), t.double()) + preloss[0]
        elif model_name == "lpkt":
            y = torch.masked_select(ys[0], sm)
            t = torch.masked_select(rshft, sm)
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
                ys, preloss = [], []
                self.optimizer.zero_grad()
                q_data = batch['raw_q_data'].to(self.device)
                response = batch['raw_response'].to(self.device)
                pid_data = batch['raw_pid_data'].to(self.device)
                id_data = batch['id_data'].to(self.device)
                # print("id_data is ", id_data)
                
                
                shft_q_data = batch['shft_q_data'].to(self.device)
                shft_response = batch['shft_response'].to(self.device)
                shft_pid_data = batch['shft_pid_data'].to(self.device)
                
                sm = batch['mask'].to(self.device)
                # print("Shape of the inputs is ", q_data.shape, response.shape, pid_data.shape, id_data.shape)
                
                cq = torch.cat((pid_data[:,0:1], shft_pid_data), dim=1)
                cc = torch.cat((q_data[:,0:1], shft_q_data), dim=1)
                cr = torch.cat((response[:,0:1], shft_response), dim=1)

                y, reg_loss = self.model(cc, cr, id_data, self.emb_dict_train, cq)
                ys.append(y[:,1:])
                preloss.append(reg_loss)
                loss = self.cal_loss(self.model, ys, response, shft_response, sm, preloss)
                
                # print("preds is ", preds)

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

    # def model_eval(self, eval_mode):
    #     self.model.eval()
    #     total_loss = 0
    #     all_preds = []
    #     all_labels = []

    #     dataloader = self.test_dataloader if eval_mode == 'test' else self.val_dataloader
    #     with torch.no_grad():
    #         for batch in dataloader:
    #             q_data = batch['q_data'].to(self.device)
    #             response = batch['response'].to(self.device)
    #             pid_data = batch['pid_data'].to(self.device)
    #             id_data = batch['id_data'].to(self.device)

    #             preds, _ = self.model(q_data, response, id_data, self.emb_dict_test, pid_data)
    #             loss = self.criterion(preds, response)
    #             total_loss += loss.item()

    #             preds = (preds > 0.5).float()
    #             all_preds.extend(preds.cpu().numpy())
    #             all_labels.extend(response.cpu().numpy())

    #     avg_loss = total_loss / len(dataloader)
    #     accuracy = accuracy_score(all_labels, all_preds)
    #     f1 = f1_score(all_labels, all_preds, average='weighted')
    #     print(f'{eval_mode.capitalize()} loss: {avg_loss}, Accuracy: {accuracy}, F1 Score: {f1}')
    #     self.val_losses.append(avg_loss)

    #     return avg_loss, accuracy, f1
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
                ys, preloss = [], []
                q_data = batch['raw_q_data'].to(self.device)
                response = batch['raw_response'].to(self.device)
                pid_data = batch['raw_pid_data'].to(self.device)
                id_data = batch['id_data'].to(self.device)
                # print("id_data is ", id_data)
                
                
                shft_q_data = batch['shft_q_data'].to(self.device)
                shft_response = batch['shft_response'].to(self.device)
                shft_pid_data = batch['shft_pid_data'].to(self.device)
                
                sm = batch['mask'].to(self.device)
                # print("Shape of the inputs is ", q_data.shape, response.shape, pid_data.shape, id_data.shape)
                
                cq = torch.cat((pid_data[:,0:1], shft_pid_data), dim=1)
                cc = torch.cat((q_data[:,0:1], shft_q_data), dim=1)
                cr = torch.cat((response[:,0:1], shft_response), dim=1)
                # print(f"response: {response}")
                # print(f"shft_response: {shft_response}")
                
                # print("cq ", -1 in cq)
                # print("cc ", -1 in cc)
                # print("cr ", -1 in cr)
                # print(f"cr: {cr}")
                
                y, reg_loss = self.model(cc, cr, id_data, emb_dic, cq)
                ys.append(y[:,1:])
                y = y[:,1:]
                preloss.append(reg_loss)
                loss = self.cal_loss(self.model, ys, response, shft_response, sm, preloss)
                total_loss += loss.item()
                
                y = torch.masked_select(y, sm).detach().cpu()
                t = torch.masked_select(shft_response, sm).detach().cpu()

                y_trues.append(t.numpy())
                y_scores.append(y.numpy())
                
            avg_loss = total_loss / len(self.train_dataloader)
            ts = np.concatenate(y_trues, axis=0)
            ps = np.concatenate(y_scores, axis=0)
            prelabels = [1 if p >= 0.5 else 0 for p in ps]

            accuracy = accuracy_score(ts, prelabels)
            f1 = f1_score(ts, prelabels, average='weighted')
            print(f'{eval_mode.capitalize()} loss: {avg_loss}, Accuracy: {accuracy}, F1 Score: {f1}')

        return avg_loss, accuracy, f1

def run_exp_akt():
    dataset_path_train = './data/dataset_gkt_bert_embed_train_past5.csv'
    dataset_path_test = './data/dataset_gkt_bert_embed_test_public_past5.csv'
    dataset_raw_path = './data/dataset_gkt_bert_embed_train_past5.csv'
    log_folder = './data'

    n_pid = 13  # Course name
    n_question = 19  # Number of unique problem IDs
    d_model = 200  # Model dimension
    n_blocks = 4  # Number of blocks
    dropout = 0.2  # Dropout rate

    experiment_pipeline = Experiment_Pipeline(200, log_folder, dataset_raw_path, 'none', n_question, n_pid, d_model, n_blocks, dropout)
    experiment_pipeline.dataset_prepare(dataset_path_train, dataset_path_test)
    experiment_pipeline.model_train(epochs=30)
    print("--------------testing--------------")
    experiment_pipeline.model_eval(eval_mode='test')

run_exp_akt()
