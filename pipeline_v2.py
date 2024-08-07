import logging
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

from torch.nn.functional import binary_cross_entropy

from torch.autograd import Variable, grad
from model.atkt import _l2_normalize_adv

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
        # print("student_data is\n", student_data)
        # print("student_data[student_id] is\n", student_data['student_id'])
        
        past_data = student_data[student_data["xy_type"] == "past"]
        future_data = student_data[student_data["xy_type"] == "future"]
        
        if len(past_data) == 0 or len(future_data) == 0:
            print(f"No valid data for student_id: {student_id}")
            return None
            
        # Process past data
        concept_past = past_data['course_name'].values
        response_past = past_data['correctness'].values
        question_past = past_data['question_id'].values
        # print("past_data['student_id'] is\n", past_data['student_id'])
        id_data_past = past_data['student_id'].values[0]
        
        # Padding
        pad_length_past = self.max_length - len(concept_past)
        mask_past = np.ones(self.max_length)
        if pad_length_past > 0:
            concept_past = np.pad(concept_past, (0, pad_length_past), 'constant', constant_values=0)
            response_past = np.pad(response_past, (0, pad_length_past), 'constant', constant_values=0)
            question_past = np.pad(question_past, (0, pad_length_past), 'constant', constant_values=0)
            mask_past[len(concept_past)-pad_length_past:] = 0
        else:
            concept_past = concept_past[:self.max_length]
            response_past = response_past[:self.max_length]
            question_past = question_past[:self.max_length]
        
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
        # print("Future data is ", future_data)
        concept_future = future_data['course_name'].values
        response_future = future_data['correctness'].values
        question_future = future_data['question_id'].values
        id_data_future = future_data['student_id'].values[0]
        
        # Padding
        pad_length_future = self.max_length - len(concept_future)
        mask_future = np.ones(self.max_length)
        if pad_length_future > 0:
            concept_future = np.pad(concept_future, (0, pad_length_future), 'constant', constant_values=0)
            response_future = np.pad(response_future, (0, pad_length_future), 'constant', constant_values=0)
            question_future = np.pad(question_future, (0, pad_length_future), 'constant', constant_values=0)
            mask_future[len(concept_future)-pad_length_future:] = 0
        else:
            concept_future = concept_future[:self.max_length]
            response_future = response_future[:self.max_length]
            question_future = question_future[:self.max_length]
    
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
            'emb_data_past': emb_data_past,
            'mask_past': mask_past,
            'concept_future': concept_future,
            'response_future': response_future,
            'question_future': question_future,
            'id_data_future': id_data_future,
            'emb_data_future': emb_data_future,
            'mask_future': mask_future,
        }

class Experiment_Pipeline():
    def __init__(self, max_length, log_folder, dataset_raw_path, load_model_type, num_q, num_c, d_model, n_blocks, dropout, model_name, emb_size, lr=1e-5):
        self.max_length = max_length
        self.set_seed(4)

        self.dataframe_raw = pd.read_csv(dataset_raw_path, sep='\t')
        self.log_folder = log_folder
        self.load_model_type = load_model_type

        self.num_q = num_q
        self.num_c = num_c
        self.d_model = d_model
        self.n_blocks = n_blocks
        self.dropout = dropout
        self.emb_size = emb_size
        self.model_name = model_name

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
            str_emb = row["content_emb"].strip('][').split(',')
            float_emb = [float(emb) for emb in str_emb]
            emb_dict_train[(int(row['student_id']), int(row['question_id']))] = float_emb
            # print("(row['student_id'], row['question_id']) is ", (row['student_id'], row['question_id']))
        # print("emb_dict_train[(136,9)] is ", emb_dict_train[(136,9)])
        for _, row in dataframe_test.iterrows():
            str_emb = row["content_emb"].strip('][').split(',')
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
        return default_collate(batch)

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
                case "saint":
                    self.model = SAINT(self.num_q, self.num_c, self.max_length, self.emb_size, NUM_ATTN_HEADS, DROPOUT).to(self.device)
                case "dkvmn":
                    self.model = DKVMN(self.num_c, dim_s=200, size_m=50).to(self.device)
                

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
                case "saint":
                    self.model = SAINT(self.num_q, self.num_c, self.max_length, self.emb_size, NUM_ATTN_HEADS, DROPOUT).to(self.device)
                case "dkvmn":
                    self.model = DKVMN(self.num_c, dim_s=DKVMN_DIM_S, size_m=DKVMN_SIZE_M).to(self.device)
                    
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
            print(ys[0].shape, mask_future.shape)
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
                # cq = torch.cat((question_past[:,0:1], question_future), dim=1)
                # cc = torch.cat((concept_past[:,0:1], concept_future), dim=1)
                # cr = torch.cat((response_past[:,0:1], response_future), dim=1)
                print("question_past.shape is ", question_past.shape)
                print("question_future.shape is ", question_future.shape)
                
                match self.model_name:
                    case 'akt':
                        # y, reg_loss = self.model(cc, cr, id_data, self.emb_dict_train, cq)
                        y, reg_loss = self.model(concept_past, response_past, id_data_past, self.emb_dict_train, question_past)
                        ys.append(y)
                        preloss.append(reg_loss)
                    case 'dkt':
                        y = self.model(concept_past.long(), response_past.long(), id_data_past, self.emb_dict_train, question_past)
                        y = (y * one_hot(concept_future.long(), self.model.num_c)).sum(-1)
                        ys.append(y)
                    case 'atkt':
                        y, features = self.model(concept_past.long(), response_past.long(), id_data_past, self.emb_dict_train, question_past)
                        # print("y is ", y)
                        # print("features is ", features)
                        y = (y * one_hot(concept_future.long(), self.model.num_c)).sum(-1)
                        # print("after processing, y is ", y)
                        loss = self.cal_loss(self.model, [y], response_past, response_future, mask_past, mask_future)
                        # at
                        features_grad = grad(loss, features, retain_graph=True)
                        p_adv = torch.FloatTensor(self.model.epsilon * _l2_normalize_adv(features_grad[0].data))
                        p_adv = Variable(p_adv).to(self.device)
                        pred_res, _ = self.model(concept_past.long(), response_past.long(), id_data_past, self.emb_dict_train, question_past, p_adv)
                        # second loss
                        pred_res = (pred_res * one_hot(concept_future.long(), self.model.num_c)).sum(-1)
                        adv_loss = self.cal_loss(self.model, [pred_res], response_past, response_future, mask_past, mask_future)
                        loss = loss + self.model.beta * adv_loss
                    case 'saint':
                        y = self.model(question_past.long(), concept_past.long(), response_past.long(), id_data_past, self.emb_dict_train)
                        ys.append(y[:, 1:])
                    case 'dkvmn':
                        y = self.model(concept_past.long(), response_past.long(), id_data_past, self.emb_dict_train, question_past)
                        ys.append(y)
                
                # print("len(ys) is ", len(ys))
                # print("response future is ", response_future[0])
                # print("mask_future is ", mask_future[0])
                # print("y is ", y)
                # print("response_future is ", response_future)
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
                
                ys, preloss = [], []
                # cq = torch.cat((question_past[:,0:1], question_future), dim=1)
                # cc = torch.cat((concept_past[:,0:1], concept_future), dim=1)
                # cr = torch.cat((response_past[:,0:1], response_future), dim=1)
                
                
                match self.model_name:
                    case 'akt':
                        # y, reg_loss = self.model(cc, cr, id_data, self.emb_dict_train, cq)
                        y, reg_loss = self.model(concept_past, response_past, id_data_past, emb_dic, question_past)
                        ys.append(y)
                        preloss.append(reg_loss)
                    case 'dkt':
                        y = self.model(concept_past.long(), response_past.long(), id_data_past, emb_dic, question_past)
                        y = (y * one_hot(concept_future.long(), self.model.num_c)).sum(-1)
                        ys.append(y)
                    case 'atkt':
                        y, _ = self.model(concept_past.long(), response_past.long(), id_data_past, emb_dic, question_past)
                        y = (y * one_hot(concept_future.long(), self.model.num_c)).sum(-1)
                    case 'saint':
                        y, h = self.model(question_past.long(), concept_past.long(), response_past.long(), id_data_past, emb_dic, True)
                        y = y[:,1:]
                    case 'dkvmn':
                        y = self.model(concept_past.long(), response_past.long(), id_data_past, emb_dic, question_past)
                        ys.append(y)
                        
                if self.model_name not in ["atkt", "atktfix"]:        
                    loss = self.cal_loss(self.model, ys, response_past, response_future, mask_past, mask_future, preloss)
                    total_loss += loss.item()
                else:
                    print("Didn't calculate loss.")
                    total_loss = 0
                
                y = torch.masked_select(y, mask_future).detach().cpu()
                t = torch.masked_select(response_future, mask_future).detach().cpu()

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

def run_exp():
    dataset_path_train = './data/gkt_new/cogedu_emb_train.csv'
    dataset_path_test = './data/gkt_new/cogedu_emb_test.csv'
    dataset_raw_path = './data/gkt_new/cogedu_emb.csv'
    log_folder = './data'

    # concept: course
    # 1-index: +1; 0-index: no need to +1
    # Embedding layer size match
    num_c = 12 + 1  # Course name
    num_q = 119 + 1  # Number of unique problem IDs
    d_model = 200  # Model dimension
    n_blocks = 4  # Number of blocks
    dropout = 0.2  # Dropout rate
    emb_size = 200
    model_name = "akt"

    experiment_pipeline = Experiment_Pipeline(20, log_folder, dataset_raw_path, 'none', num_q, num_c, d_model, n_blocks, dropout, model_name, emb_size)
    experiment_pipeline.dataset_prepare(dataset_path_train, dataset_path_test)
    experiment_pipeline.model_train(epochs=30)
    print("--------------testing--------------")
    experiment_pipeline.model_eval(eval_mode='test')

run_exp()
