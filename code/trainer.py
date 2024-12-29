import os
import copy
import time
import pickle
import torch
import argparse
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from itertools import cycle
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import random_split
import scipy.sparse as sp
from result import *
from dataset import *
from utils import *
from model import *
import matplotlib.pyplot as plt

# class MyLoss(nn.Module):
#     def __init__(self, margin_type = 'sml1', margin_scalar = 0.25):
#         super(MyLoss, self).__init__()
#         self.margin_type = margin_type
#         self.margin_scalar = margin_scalar

#     def forward(self, x, y):
#         if self.margin_type == 'sml1':
#             loss = nn.SmoothL1Loss(reduction='none')(y, x)
#         else:
#             loss = nn.MSELoss(reduction='none')(y, x)

#         margin = x * self.margin_scalar
#         margin.requires_grad_(False)
#         diff = torch.abs(y - x)
#         mask = diff <= margin
#         loss = loss * mask
        
#         # Return the average loss
#         return loss.sum() / mask.sum() if mask.sum() > 0 else torch.tensor(0.0, device=loss.device)

# device = 'cuda:0'
device = torch.device('cuda:2')
mse_loss = torch.nn.MSELoss()

def domain_loss_fn(pred):
    return -torch.mean(torch.log(pred.sigmoid()))

class AADATrainer():
    def __init__(self, seed, lr, batch_size, max_epoch, GDSC_drug_graph, TCGA_drug_graph, train_dataset, val_dataset, tcga_unlabel_dataset, TCGA_dataset):
        self.cancer_fe = FE()
        self.classifier = DNN()
        self.autoEncoder = AutoEncoder()
        self.cancer_fe.to(device)
        self.classifier.to(device)
        self.autoEncoder.to(device)

        self.save_cancer_fe = None
        self.save_classifier = None
        self.save_autoEncoder = None

        # self.tcga_size = batch_size
        self.tcga_size = 512
        self.val_batch_size = 5629
        self.test_batch_size = 1
        self.seed = seed
        self.lr = lr
        self.batch_size = batch_size
        self.max_epoch = max_epoch

        self.set_optimizer()
        
        self.GDSC_drug_graph = GDSC_drug_graph
        self.TCGA_drug_graph = TCGA_drug_graph
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        # split_ratio = [0.1, 0.9]  
        # lengths = [int(split_ratio[i] * len(tcga_unlabel_dataset))+i for i in range(len(split_ratio))]  
        # datasets = random_split(tcga_unlabel_dataset, lengths)  

        # self.tcga_unlabel_dataset = datasets[0]
        # print(len(self.tcga_unlabel_dataset))
        self.tcga_unlabel_dataset = tcga_unlabel_dataset
        self.TCGA_dataset = TCGA_dataset
       
    def set_optimizer(self):
        self.optimizer_cancer = optim.Adam(self.cancer_fe.parameters(), lr=self.lr)
        self.optimizer_cls = optim.Adam(self.classifier.parameters(),lr=self.lr)
        self.optimizer_ae = optim.Adam(self.autoEncoder.parameters(),lr=self.lr)

        self.scheduler_cancer = optim.lr_scheduler.StepLR(self.optimizer_cancer, step_size=1, gamma=0.9)
        self.scheduler_cls = optim.lr_scheduler.StepLR(self.optimizer_cls, step_size=1,gamma=0.9)
        self.scheduler_ae = optim.lr_scheduler.StepLR(self.optimizer_ae, step_size=1, gamma=0.9) 

    def set_seed(self):
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)

        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    def zero_grad(self):
        self.optimizer_cancer.zero_grad()
        self.optimizer_cls.zero_grad()
        self.optimizer_ae.zero_grad()

    def get_data(self, graph_id_list, type):
        adjs = []
        features = []
        if type == 'GDSC':
            for gid in graph_id_list:
                tmp = self.GDSC_drug_graph[str(gid.item())]
                tmp = (tmp['feature'], tmp['adj'])
                adjs.append(torch.tensor(tmp[1]).unsqueeze(0))
                features.append(torch.tensor(tmp[0]).unsqueeze(0))
            features = torch.cat(features, dim=0).float().to(device)
            adjs = torch.cat(adjs, dim=0).float().to(device)
            return adjs, features
        else:
            seen_flag = False
            for gid in graph_id_list:
                str_gid = str(gid.item())
                if str_gid in self.GDSC_drug_graph:
                    seen_flag = True
                tmp = self.TCGA_drug_graph[str_gid]
                tmp = (tmp['feature'], tmp['adj'])
                adjs.append(torch.tensor(tmp[1]).unsqueeze(0))
                features.append(torch.tensor(tmp[0]).unsqueeze(0))
            features = torch.cat(features, dim=0).float().to(device)
            adjs = torch.cat(adjs, dim=0).float().to(device)
            return adjs, features, seen_flag
    def fit(self):
        print('train size:{0} val size:{1} test size:{2}'.format(len(self.train_dataset), len(self.val_dataset),len(self.TCGA_dataset)))
        print('tcga unlabel size:{0}'.format(len(self.tcga_unlabel_dataset)))
        print('batch size:{0}, tcga size:{1}'.format(self.batch_size, self.tcga_size))
        print('lr:{0}'.format(self.lr))
        
        daDataLoader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)
        daTCGADataLoader = DataLoader(self.tcga_unlabel_dataset, batch_size=self.tcga_size, shuffle=True, drop_last = True)

        valDataLoader = DataLoader(self.val_dataset, batch_size=self.val_batch_size, shuffle=True)
        TCGADataLoader = DataLoader(self.TCGA_dataset, batch_size = self.test_batch_size, shuffle=False)

        seen_drug = set()
        pos_weight = torch.tensor([1.0]).to(device)
        print('pos weight:{0}'.format(pos_weight.item()))
        gdsc_predict_loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        tcga_predict_loss_fn = torch.nn.BCEWithLogitsLoss()
        
        reconstruction_loss_fn = nn.SmoothL1Loss()
        # myreconstruction_loss_fn = MyLoss()

        self.best_auc = 0
        self.best_test_auc = 0
        self.wait = 0
        self.drop = 0
        self.val_auc = None

        source_loss_list = []
        tcga_loss1_list = []
        gdsc_loss2_list = []
        tcga_loss2_list = []
        loss2_list = []
        margin_list = []
        for i in range(self.max_epoch):
            print('\nepoch:', i)
            r,m = 0.1,0.02
            margin = 0.3
            source_lamda = 1
            print('r:{0},m:{1},source lamda:{2},margin:{3}'.format(r,m,source_lamda,margin))
            if i >= 1 :
                self.scheduler_cancer.step()
                self.scheduler_cls.step()
                self.scheduler_ae.step()
            self.cancer_fe.train()
            self.classifier.train()
            self.autoEncoder.train()

            batch_source_loss = 0
            batch_tcga_step1_loss = 0
            batch_step2_loss = 0
            batch_gdsc_step2_loss = 0
            batch_tcga_step2_loss = 0
            for step, batch in enumerate(zip(daDataLoader, cycle(daTCGADataLoader))):
                self.zero_grad()
                g,tcga_expr_data = batch
                graph_gdsc_id, gdsc_expr_data, label = g
                if i == 0 and step == 0:
                    print(graph_gdsc_id[:15])
                label = label.view(self.batch_size,1).float().to(device)

                gdsc_adjs, gdsc_features = self.get_data(graph_gdsc_id, 'GDSC')
                
                gdsc_expr_data = gdsc_expr_data.to(device).float().view(self.batch_size, 702)
                tcga_expr_data = tcga_expr_data.view(self.tcga_size,702).to(device).float()

                # step 1
                gdsc_expr = self.cancer_fe(gdsc_expr_data)
                tcga_expr = self.cancer_fe(tcga_expr_data)
                source_pred = self.classifier(gdsc_features, gdsc_adjs, gdsc_expr)
                reconstruct = self.autoEncoder(tcga_expr)

                re_loss = reconstruction_loss_fn(reconstruct, tcga_expr_data)
                source_loss = gdsc_predict_loss_fn(source_pred, label)

                loss = source_loss + r * re_loss
                loss.backward()
                
                batch_source_loss += source_loss.item()
                batch_tcga_step1_loss += re_loss.item()*r
                source_loss_list.append(source_loss.item())
                tcga_loss1_list.append(re_loss.item())
                
                self.optimizer_cancer.step()
                self.optimizer_cls.step()

                self.zero_grad()

                # step 2
                gdsc_expr = self.cancer_fe(gdsc_expr_data)
                reconstruct_gdsc = self.autoEncoder(gdsc_expr)
                
                tcga_expr = self.cancer_fe(tcga_expr_data)
                reconstruct_tcga = self.autoEncoder(tcga_expr)
                
                tcga_loss = reconstruction_loss_fn(reconstruct_tcga, tcga_expr_data)
                gdsc_loss = reconstruction_loss_fn(reconstruct_gdsc, gdsc_expr_data)

                
                loss = gdsc_loss + max(0, margin - tcga_loss) * m

                t = reconstruction_loss_fn(reconstruct_tcga, tcga_expr_data)
                gdsc_loss2_list.append(gdsc_loss.item())
                tcga_loss2_list.append(t.item())
                loss2_list.append(loss.item())
                margin_list.append(margin)

                loss.backward()
                batch_step2_loss += loss.item()
                batch_gdsc_step2_loss += gdsc_loss.item()
                batch_tcga_step2_loss += tcga_loss.item()

                self.optimizer_ae.step()
                self.zero_grad()     
 
            print('source_loss:', batch_source_loss)
            print('step1_tcga_loss:', batch_tcga_step1_loss)
            print('step2_loss:', batch_step2_loss)
            print('step2_gdsc_loss:', batch_gdsc_step2_loss)
            print('step2_tcga_loss:', batch_tcga_step2_loss)

            # x = [i for i in range(len(loss2_list))]
            # plt.plot(x, source_loss_list,color='pink',linestyle='solid',label='source loss')
            # plt.plot(x, tcga_loss1_list,color='red',linestyle='solid', label='tcga loss1')
            # plt.plot(x, gdsc_loss2_list,color='yellow',linestyle='dashed', label='gdsc_loss2')
            # plt.plot(x, tcga_loss2_list,color='blue',linestyle='dashed',label='tcga loss2')
            # plt.plot(x, loss2_list,color='red',linestyle='dotted',label='loss2')
            # plt.plot(x, margin_list,color='black',linestyle='dashdot',label='margin')

            # # plt.legend(['source loss', 'tcga loss1', 'gdsc loss2', 'tcga loss2', 'loss2', 'margin'])
            # plt.legend()
            # plt.savefig(cwd + '/loss-test.png')
            # plt.close()

            # validation
            labels = []
            preds = []

            self.cancer_fe.eval()
            self.classifier.eval()
            self.autoEncoder.eval()

            with torch.no_grad():
                step = 0 
                for batch in valDataLoader:
                    step += 1
                    graph_id, expr_data, label = batch
                    label = label.view(self.val_batch_size).float().to(device)
                    adjs, features = self.get_data(graph_id, 'GDSC')
                    
                    expr_data = expr_data.to(device).float().view(self.val_batch_size, 702)

                    expr = self.cancer_fe(expr_data)
                    pred = self.classifier(features, adjs, expr)
                    labels.extend(label.cpu().detach().numpy())
                    preds.extend(pred.sigmoid().cpu().detach().numpy())

                auc = metrics.roc_auc_score(labels, preds)
                print('validation auc:', auc)
                l = np.array(labels)
                p = np.array(preds)
                valresult = ResultMetirc(l,l,l,p,p,p)
                print('validation result:\n', valresult)
            self.val_auc = auc

            self.cancer_fe.eval()
            self.classifier.eval()
            self.autoEncoder.eval()

            total_loss = 0
            total_samples = 0
            labels = []
            seen_labels = []
            unseen_labels = []
            preds = []
            seen_preds = []
            unseen_preds = []

            with torch.no_grad():
                step = 0 
                for batch in TCGADataLoader:
                    step += 1
                    graph_id, expr_data, label = batch
                    label = label.view(self.test_batch_size).float().to(device)
                    adjs, features, seen_flag = self.get_data(graph_id, 'TCGA')
                    dim = features.shape[-1]
                    features = features.view(-1,100,dim)

                    expr_data = expr_data.to(device).float().view(-1,702)

                    expr = self.cancer_fe(expr_data)
                    pred = self.classifier(features, adjs, expr).view(self.test_batch_size)
                    
                    ls = tcga_predict_loss_fn(pred, label)
                    total_loss += ls.item()
                    total_samples += self.test_batch_size

                    labels.append(label.item())
                    preds.append(pred.sigmoid().item())

                    if seen_flag:
                        seen_labels.append(label.item())
                        seen_preds.append(pred.sigmoid().item())
                    else:
                        unseen_labels.append(label.item())
                        unseen_preds.append(pred.sigmoid().item())
                        
                labels = np.array(labels)
                preds = np.array(preds)
                
                seen_labels = np.array(seen_labels)
                seen_preds = np.array(seen_preds)

                unseen_labels = np.array(unseen_labels)
                unseen_preds = np.array(unseen_preds)

                result_metric = ResultMetirc(labels,seen_labels,unseen_labels,preds,seen_preds,unseen_preds)
                print('avg loss:', total_loss / total_samples)
                print('metric:\n', result_metric)
                print('<--------------------->')

                if self.val_auc > self.best_auc:
                    self.best_auc = self.val_auc
                    self.wait = 0
                    self.best_test_auc = result_metric
                    self.save_cancer_fe = copy.deepcopy(self.cancer_fe)
                    self.save_classifier = copy.deepcopy(self.classifier)
                    self.save_autoEncoder = copy.deepcopy(self.autoEncoder)
                else:
                    self.drop += 1
                    self.wait += 1
                    if self.wait > 3:
                        print('best auc:', self.best_test_auc)
                        return self.best_test_auc, [self.save_cancer_fe, self.save_classifier, self.save_autoEncoder]
        print('best auc:', self.best_test_auc)
        return self.best_test_auc, [self.save_cancer_fe, self.save_classifier, self.save_autoEncoder]


