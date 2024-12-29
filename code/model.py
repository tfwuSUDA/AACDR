import os
import torch
import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torch.nn.init as init
from torch.nn.parameter import Parameter

class GCNLayer(nn.Module):
    def __init__(self,in_channels, out_channels, dropout_rate = 0.2):
        super(GCNLayer,self).__init__()
        self.fc = nn.Linear(in_channels, out_channels, bias = False)
        if dropout_rate != 0:
            self.dpt = True
            self.dropout = nn.Dropout(dropout_rate)
        else:
            self.dpt = False
        
    def forward(self, features, edges):
        outputs = self.fc(features)
        outputs = torch.matmul(edges,outputs)
        if self.dpt:
            outputs = self.dropout(outputs)
        return outputs
      
class GCNConv(nn.Module):
    def __init__(self):
        super(GCNConv, self).__init__()
        self.gcn1 = GCNLayer(75, 128, dropout_rate=0.2)
        self.gcn2 = GCNLayer(128, 128, dropout_rate=0.2)
        self.gcn3 = GCNLayer(128, 128, dropout_rate=0.2)
        self.gcn4 = GCNLayer(128, 128, dropout_rate=0)

    def forward(self, features, adjs):
        out1 = self.gcn1(features, adjs)
        out1 = F.relu(out1)
        out2 = self.gcn2(out1, adjs)
        out2 = F.relu(out2)
        out3 = self.gcn3(out2, adjs)
        out3 = F.relu(out3)
        out3 = out2 + out3
        out = self.gcn4(out3, adjs)
        # out = F.relu(out4)

        return out[:, -1, :]


class TransposeBatchNorm1d(nn.Module):
    def __init__(self, dim1, dim2, feature):
        super(TransposeBatchNorm1d, self).__init__()
        self.dim1 = dim1
        self.dim2 = dim2
        self.bn = nn.BatchNorm1d(feature)
    
    def forward(self, x):
        x = x.transpose(self.dim1, self.dim2)
        x = self.bn(x)
        return x.transpose(self.dim1, self.dim2)

class GINLayer(nn.Module):
    def __init__(self, features = [128, 256, 256], dp_rate = 0.3):
        super(GINLayer, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(features[0], features[1]),
            # TransposeBatchNorm1d(1,2,features[1]),
            nn.ReLU(),
            nn.Dropout(dp_rate),
            
            nn.Linear(features[1], features[2]),
            # TransposeBatchNorm1d(1,2,features[1]),
            nn.ReLU(),
            # nn.Dropout(0.2),

            # nn.Linear(features[1], features[2]),
            # nn.LeakyReLU(1/5.5),
            # nn.Dropout(0.3)
            # nn.ReLU()
        )
        self.eps = nn.Parameter(torch.Tensor([0]))

    def forward(self, features, adjs):
        hj = torch.bmm(adjs, features)
        hi = self.eps * features
        out = self.mlp(hi + hj)

        return out

class GINConv(nn.Module):
    def __init__(self, in_size = 38, h_dim = 128):
        super(GINConv, self).__init__()
        self.gin1 = GINLayer([75, 128, 128])
        self.gin2 = GINLayer([128, 128, 128])
        self.gin3 = GINLayer([128, 128, 128])
        self.gin4 = GINLayer([128, 128, 96])

    
    def forward(self, features, adjs):
        out1 = self.gin1(features, adjs)
        out2 = self.gin2(out1, adjs)
        out3 = self.gin3(out2, adjs)
        
        out3 = out3 + out2
        out = self.gin4(out3, adjs)

        out = torch.mean(out, dim=1)
        return out

    
class FE(nn.Module):
    def __init__(self, n_input=702, hidden_size = 256, out_size = 128):
        super(FE,self).__init__()
        self.out_size = out_size
        self.hidden_size = hidden_size        

        self.encoder = nn.Sequential(
            nn.Linear(n_input, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.02),
            nn.Dropout(0.2),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Linear(128, 128)
        )


    def forward(self, x):
        x = self.encoder(x)
        return x

            
class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.02),
            nn.Dropout(0.2),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 702)
        )

    def forward(self, expr):
        out = self.encoder(expr)
        return out
    
class DNN(nn.Module):
    def __init__(self):
        super(DNN, self).__init__()
        self.drug = GINConv(75)
        self.cls = nn.Sequential(
            nn.Linear(128+96, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.02),
            nn.Dropout(0.2),

            nn.Linear(128,64),
            nn.ReLU(),

            nn.Linear(64,32),
            nn.ReLU(),

            nn.Linear(32,1)
        )
    def forward(self, features, adjs, expr):
        drug_emb = self.drug(features, adjs)
        f = torch.cat((drug_emb, expr), dim=-1)
        out = self.cls(f)
        return out