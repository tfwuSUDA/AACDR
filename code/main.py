import os
import time
import pickle
import torch
import argparse
import statistics
import numpy as np
from sklearn import metrics
from result import *
from trainer import *

parser = argparse.ArgumentParser(description='.')
parser.add_argument('--description', type=str, help='model description')
parser.add_argument('--id', type=str, help = 'experiment id')
parser.add_argument('--model_nums', type=int, default=100, help='model number')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--max_epoch', type=int, default=10, help='max epoch for every model')
parser.add_argument('--lr', type=float, default=0.0005)
parser.add_argument('--batch_size', type=int, default=768)

args = parser.parse_args()
description = args.description
max_epoch = args.max_epoch
id = args.id
model_nums = args.model_nums
seed = args.seed
lr = args.lr
batch_size = args.batch_size

print('seed:', seed)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

print('experminent description:{0}\nmodel_nums:{1}\tmax epoch:{2}\tlr:{3}\tbatch size:{4}'.format(description, model_nums, max_epoch, lr, batch_size))

GDSC_drug_list = read_drug_list('GDSC')
GDSC_drug_graph = read_drug_graph('GDSC', GDSC_drug_list)
TCGA_drug_list = read_drug_list('TCGA')
TCGA_drug_graph = read_drug_graph('TCGA', TCGA_drug_list)

cwd = os.getcwd().split(r'/')[:-1]
cwd = '/'.join(cwd)

with open(cwd + '/data/GDSC/GDSC_only_dataset.pkl', 'rb')as f:
    gdsc_dataset = pickle.load(f)

with open(cwd + '/data/zy/CDR/data/TCGA/TCGA_unlabel_dataset.pkl', 'rb')as f2: 
    tcga_unlabel_dataset = pickle.load(f2)

with open(cwd + '/data/zy/CDR/data/TCGA/TCGA_dataset.pkl', 'rb')as f2:
    TCGA_dataset = pickle.load(f2)
    
def save_record(record):
    res = ''
    total = {
        'whole':{'auc':[], 'acc':[], 'precision':[], 'recall':[], 'f1':[]},
        'seen':{'auc':[], 'acc':[], 'precision':[], 'recall':[], 'f1':[]},
        'unseen':{'auc':[], 'acc':[], 'precision':[], 'recall':[], 'f1':[]}
    }
    result = {
        'whole':{'auc':{'mean':0, 'std':1}, 'acc':{'mean':0, 'std':1}, 'precision':{'mean':0, 'std':1}, 'recall':{'mean':0, 'std':1}, 'f1':{'mean':0, 'std':1}},
        'seen':{'auc':{'mean':0, 'std':1}, 'acc':{'mean':0, 'std':1}, 'precision':{'mean':0, 'std':1}, 'recall':{'mean':0, 'std':1}, 'f1':{'mean':0, 'std':1}},
        'unseen':{'auc':{'mean':0, 'std':1}, 'acc':{'mean':0, 'std':1}, 'precision':{'mean':0, 'std':1}, 'recall':{'mean':0, 'std':1}, 'f1':{'mean':0, 'std':1}}
    }
    for CLS in record:
        s = CLS.saved_metric()
        res += s
        for t in ['auc', 'acc', 'precision', 'recall', 'f1']:
            m = CLS.metrics_tuple()
            total['whole'][t].append(m[0][t])
            total['seen'][t].append(m[1][t])
            total['unseen'][t].append(m[2][t])
        res += '------------------------------\n'
    l = len(record)
    for c in ['whole', 'seen', 'unseen']:
        for t in ['auc', 'acc', 'precision', 'recall', 'f1']:
            result[c][t]['mean'] = statistics.mean(total[c][t])
            result[c][t]['std'] = math.sqrt(statistics.variance(total[c][t]))
    with open(cwd + '/ckpt/log_'+str(seed)+'_'+str(id) + '_' + description +'.txt', 'w')as f:
        f.write(res)
    print(result)

record = []
best_auc = -1
worst_auc = -1
train_dataset, val_dataset = train_test_split(gdsc_dataset, test_size = 0.05, random_state = seed)

for model_number in range(model_nums):
    print('<============================>')
    print('model No.{0}'.format(model_number))
    

    T = AADATrainer(seed, lr, batch_size, max_epoch, GDSC_drug_graph, TCGA_drug_graph, train_dataset, val_dataset, tcga_unlabel_dataset, TCGA_dataset)
    metric, models = T.fit()
    record.append(metric)
    torch.save(models[0], cwd + '/ckpt/model_' + str(metric.whole['auc']) + 'fe_'+str(seed)+'_'+str(id) + '_' + description +'.pt')
    torch.save(models[1], cwd + '/ckpt/model_' + str(metric.whole['auc']) + 'dnn_'+str(seed)+'_'+str(id) + '_' + description +'.pt')
    torch.save(models[2], cwd + '/ckpt/model_' + str(metric.whole['auc']) + 'ecd_'+str(seed)+'_'+str(id) + '_' + description +'.pt')
    if model_number > 1:
        save_record(record=record)
for r in record:
    print(r)
    print('----')
print()
save_record(record=record)

