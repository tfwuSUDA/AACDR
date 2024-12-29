import numpy as np
from sklearn import metrics

class ResultMetirc:
    def __init__(self, labels, labels_seen, labels_unseen, preds, preds_seen, preds_unseen):
        self.preds = preds
        self.labels = labels

        self.preds_seen = preds_seen
        self.preds_unseen = preds_unseen

        self.labels_seen = labels_seen
        self.labels_unseen = labels_unseen

        self.nums = int(self.labels.shape[0])
        self.ones = int(np.sum(self.labels))
        self.zeros = int(self.nums - self.ones)
        self.whole_des = ' {0}(+):{1}(-) in {2} samples'.format(self.ones, self.zeros, self.nums)
        
        self.nums_seen = int(self.labels_seen.shape[0])
        self.ones_seen = int(np.sum(self.labels_seen))
        self.zeros_seen = int(self.nums_seen - self.ones_seen)
        self.seen_des = ' {0}(+):{1}(-) in {2} samples'.format(self.ones_seen, self.zeros_seen, self.nums_seen)

        self.nums_unseen = int(self.labels_unseen.shape[0])
        self.ones_unseen = int(np.sum(self.labels_unseen))
        self.zeros_unseen = int(self.nums_unseen - self.ones_unseen)
        self.unseen_des = ' {0}(+):{1}(-) in {2} samples'.format(self.ones_unseen, self.zeros_unseen, self.nums_unseen)

        auc = metrics.roc_auc_score(self.labels, self.preds)
        fpr, tpr, thresholds = metrics.roc_curve(self.labels, self.preds)
        best_threshold_idx = np.argmax(tpr-fpr)
        best_threshold = thresholds[best_threshold_idx]
        p = preds.copy()
        p[p>=best_threshold] = 1
        p[p<best_threshold] = 0
        acc = metrics.accuracy_score(self.labels, p)
        precision = metrics.precision_score(self.labels, p)
        recall = metrics.recall_score(self.labels, p)
        f1 = metrics.f1_score(self.labels, p)
        self.whole = {
            'auc': auc,
            'acc': acc,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }


        auc_seen = metrics.roc_auc_score(self.labels_seen, self.preds_seen)
        fpr, tpr, thresholds = metrics.roc_curve(self.labels_seen, self.preds_seen)
        best_threshold_idx = np.argmax(tpr-fpr)
        best_threshold = thresholds[best_threshold_idx]
        p = preds_seen.copy()
        p[p>=best_threshold] = 1
        p[p<best_threshold] = 0
        acc_seen = metrics.accuracy_score(self.labels_seen, p)
        precision_seen = metrics.precision_score(self.labels_seen, p)
        recall_seen = metrics.recall_score(self.labels_seen, p)
        f1_seen = metrics.f1_score(self.labels_seen, p)
        self.seen = {
            'auc': auc_seen,
            'acc': acc_seen,
            'precision': precision_seen,
            'recall': recall_seen,
            'f1': f1_seen
        }
        
        self.auc_unseen = metrics.roc_auc_score(self.labels_unseen, self.preds_unseen)
        auc_unseen= metrics.roc_auc_score(self.labels_unseen, self.preds_unseen)
        fpr, tpr, thresholds = metrics.roc_curve(self.labels_unseen, self.preds_unseen)
        best_threshold_idx = np.argmax(tpr-fpr)
        best_threshold = thresholds[best_threshold_idx]
        p = preds_unseen.copy()
        p[p>=best_threshold] = 1
        p[p<best_threshold] = 0
        acc_unseen = metrics.accuracy_score(self.labels_unseen, p)
        precision_unseen = metrics.precision_score(self.labels_unseen, p)
        recall_unseen = metrics.recall_score(self.labels_unseen, p)
        f1_unseen = metrics.f1_score(self.labels_unseen, p)
        self.unseen= {
            'auc': auc_unseen,
            'acc': acc_unseen,
            'precision': precision_unseen,
            'recall': recall_unseen,
            'f1': f1_unseen
        }
        
    def metrics_tuple(self):
        return self.whole, self.seen, self.unseen

    def __repr__(self):
        return 'Whole:{0}{1}\nSeen:{2}{3}\nUnseen:{4}{5}\n'.format(self.whole, self.whole_des, self.seen, self.seen_des, self.unseen, self.unseen_des)

    def saved_metric(self):
        s = ''
        for idx, idc in enumerate([self.whole,self.seen,self.unseen]):
            s += ['whole', 'seen', 'unseen'][idx] + ':\t'
            des = [self.whole_des, self.seen_des, self.unseen_des][idx]
            for i,key in enumerate(['auc', 'acc', 'precision', 'recall', 'f1']):
                if i != 4:    
                    s += str(idc[key]) + '\t'
                else:
                    s += str(idc[key]) +' '+ des + '\n'
            
        return s
