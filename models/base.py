from abc import ABC, abstractmethod

import torch
from torch import nn
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
# import warnings
# warnings.filterwarnings('ignore')
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score, auc, f1_score
import pickle
from sklearn.metrics import confusion_matrix
from collections import defaultdict
import os

class KGModel(nn.Module, ABC):

    def __init__(self, sizes, rank, dropout, gamma, data_type, bias, init_size):
        super(KGModel, self).__init__()
        if data_type == 'double':
            self.data_type = torch.double
        else:
            self.data_type = torch.float
        self.sizes = sizes
        self.rank = rank
        self.dropout = dropout
        self.bias = bias
        self.init_size = init_size
        self.gamma = nn.Parameter(torch.Tensor([gamma]), requires_grad=False)
        self.entity = nn.Embedding(sizes[0], rank)
        self.rel = nn.Embedding(sizes[1], rank)
        self.bh = nn.Embedding(sizes[0], 1)
        self.bh.weight.data = torch.zeros((sizes[0], 1), dtype=self.data_type)
        self.bt = nn.Embedding(sizes[0], 1)
        self.bt.weight.data = torch.zeros((sizes[0], 1), dtype=self.data_type)

    @abstractmethod
    def get_queries(self, queries):
        pass
    
    @abstractmethod
    def get_queries_lp(self, queries, r):
        pass

    @abstractmethod
    def get_rhs(self, queries, eval_mode):
        pass

    @abstractmethod
    def similarity_score(self, lhs_e, rhs_e, eval_mode):
        pass

    def score(self, lhs, rhs, eval_mode):
        lhs_e, lhs_biases = lhs
        rhs_e, rhs_biases = rhs
        score = self.similarity_score(lhs_e, rhs_e, eval_mode)
        if self.bias == 'constant':
            return self.gamma.item() + score
        elif self.bias == 'learn':
            if eval_mode:
                return lhs_biases + rhs_biases.t() + score
            else:
                return lhs_biases + rhs_biases + score
        else:
            return score

    def get_factors(self, queries):
        head_e = self.entity(queries[:, 0])
        rel_e = self.rel(queries[:, 1])
        rhs_e = self.entity(queries[:, 2])
        return head_e, rel_e, rhs_e

    def forward(self, queries, eval_mode=False):
        lhs_e, lhs_biases = self.get_queries(queries)
        rhs_e, rhs_biases = self.get_rhs(queries, eval_mode)
        predictions = self.score((lhs_e, lhs_biases), (rhs_e, rhs_biases), eval_mode)
        factors = self.get_factors(queries)
        return predictions, factors

    def get_ranking(self, queries, filters, batch_size=1000):
        ranks = torch.ones(len(queries))
        with torch.no_grad():
            b_begin = 0
            candidates = self.get_rhs(queries, eval_mode=True)
            while b_begin < len(queries):
                these_queries = queries[b_begin:b_begin + batch_size].cuda()

                q = self.get_queries(these_queries)
                rhs = self.get_rhs(these_queries, eval_mode=False)

                scores = self.score(q, candidates, eval_mode=True)
                targets = self.score(q, rhs, eval_mode=False)

                for i, query in enumerate(these_queries):
                    filter_out = filters[(query[0].item(), query[1].item())]
                    filter_out += [queries[b_begin + i, 2].item()]
                    scores[i, torch.LongTensor(filter_out)] = -1e6
                ranks[b_begin:b_begin + batch_size] += torch.sum(
                    (scores >= targets).float(), dim=1
                ).cpu()
                
                b_begin += batch_size
                
        return ranks
    
    def curve_plot(self, fprs, tprs, direc, metric):
        if metric == 'roc':
            if direc =='rhs':
                plt.plot(fprs , tprs, label='ROC')
            else:
                plt.plot(fprs , tprs, label='ROC(lhs)')
            plt.plot([0, 1], [0, 1], 'k--', label='Random')

            start, end = plt.xlim()
            plt.xticks(np.round(np.arange(start, end, 0.1),2))
            plt.xlim(0,1)
            plt.ylim(0,1)
            plt.xlabel('FPR( 1 - Sensitivity )')
            plt.ylabel('TPR( Recall )')
            plt.legend()
            
        elif metric == 'pr':
            if direc =='rhs':
                plt.plot(fprs , tprs, label='P-R')
            else:
                plt.plot(fprs , tprs, label='P-R(lhs)')
            plt.plot([1, 0], [0, 1], 'k--', label='Random')
            
            start, end = plt.xlim()
            plt.xticks(np.round(np.arange(start, end, 0.1),2))
            plt.xlim(0,1)
            plt.ylim(0,1)
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.legend()
            
        
    def get_roc(
            self, queries: torch.Tensor,
            batch_size: int = 1000,
            direc='rhs', negs=[], save_path=None, num_rel=7
    ):
        
        pos_arr=np.array([])
        neg_arr=np.array([])

        with tqdm(total=queries.shape[0], unit='ex') as bar:
            bar.set_description(f'Evaluation')
            with torch.no_grad():
                b_begin = 0
                
                while b_begin < len(queries):
                    these_queries = queries[b_begin:b_begin + batch_size].cuda()
                    these_negs = negs[b_begin:b_begin + batch_size].cuda()

                    preds = [self.get_queries_lp(these_queries, torch.LongTensor([r]*these_queries.shape[0]).cuda()) for r in range(num_rel)]
                    
                    rhs = self.get_rhs(these_queries, eval_mode=False) # target
                    neg_rhs = self.get_rhs(these_negs, eval_mode=False)
                    
                    pos_scores = torch.max(torch.stack([self.score(q, rhs, eval_mode=False) for q in preds]), dim=0)
                    neg_scores = torch.max(torch.stack([self.score(q, neg_rhs, eval_mode=False) for q in preds]), dim=0)

                    pos_arr = np.append(pos_arr, pos_scores.values.cpu())
                    neg_arr = np.append(neg_arr, neg_scores.values.cpu())

                    b_begin += batch_size
                    bar.update(batch_size)

        y_true = np.concatenate([np.ones(len(pos_arr)), np.zeros(len(neg_arr))])
        y_scores = np.append(pos_arr, neg_arr)
        y_pred = y_scores
        
        plt.subplot(3,1,1)
        sns.histplot(data=pos_arr, kde=True, color='blue', stat="proportion")
        sns.histplot(data=neg_arr, kde=True, color='red', stat="proportion")
        fprs, tprs, thres = roc_curve(y_true, y_pred)
        plt.subplot(3,1,2)
        self.curve_plot(fprs, tprs, direc, 'roc')
        
        prec, recall, thres = precision_recall_curve(y_true, y_pred)
        plt.subplot(3,1,3)
        self.curve_plot(prec, recall, direc, 'pr')
        
        roauc = roc_auc_score(y_true, y_pred)
        prauc = auc(recall, prec)

        numerator = 2 * recall * prec
        denom = recall + prec
        f1_scores = np.divide(numerator, denom, out=np.zeros_like(denom), where=(denom!=0))
        max_f1 = np.max(f1_scores)
        
        print('AUROC: ', roauc)
        print('AUPRC: ', prauc)
        print('F1-score: ', max_f1)

        plt.savefig(save_path+'/ent_predict.png', dpi=600)

        return roauc, prauc, max_f1
    
    def get_rel_acc(
        self, queries: torch.Tensor,
        batch_size: int = 1000,
        save_path=None, num_rel=7
    ):
        with tqdm(total=queries.shape[0], unit='ex') as bar:
            bar.set_description(f'Evaluation')
            with torch.no_grad():
                b_begin = 0
                y_pred=np.array([])
                y_scores=np.array([])
                y_true=queries[:, 1].numpy()

                while b_begin < len(queries):
                    these_queries = queries[b_begin:b_begin + batch_size].cuda()

                    preds = [self.get_queries_lp(these_queries, torch.LongTensor([r]*these_queries.shape[0]).cuda()) for r in range(num_rel)]
                    
                    
                    rhs = self.get_rhs(these_queries, eval_mode=False) # target

                    p_scores = torch.stack([10.0+self.score(q, rhs, eval_mode=False) for q in preds]).squeeze()
                    softmax = torch.nn.Softmax(dim=0)
                    pos_scores = softmax(p_scores)
                    pos_scores = pos_scores.cpu().numpy().T
                    y_scores = np.append(y_scores, pos_scores)
                    pred_idx = np.argmax(pos_scores, axis=1)

                    y_pred = np.append(y_pred, pred_idx)

                    b_begin += batch_size
                    bar.update(batch_size)
        
        with open(file=save_path+'/rel_score.pickle', mode='wb') as f:
            pickle.dump(y_scores, f)
        with open(file=save_path+'/rel_pred.pickle', mode='wb') as f:
            pickle.dump(y_pred, f)
        
        accuracy = get_accuracy(y_true, y_pred)
        mic_f1 = f1_score(y_true, y_pred, average='micro')
        mac_f1 = f1_score(y_true, y_pred, average='macro')

        return mac_f1, mic_f1, accuracy
    
    def compute_metrics(self, examples, filters, batch_size=500):
        mean_rank = {}
        mean_reciprocal_rank = {}
        hits_at = {}

        for m in ["rhs"]:
            q = examples.clone()
            ranks = self.get_ranking(q, filters[m], batch_size=batch_size)
            mean_rank[m] = torch.mean(ranks).item()
            mean_reciprocal_rank[m] = torch.mean(1. / ranks).item()
            hits_at[m] = torch.FloatTensor((list(map(
                lambda x: torch.mean((ranks <= x).float()).item(),
                (1, 10, 50, 100)
            ))))

        return mean_rank, mean_reciprocal_rank, hits_at
    
    def compute_rel_rank(self, examples, filters, batch_size=500, num_rel=5):
        mean_rank = {}
        mean_reciprocal_rank = {}
        hits_at = {}

        for r in range(num_rel):
            mean_rank[r] = {}
            mean_reciprocal_rank[r] = {}
            hits_at[r] = {}
            for m in ["rhs"]:
                q = examples.clone()
                q[:, 1] = r
                ranks = self.get_ranking(q, filters[m], batch_size=batch_size)
                mean_rank[r][m] = torch.mean(ranks).item()
                mean_reciprocal_rank[r][m] = torch.mean(1. / ranks).item()
                hits_at[r][m] = torch.FloatTensor((list(map(
                    lambda x: torch.mean((ranks <= x).float()).item(),
                    (1, 10, 50, 100)
                ))))

        return mean_rank, mean_reciprocal_rank, hits_at

    def compute_roc(self, examples, neg_trp=[], batch_size=500, save_path=None, num_rel=5):
        plt.figure(figsize=(6, 18))
        for m in ["rhs"]:
            q = examples.clone()
            if m == "rhs":
                negs = torch.from_numpy(np.array(neg_trp).astype("int64"))
                roauc, prauc, max_f1 = self.get_roc(q, batch_size=batch_size, direc=m, negs=negs, save_path=save_path, num_rel=num_rel)
                
        return roauc, prauc, max_f1

    def compute_rel_acc(self, examples, batch_size=500, save_path=None, num_rel=5):
        q = examples.clone()
        mac_f1, mic_f1, acc = self.get_rel_acc(q, batch_size=batch_size, save_path=save_path, num_rel=num_rel)

        return mac_f1, mic_f1, acc
    

def get_accuracy(y_true, y_pred):
    accuracy = defaultdict(int)
    counter = defaultdict(int)
    for true, pred in zip(y_true, y_pred):
        if true == pred:
            accuracy[true]+=1
        else:
            accuracy[true]+=0
        counter[true]+=1
    accuracy = {i:k/j for i, (k, j) in enumerate(zip(accuracy.values(),counter.values()))}
    return accuracy