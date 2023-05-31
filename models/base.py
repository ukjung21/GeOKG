
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
from sklearn.metrics import roc_curve, roc_auc_score, RocCurveDisplay, precision_recall_curve, PrecisionRecallDisplay, balanced_accuracy_score, average_precision_score, auc#
import pickle

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
                # return lhs_biases + rhs_biases.t() + score
                return lhs_biases + rhs_biases.t() + score
            else:
                # return lhs_biases + rhs_biases + score
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
                #     scores[i, torch.LongTensor(filter_out)] = -1e6
                # ranks[b_begin:b_begin + batch_size] += torch.sum(
                #     (scores >= targets).float(), dim=1
                # ).cpu()
                    scores[i, torch.LongTensor(filter_out)] = 1e6
                ranks[b_begin:b_begin + batch_size] += torch.sum(
                    (scores <= targets).float(), dim=1
                ).cpu()
                b_begin += batch_size
        return ranks
    
    def curve_plot(self, fprs, tprs, direc, metric):
        if metric == 'roc':
            # ROC Curve를 plot 곡선으로 그림.
            if direc =='rhs':
                plt.plot(fprs , tprs, label='ROC(rhs)')
            else:
                plt.plot(fprs , tprs, label='ROC(lhs)')
            plt.plot([0, 1], [0, 1], 'k--', label='Random')
            # 가운데 대각선 직선을 그림. 
            # plt.plot([0, 1], [0, 1], 'k--', label='Random')

            # FPR X 축의 Scale을 0.1 단위로 변경, X,Y 축명 설정등   
            start, end = plt.xlim()
            plt.xticks(np.round(np.arange(start, end, 0.1),2))
            plt.xlim(0,1)
            plt.ylim(0,1)
            plt.xlabel('FPR( 1 - Sensitivity )')
            plt.ylabel('TPR( Recall )')
            plt.legend()
        elif metric == 'pr':
             # ROC Curve를 plot 곡선으로 그림.
            if direc =='rhs':
                plt.plot(fprs , tprs, label='P-R(rhs)')
            else:
                plt.plot(fprs , tprs, label='P-R(lhs)')
            plt.plot([1, 0], [0, 1], 'k--', label='Random')
            # 가운데 대각선 직선을 그림. 
            # plt.plot([0, 1], [0, 1], 'k--', label='Random')

            # FPR X 축의 Scale을 0.1 단위로 변경, X,Y 축명 설정등   
            start, end = plt.xlim()
            plt.xticks(np.round(np.arange(start, end, 0.1),2))
            plt.xlim(0,1)
            plt.ylim(0,1)
            plt.xlabel('Precision')
            plt.ylabel('Recall')
            plt.legend()
            
        
    def get_roc(
            self, queries: torch.Tensor, # 50000 * (lhs, rel, rhs)
            batch_size: int = 1000,
            direc='rhs', negs=[], save_path=None
    ):
        with tqdm(total=queries.shape[0], unit='ex') as bar:
            bar.set_description(f'Evaluation')
            with torch.no_grad():
                b_begin = 0
                pos_arr=np.array([])
                neg_arr=np.array([])

                while b_begin < len(queries):
                    these_queries = queries[b_begin:b_begin + batch_size].cuda()
                    these_negs = negs[b_begin:b_begin + batch_size].cuda()

                    # q = self.get_queries(these_queries) # predict
                    preds = [self.get_queries_lp(these_queries, torch.LongTensor([r]*these_queries.shape[0]).cuda()) for r in range(12)]
                    # with open(file='/home/ukjung18/GIE1/GIE/GIE-master/LOG_DIR/curv.pickle', mode='wb') as f:
                    #     pickle.dump(preds, f)
                    
                    rhs = self.get_rhs(these_queries, eval_mode=False) # target
                    neg_rhs = self.get_rhs(these_negs, eval_mode=False)

                    # pos_scores = self.score(q, rhs, eval_mode=False)
                    # neg_scores = self.score(q, neg_rhs, eval_mode=False)
                    
                    # pos_scores = torch.max(torch.stack([self.score(q, rhs, eval_mode=False) for q in preds]), dim=0)
                    # neg_scores = torch.max(torch.stack([self.score(q, neg_rhs, eval_mode=False) for q in preds]), dim=0)
                    
                    pos_scores = torch.min(torch.stack([self.score(q, rhs, eval_mode=False) for q in preds]), dim=0)
                    neg_scores = torch.min(torch.stack([self.score(q, neg_rhs, eval_mode=False) for q in preds]), dim=0)

                    pos_arr = np.append(pos_arr, pos_scores.values.cpu())
                    neg_arr = np.append(neg_arr, neg_scores.values.cpu())

                    b_begin += batch_size
                    bar.update(batch_size)

        # y_true = np.concatenate([np.ones(len(pos_arr)), np.zeros(len(neg_arr))])
        y_true = np.concatenate([np.zeros(len(pos_arr)), np.ones(len(neg_arr))])
        y_scores = np.append(pos_arr, neg_arr)
        y_pred = y_scores
        
        plt.subplot(3,1,1)
        sns.histplot(data=pos_arr, kde=True, color='blue', stat="proportion")
        sns.histplot(data=neg_arr, kde=True, color='red', stat="proportion")
        # sns.distplot(pos_list, color='blue', label='rhs positive')
        # sns.distplot(neg_list, color='red', label='rhs negative')
        # plt.legend(title='entity')
        # with open(file='/home/ukjung18/GIE1/GIE/GIE-master/LOG_DIR/test_true.pickle', mode='wb') as f:
        #     pickle.dump(y_true, f)
        # with open(file='/home/ukjung18/GIE1/GIE/GIE-master/LOG_DIR/test_pred.pickle', mode='wb') as f:
        #     pickle.dump(y_pred, f)
        fprs, tprs, thres = roc_curve(y_true, y_pred)
        plt.subplot(3,1,2)
        self.curve_plot(fprs, tprs, direc, 'roc')
        
        prec, recall, thres = precision_recall_curve(y_true, y_pred)
        plt.subplot(3,1,3)
        self.curve_plot(prec, recall, direc, 'pr')
        
        roauc = roc_auc_score(y_true, y_pred)
        ap_score = average_precision_score(y_true, y_pred)
        prauc = auc(recall, prec)
        
        numerator = 2 * recall * prec
        denom = recall + prec
        f1_scores = np.divide(numerator, denom, out=np.zeros_like(denom), where=(denom!=0))
        max_f1 = np.max(f1_scores)
        
#         if direc == 'rhs':
#             with open(file=save_path+'/eval_metrics', mode='a') as f:
#                 f.write("\t Test PRAUC (rhs): {:.4f}".format(prauc))
#                 f.write("\t Test max F1 Score (rhs): {:.4f}".format(max_f1))
                
#         elif direc == 'lhs':
#             with open(file=save_path+'/eval_metrics', mode='a') as f:
#                 f.write("\t Test PRAUC (lhs): {:.4f}".format(prauc))
#                 f.write("\t Test max F1 Score (lhs): {:.4f}".format(max_f1))

        # logging.info("Average_precision_score: {}".format(ap_score))
        print('Average_precision_score:', ap_score)
        best_thres = thres[np.argmin(np.abs(fprs+tprs-1))]
        # logging.info("Best threshold for entity predict: {}".format(best_thres))
        print('threshold for entity predict: ', best_thres)

        # if direc == 'lhs':
        #     plt.savefig(save_path+'/ent_predict.png', dpi=600)
        plt.savefig(save_path+'/ent_predict.png', dpi=600)

        return roauc, prauc, max_f1
    
    def get_rel_acc(
        self, queries: torch.Tensor, # 50000 * (lhs, rel, rhs)
        batch_size: int = 1000,
        direc='rhs', save_path=None
    ):
#         with tqdm(total=queries.shape[0], unit='ex') as bar:
#             bar.set_description(f'Evaluation')
#             with torch.no_grad():
#                 b_begin = 0
#                 y_scores=list()
#                 y_true=queries[:, 1].numpy()

#                 while b_begin < len(queries):
#                     these_queries = queries[b_begin:b_begin + batch_size].cuda()
#                     # these_negs = negs[b_begin:b_begin + batch_size].cuda()

#                     # q = self.get_queries(these_queries) # predict
#                     preds = [self.get_queries_lp(these_queries, torch.LongTensor([r]*these_queries.shape[0]).cuda()) for r in range(6)]
                    
                    
#                     rhs = self.get_rhs(these_queries, eval_mode=False) # target
#                     # neg_rhs = self.get_rhs(these_negs, eval_mode=False)

#                     # pos_scores = self.score(q, rhs, eval_mode=False)
#                     # neg_scores = self.score(q, neg_rhs, eval_mode=False)
#                     p_scores = torch.stack([10.0-self.score(q, rhs, eval_mode=False) for q in preds]).squeeze()
#                     softmax = torch.nn.Softmax(dim=0)
#                     pos_scores = softmax(p_scores)
                    

#                     y_scores.append(pos_scores.cpu())

#                     b_begin += batch_size
#                     bar.update(batch_size)
        y_true=queries[:, 1].numpy()
        queries = queries.cuda()
        preds = [self.get_queries_lp(queries, torch.LongTensor([r]*queries.shape[0]).cuda()) for r in range(6)]
                    
        rhs = self.get_rhs(queries, eval_mode=False) # target
        
        p_scores = torch.stack([10.0-self.score(q, rhs, eval_mode=False) for q in preds]).squeeze()
        softmax = torch.nn.Softmax(dim=0)
        pos_scores = softmax(p_scores)
        
        y_scores = pos_scores.cpu().detach().numpy().T
        y_pred = np.argmax(y_scores, axis=1)
        
        with open(file=save_path+'/rel_score.pickle', mode='wb') as f:
            pickle.dump(y_scores, f)
        with open(file=save_path+'/rel_pred.pickle', mode='wb') as f:
            pickle.dump(y_pred, f)
        
        accuracy = np.sum(y_true == y_pred) / y_true.shape[0]


#         fprs, tprs, thres = roc_curve(y_true, y_pred)
#         plt.subplot(2,1,1)
#         self.curve_plot(fprs, tprs, direc, 'roc')
        
#         prec, recall, thres = precision_recall_curve(y_true, y_pred)
#         plt.subplot(2,1,2)
#         self.curve_plot(prec, recall, direc, 'pr')
        
#         roauc = roc_auc_score(y_true, y_pred)
#         ap_score = average_precision_score(y_true, y_pred)
#         prauc = auc(recall, prec)
        
#         numerator = 2 * recall * prec
#         denom = recall + prec
#         f1_scores = np.divide(numerator, denom, out=np.zeros_like(denom), where=(denom!=0))
#         max_f1 = np.max(f1_scores)
        
#         print('Average_precision_score:', ap_score)
#         best_thres = thres[np.argmin(np.abs(fprs+tprs-1))]

#         print('threshold for entity predict: ', best_thres)

#         plt.savefig(save_path+'/ent_predict.png', dpi=600)

        return accuracy
    
    def compute_metrics(self, examples, filters, batch_size=500):
        mean_rank = {}
        mean_reciprocal_rank = {}
        hits_at = {}

        # for m in ["rhs", "lhs"]:
        for m in ["rhs"]:
            q = examples.clone()
            if m == "lhs":
                tmp = torch.clone(q[:, 0])
                q[:, 0] = q[:, 2]
                q[:, 2] = tmp
                q[:, 1] += self.sizes[1] // 2
            ranks = self.get_ranking(q, filters[m], batch_size=batch_size)
            mean_rank[m] = torch.mean(ranks).item()
            mean_reciprocal_rank[m] = torch.mean(1. / ranks).item()
            hits_at[m] = torch.FloatTensor((list(map(
                lambda x: torch.mean((ranks <= x).float()).item(),
                (1, 3, 10)
            ))))

        return mean_rank, mean_reciprocal_rank, hits_at

    def compute_roc(self, examples, neg_trp=[], neg_inv_trp=[], batch_size=500, save_path=None):
        plt.figure(figsize=(6, 18))
        for m in ["rhs"]:
            q = examples.clone()
            if m == "rhs":
                # negs = np.array([t[2] for t in neg_trp])
                negs = torch.from_numpy(np.array(neg_trp).astype("int64"))
                rhs_roauc, rhs_prauc, rhs_f1 = self.get_roc(q, batch_size=batch_size, direc=m, negs=negs, save_path=save_path)
            else:
                tmp = torch.clone(q[:, 0])
                q[:, 0] = q[:, 2]
                q[:, 2] = tmp
                q[:, 1] += self.sizes[1] // 2
                # negs = np.array([h[0] for h in neg_inv_trp])
                tmp_neg = neg_inv_trp[:, 0].copy()
                neg_inv_trp[:, 0] = q[:, 2]
                neg_inv_trp[:, 2] = tmp_neg
                neg_inv_trp[:, 1] += self.sizes[1] // 2
                negs = torch.from_numpy(np.array(neg_inv_trp).astype("int64"))
                lhs_roauc, lhs_prauc, lhs_f1 = self.get_roc(q, batch_size=batch_size, direc=m, negs=negs, save_path=save_path)

        # return rhs_roauc, lhs_roauc, rhs_prauc, lhs_prauc, rhs_f1, lhs_f1
        return rhs_roauc, rhs_prauc, rhs_f1

    def compute_rel_acc(self, examples, batch_size=500, save_path=None):
        # plt.figure(figsize=(9, 18))
        q = examples.clone()
        acc = self.get_rel_acc(q, batch_size=batch_size, direc='rhs', save_path=save_path)

        return acc