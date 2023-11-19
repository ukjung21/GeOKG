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

from models.base import KGModel
from utils.euclidean import euc_sqdistance, givens_rotations, givens_reflection

EUC_MODELS = ["TransE", "CP", "MurE"]


class BaseE(KGModel):

    def __init__(self, args):
        super(BaseE, self).__init__(args.sizes, args.rank, args.dropout, args.gamma, args.dtype, args.bias,
                                    args.init_size)
        # if args.prtd:
        #     prtd_term = torch.tensor(np.load(args.prtd+'/entity_embedding.npy'), dtype=self.data_type)
        #     prot = self.init_size * torch.randn((self.sizes[0]-prtd_term.shape[0], self.rank), dtype=self.data_type)
        #     entity_embs = torch.vstack((prtd_term, prot))
        #     self.entity = nn.Embedding.from_pretrained(entity_embs, freeze=False)
        #     prtd_bh = torch.tensor(np.load(args.prtd+'/bh_embedding.npy'), dtype=self.data_type)
        #     prot = self.init_size * torch.randn((self.sizes[0]-prtd_bh.shape[0], 1), dtype=self.data_type)
        #     bh_embs = torch.vstack((prtd_bh, prot))
        #     self.bh = nn.Embedding.from_pretrained(bh_embs, freeze=False)
        #     prtd_bt = torch.tensor(np.load(args.prtd+'/bt_embedding.npy'), dtype=self.data_type)
        #     prot = self.init_size * torch.randn((self.sizes[0]-prtd_bt.shape[0], 1), dtype=self.data_type)
        #     bt_embs = torch.vstack((prtd_bt, prot))
        #     self.bt = nn.Embedding.from_pretrained(bt_embs, freeze=False)
            
        #     prtd_rel = torch.tensor(np.load(args.prtd+'/relation_embedding.npy'), dtype=self.data_type)
        #     prot = self.init_size * torch.randn((self.sizes[1]-prtd_rel.shape[0], self.rank), dtype=self.data_type)
        #     rel_embs = torch.vstack((prot, prtd_rel))
        #     self.rel = nn.Embedding.from_pretrained(rel_embs, freeze=False)
        #     prtd_rel_diag = torch.tensor(np.load(args.prtd+'/diag_relation_embedding.npy'), dtype=self.data_type)
        #     prot = 2 * torch.randn((self.sizes[1]-prtd_rel_diag.shape[0], self.rank), dtype=self.data_type) - 1.0
        #     rel_diag_embs = torch.vstack((prot, prtd_rel_diag))
        #     self.rel_diag = nn.Embedding.from_pretrained(rel_diag_embs, freeze=False)
            
        # else:
        self.entity.weight.data = self.init_size * torch.randn((self.sizes[0], self.rank), dtype=self.data_type)       
        self.rel.weight.data = self.init_size * torch.randn((self.sizes[1], self.rank), dtype=self.data_type)
        self.rel_diag = nn.Embedding(self.sizes[1], self.rank)
        self.rel_diag.weight.data = 2 * torch.rand((self.sizes[1], self.rank), dtype=self.data_type) - 1.0

    # def get_rhs(self, queries, eval_mode):
    #     if eval_mode:
    #         return self.entity.weight, self.bt.weight
    #     else:
    #         return self.entity(queries[:, 2]), self.bt(queries[:, 2])

    # def similarity_score(self, lhs_e, rhs_e, eval_mode):
    #     if self.sim == "dot":
    #         if eval_mode:
    #             score = lhs_e @ rhs_e.transpose(0, 1)
    #         else:
    #             score = torch.sum(lhs_e * rhs_e, dim=-1, keepdim=True)
    #     else:
    #         score = - euc_sqdistance(lhs_e, rhs_e, eval_mode)
    #     return score

class HAKE(BaseE):
    def __init__(self, args, modulus_weight=1.0, phase_weight=0.5):
        super(HAKE, self).__init__(args)
        self.epsilon = 2.0

        self.gamma = nn.Parameter(
            torch.Tensor([6.0]),
            requires_grad=False
        )

        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / args.rank]),
            requires_grad=False
        )

        self.entity = nn.Embedding(args.sizes[0], args.rank * 2)
        nn.init.uniform_(
            tensor=self.entity.weight.data,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

        self.rel = nn.Embedding(args.sizes[0], args.rank * 3)
        nn.init.uniform_(
            tensor=self.rel.weight.data,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

        nn.init.ones_(
            tensor=self.rel.weight.data[:, args.rank:2 * args.rank]
        )

        nn.init.zeros_(
            tensor=self.rel.weight.data[:, 2 * args.rank:3 * args.rank]
        )

        self.phase_weight = nn.Parameter(torch.Tensor([[phase_weight * self.embedding_range.item()]]))
        self.modulus_weight = nn.Parameter(torch.Tensor([[modulus_weight]]))

        self.pi = 3.14159262358979323846
        
    def get_rhs(self, queries, eval_mode):
        if eval_mode:
            return queries, self.bt.weight
        else:
            return queries, self.bt(queries[:, 2])
        
    def euc_sqdistance(x, y, eval_mode=False):
        x2 = torch.sum(x * x, dim=-1, keepdim=True)
        y2 = torch.sum(y * y, dim=-1, keepdim=True)
        if eval_mode:
            y2 = y2.t()
            xy = x @ y.t()
        else:
            # assert x.shape[0] == y.shape[0]
            xy = torch.sum(x * y, dim=-1, keepdim=True)
        return x2 + y2 - 2 * xy

    def similarity_score(self, lhs_e, rhs_e, eval_mode):
        
        head = lhs_e
        queries = rhs_e
        # print('queries', queries.shape)
        
        if eval_mode:
            rel = self.rel.weight
            tail = self.entity.weight
            phase_head, mod_head = torch.chunk(head, 2, dim=1)
            phase_relation, mod_relation, bias_relation = torch.chunk(rel, 3, dim=1)
            phase_tail, mod_tail = torch.chunk(tail, 2, dim=1)

            phase_head = phase_head / (self.embedding_range.item() / self.pi)
            phase_relation = phase_relation / (self.embedding_range.item() / self.pi)
            phase_tail = phase_tail / (self.embedding_range.item() / self.pi)
            
            x = phase_head + phase_relation[queries[:, 1]]
            y = phase_tail
            
            x2 = torch.sum(x * x, dim=-1, keepdim=True)
            y2 = torch.sum(y * y, dim=-1, keepdim=True)
            
            y2 = y2.t()
            xy = x @ y.t()

            phase_score = x2 + y2 - 2 * xy

            mod_relation = torch.abs(mod_relation)
            bias_relation = torch.clamp(bias_relation, max=1)
            indicator = (bias_relation < -mod_relation)
            bias_relation[indicator] = -mod_relation[indicator]
            
            z = mod_head * (mod_relation[queries[:, 1]] + bias_relation[queries[:, 1]])
            w = mod_tail * (1 - bias_relation)
            
            z2 = torch.sum(z * z, dim=-1, keepdim=True)
            w2 = torch.sum(w * w, dim=-1, keepdim=True)
            
            w2 = w2.t()
            zw = z @ w.t()

            r_score = z2 + w2 - 2 * zw

            phase_score = torch.abs(torch.sin(torch.sqrt(phase_score) / 2)) * self.phase_weight
            r_score = torch.sqrt(r_score) * self.modulus_weight
            
        else:
            rel = self.rel(queries[:,1])
            tail = self.entity(queries[:, 2])
            phase_head, mod_head = torch.chunk(head, 2, dim=1)
            phase_relation, mod_relation, bias_relation = torch.chunk(rel, 3, dim=1)
            phase_tail, mod_tail = torch.chunk(tail, 2, dim=1)
            # print('phase_head', phase_head.shape)
            phase_head = phase_head / (self.embedding_range.item() / self.pi)
            phase_relation = phase_relation / (self.embedding_range.item() / self.pi)
            phase_tail = phase_tail / (self.embedding_range.item() / self.pi)
            # print('phase_head', phase_head)
            x = phase_head + phase_relation
            y = phase_tail
            
            x2 = torch.sum(x * x, dim=-1, keepdim=True)
            y2 = torch.sum(y * y, dim=-1, keepdim=True)
            xy = torch.sum(x * y, dim=-1, keepdim=True)

            phase_score = x2 + y2 - 2 * xy

            # phase_score = (phase_head + phase_relation) - phase_tail

            mod_relation = torch.abs(mod_relation)
            bias_relation = torch.clamp(bias_relation, max=1)
            indicator = (bias_relation < -mod_relation)
            bias_relation[indicator] = -mod_relation[indicator]
            
            z = mod_head * (mod_relation[queries[:, 1]] + bias_relation[queries[:, 1]])
            w = mod_tail * (1 - bias_relation)
            
            z2 = torch.sum(z * z, dim=-1, keepdim=True)
            w2 = torch.sum(w * w, dim=-1, keepdim=True)
            zw = torch.sum(z * w, dim=-1, keepdim=True)

            r_score = z2 + w2 - 2 * zw

            phase_score = torch.abs(torch.sin(torch.sqrt(phase_score) / 2)) * self.phase_weight
            r_score = torch.sqrt(r_score) * self.modulus_weight
            
            # print('r_score', r_score)

            # r_score = mod_head * (mod_relation + bias_relation) - mod_tail * (1 - bias_relation)

            # phase_score = torch.abs(torch.sin(phase_score / 2)) * self.phase_weight
            # r_score = r_score * self.modulus_weight

        return self.gamma.item() - (phase_score + r_score)


    def func(self, head, rel, tail, batch_type):
        phase_head, mod_head = torch.chunk(head, 2, dim=2)
        phase_relation, mod_relation, bias_relation = torch.chunk(rel, 3, dim=2)
        phase_tail, mod_tail = torch.chunk(tail, 2, dim=2)

        phase_head = phase_head / (self.embedding_range.item() / self.pi)
        phase_relation = phase_relation / (self.embedding_range.item() / self.pi)
        phase_tail = phase_tail / (self.embedding_range.item() / self.pi)

        if batch_type == BatchType.HEAD_BATCH:
            phase_score = phase_head + (phase_relation - phase_tail)
        else:
            phase_score = (phase_head + phase_relation) - phase_tail

        mod_relation = torch.abs(mod_relation)
        bias_relation = torch.clamp(bias_relation, max=1)
        indicator = (bias_relation < -mod_relation)
        bias_relation[indicator] = -mod_relation[indicator]

        r_score = mod_head * (mod_relation + bias_relation) - mod_tail * (1 - bias_relation)

        phase_score = torch.sum(torch.abs(torch.sin(phase_score / 2)), dim=2) * self.phase_weight
        r_score = torch.norm(r_score, dim=2) * self.modulus_weight

        return self.gamma.item() - (phase_score + r_score)
    
    def euc_sqdistance(x, y, eval_mode=False):
        x2 = torch.sum(x * x, dim=-1, keepdim=True)
        y2 = torch.sum(y * y, dim=-1, keepdim=True)
        if eval_mode:
            y2 = y2.t()
            xy = x @ y.t()
        else:
            # assert x.shape[0] == y.shape[0]
            xy = torch.sum(x * y, dim=-1, keepdim=True)
        return x2 + y2 - 2 * xy
    
    def get_queries(self, queries):     
        lhs_e = self.entity(queries[:, 0])
        lhs_biases = self.bh(queries[:, 0])

        return lhs_e, lhs_biases
        
    def get_queries_lp(self, queries, r):
        head = self.entity(queries[:, 0])
        rel = self.rel(r)
        tail = self.entity(queries[:, 2])
        phase_head, mod_head = torch.chunk(head, 2, dim=1)
        phase_relation, mod_relation, bias_relation = torch.chunk(rel, 3, dim=1)
        phase_tail, mod_tail = torch.chunk(tail, 2, dim=1)

        phase_head = phase_head / (self.embedding_range.item() / self.pi)
        phase_relation = phase_relation / (self.embedding_range.item() / self.pi)
        phase_tail = phase_tail / (self.embedding_range.item() / self.pi)

        phase_score = (phase_head + phase_relation) - phase_tail

        mod_relation = torch.abs(mod_relation)
        bias_relation = torch.clamp(bias_relation, max=1)
        indicator = (bias_relation < -mod_relation)
        bias_relation[indicator] = -mod_relation[indicator]

        r_score = mod_head * (mod_relation + bias_relation) - mod_tail * (1 - bias_relation)

        phase_score = torch.sum(torch.abs(torch.sin(phase_score / 2)), dim=1) * self.phase_weight
        r_score = torch.norm(r_score, dim=1) * self.modulus_weight

        return self.gamma.item() - (phase_score + r_score)
    
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
        # (lhs_e, lhs_biases), att_wts = self.get_queries(queries)
        lhs_e, lhs_biases = self.get_queries(queries)
        rhs_e, rhs_biases = self.get_rhs(queries, eval_mode)
        predictions = self.score((lhs_e, lhs_biases), (rhs_e, rhs_biases), eval_mode)
        factors = self.get_factors(queries)
        return predictions, factors

    def get_ranking(self, queries, filters, batch_size=1000):
        ranks = torch.ones(len(queries))
        with torch.no_grad():
            b_begin = 0
            # candidates = self.get_rhs(queries, eval_mode=True)
            while b_begin < len(queries):
                these_queries = queries[b_begin:b_begin + batch_size].cuda()

                # q, att_wts = self.get_queries(these_queries)
                q = self.get_queries(these_queries)
                rhs = self.get_rhs(these_queries, eval_mode=False)
                candidates = self.get_rhs(these_queries, eval_mode=True)

                scores = self.score(q, candidates, eval_mode=True)
                targets = self.score(q, rhs, eval_mode=False)
                # print('scores', scores.shape)
                # print('targets', targets.shape)

                for i, query in enumerate(these_queries):
                    filter_out = filters[(query[0].item(), query[1].item())]
                    filter_out += [queries[b_begin + i, 2].item()]
                    scores[i, torch.LongTensor(filter_out)] = -1e6
                ranks[b_begin:b_begin + batch_size] += torch.sum(
                    (scores >= targets).float(), dim=1
                ).cpu()
                #     scores[i, torch.LongTensor(filter_out)] = 1e6
                # ranks[b_begin:b_begin + batch_size] += torch.sum(
                #     (scores <= targets).float(), dim=1
                # ).cpu()
                b_begin += batch_size
        # return ranks, att_wts
        return ranks
    
    def curve_plot(self, fprs, tprs, direc, metric):
        if metric == 'roc':
            # ROC Curve를 plot 곡선으로 그림.
            if direc =='rhs':
                plt.plot(fprs , tprs, label='ROC')
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
                plt.plot(fprs , tprs, label='P-R')
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
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.legend()
            
        
    def get_roc(
            self, queries: torch.Tensor, # 500 * (lhs, rel, rhs)
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

                    q = self.get_queries(these_queries) # predict
                    # preds = [self.get_queries_lp(these_queries, torch.LongTensor([r]*these_queries.shape[0]).cuda()) for r in range(num_rel)]
                    # with open(file='/home/ukjung18/GIE1/GIE/GIE-master/LOG_DIR/curv.pickle', mode='wb') as f:
                    #     pickle.dump(preds, f)
                    var_rhs = []
                    neg_var_rhs = []
                    for r in range(num_rel):
                        these_queries[:,1] = torch.LongTensor([r]*these_queries.shape[0]).cuda()
                        these_negs[:,1] = torch.LongTensor([r]*these_queries.shape[0]).cuda()
                        var_rhs.append(self.get_rhs(these_queries, eval_mode=False))
                        neg_var_rhs.append(self.get_rhs(these_negs, eval_mode=False))
                    # rhs = self.get_rhs(these_queries, eval_mode=False) # target
                    # neg_rhs = self.get_rhs(these_negs, eval_mode=False)
                    
                    pos_scores = torch.max(torch.stack([self.score(q, rhs, eval_mode=False) for rhs in var_rhs]), dim=0)
                    neg_scores = torch.max(torch.stack([self.score(q, neg_rhs, eval_mode=False) for neg_rhs in neg_var_rhs]), dim=0)

                    pos_arr = np.append(pos_arr, pos_scores.values.cpu())
                    neg_arr = np.append(neg_arr, neg_scores.values.cpu())

                    b_begin += batch_size
                    bar.update(batch_size)

        # y_true = np.concatenate([np.ones(len(pos_arr)), np.zeros(len(neg_arr))])
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
        # ap_score = average_precision_score(y_true, y_pred)
        prauc = auc(recall, prec)

        # mac_f1 = f1_score(y_true, y_pred, average='macro')
        # mic_f1 = f1_score(y_true, y_pred, average='micro')
        numerator = 2 * recall * prec
        denom = recall + prec
        f1_scores = np.divide(numerator, denom, out=np.zeros_like(denom), where=(denom!=0))
        max_f1 = np.max(f1_scores)
        
        print('AUROC: ', roauc)
        print('AUPRC: ', prauc)
        print('Max F1: ', max_f1)
        # print('Macro F1: ', mac_f1)
        # print('Micro F1: ', mic_f1)
        # print('threshold for entity predict: ', best_thres)

        # if direc == 'lhs':
        #     plt.savefig(save_path+'/ent_predict.png', dpi=600)
        plt.savefig(save_path+'/ent_predict.png', dpi=600)

        return roauc, prauc, max_f1
        # return roauc, prauc, mac_f1, mic_f1
    
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
                    # these_negs = negs[b_begin:b_begin + batch_size].cuda()
                    
                    q = self.get_queries(these_queries) # predict
                    # preds = [self.get_queries_lp(these_queries, torch.LongTensor([r]*these_queries.shape[0]).cuda()) for r in range(num_rel)]
                    # with open(file='/home/ukjung18/GIE1/GIE/GIE-master/LOG_DIR/curv.pickle', mode='wb') as f:
                    #     pickle.dump(preds, f)
                    var_rhs = []
                    for r in range(num_rel):
                        these_queries[:,1] = torch.LongTensor([r]*these_queries.shape[0]).cuda()
                        var_rhs.append(self.get_rhs(these_queries, eval_mode=False))

                    # preds = [self.get_queries_lp(these_queries, torch.LongTensor([r]*these_queries.shape[0]).cuda()) for r in range(num_rel)]
                    
                    
                    # rhs = self.get_rhs(these_queries, eval_mode=False) # target
                    # neg_rhs = self.get_rhs(these_negs, eval_mode=False)

                    # pos_scores = self.score(q, rhs, eval_mode=False)
                    # neg_scores = self.score(q, neg_rhs, eval_mode=False)
                    p_scores = torch.stack([10.0+self.score(q, rhs, eval_mode=False) for rhs in var_rhs]).squeeze()
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
        # prec = precision_score(y_true, y_pred)
        # recall = recall_score(y_true, y_pred)
        mic_f1 = f1_score(y_true, y_pred, average='micro')
        wa_f1 = f1_score(y_true, y_pred, average='weighted')
        
        # conf_mat = confusion_matrix(y_true, y_pred)
        # kappa = quadratic_weighted_kappa(conf_mat)

        # print('Accuracy: ', accuracy)
        # print('Cohen\'s Kappa: ', kappa)
        # print('Macro Average F1 score: ', mac_f1)
        # print('Micro Average F1 score: ', mic_f1)
        # print('Weighted Average F1 score: ', wa_f1)

        # return accuracy, wa_f1
        return mic_f1, wa_f1, accuracy
    
    def compute_metrics(self, examples, filters, batch_size=500):
        mean_rank = {}
        mean_reciprocal_rank = {}
        hits_at = {}

        # for m in ["rhs", "lhs"]:
        for m in ["rhs"]:
            q = examples.clone()
            # if m == "lhs":
            #     tmp = torch.clone(q[:, 0])
            #     q[:, 0] = q[:, 2]
            #     q[:, 2] = tmp
            #     q[:, 1] += self.sizes[1] // 2
            ranks = self.get_ranking(q, filters[m], batch_size=batch_size)    
            # ranks, att_wts = self.get_ranking(q, filters[m], batch_size=batch_size)
            # if m == 'rhs':
            #     att_wts = att_wts.cpu().numpy()
            #     np.save(save_dir+'attention_weights.npy', att_wts)
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

        # for m in ["rhs", "lhs"]:
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
                # negs = np.array([t[2] for t in neg_trp])
                negs = torch.from_numpy(np.array(neg_trp).astype("int64"))
                roauc, prauc, max_f1 = self.get_roc(q, batch_size=batch_size, direc=m, negs=negs, save_path=save_path, num_rel=num_rel)
                
        # return rhs_roauc, lhs_roauc, rhs_prauc, lhs_prauc, rhs_f1, lhs_f1
        return roauc, prauc, max_f1

    def compute_rel_acc(self, examples, batch_size=500, save_path=None, num_rel=5):
        # plt.figure(figsize=(9, 18))
        q = examples.clone()
        # acc, waf1 = self.get_rel_acc(q, batch_size=batch_size, save_path=save_path, num_rel=num_rel)
        mic_f1, wa_f1, acc = self.get_rel_acc(q, batch_size=batch_size, save_path=save_path, num_rel=num_rel)

        # return acc, waf1
        return mic_f1, wa_f1, acc

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