
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from models.base import KGModel
from utils.euclidean import givens_rotations, euc_sqdistance
from utils.hyperbolic import mobius_add, expmap0, logmap0, project, hyp_distance_multi_c

HYP_MODELS = ["GeOKG", "GeOKG_go"]


class BaseH(KGModel):

    def __init__(self, args):
        super(BaseH, self).__init__(args.sizes, args.rank, args.dropout, args.gamma, args.dtype, args.bias,
                                    args.init_size)
        if args.prtd:
            prtd_term = torch.tensor(np.load(args.prtd+'/entity_embedding.npy'), dtype=self.data_type)
            prot = self.init_size * torch.randn((self.sizes[0]-prtd_term.shape[0], self.rank), dtype=self.data_type)
            entity_embs = torch.vstack((prtd_term, prot))
            self.entity = nn.Embedding.from_pretrained(entity_embs, freeze=False)
            prtd_bh = torch.tensor(np.load(args.prtd+'/bh_embedding.npy'), dtype=self.data_type)
            prot = self.init_size * torch.randn((self.sizes[0]-prtd_bh.shape[0], 1), dtype=self.data_type)
            bh_embs = torch.vstack((prtd_bh, prot))
            self.bh = nn.Embedding.from_pretrained(bh_embs, freeze=False)
            prtd_bt = torch.tensor(np.load(args.prtd+'/bt_embedding.npy'), dtype=self.data_type)
            prot = self.init_size * torch.randn((self.sizes[0]-prtd_bt.shape[0], 1), dtype=self.data_type)
            bt_embs = torch.vstack((prtd_bt, prot))
            self.bt = nn.Embedding.from_pretrained(bt_embs, freeze=False)
            
            prtd_rel = torch.tensor(np.load(args.prtd+'/relation_embedding.npy'), dtype=self.data_type)
            prot = self.init_size * torch.randn((self.sizes[1]-prtd_rel.shape[0], self.rank), dtype=self.data_type)
            rel_embs = torch.vstack((prot, prtd_rel))
            self.rel = nn.Embedding.from_pretrained(rel_embs, freeze=False)
            prtd_rel_diag = torch.tensor(np.load(args.prtd+'/diag_relation_embedding.npy'), dtype=self.data_type)
            prot = 2 * torch.randn((self.sizes[1]-prtd_rel_diag.shape[0], self.rank), dtype=self.data_type) - 1.0
            rel_diag_embs = torch.vstack((prot, prtd_rel_diag))
            self.rel_diag = nn.Embedding.from_pretrained(rel_diag_embs, freeze=False)
            prtd_rel_diag1 = torch.tensor(np.load(args.prtd+'/diag1_relation_embedding.npy'), dtype=self.data_type)
            prot = 2 * torch.randn((self.sizes[1]-prtd_rel_diag1.shape[0], self.rank), dtype=self.data_type) - 1.0
            rel_diag1_embs = torch.vstack((prot, prtd_rel_diag1))
            self.rel_diag1 = nn.Embedding.from_pretrained(rel_diag1_embs, freeze=False)
            prtd_rel_diag2 = torch.tensor(np.load(args.prtd+'/diag2_relation_embedding.npy'), dtype=self.data_type)
            prot = 2 * torch.randn((self.sizes[1]-prtd_rel_diag2.shape[0], self.rank), dtype=self.data_type) - 1.0
            rel_diag2_embs = torch.vstack((prot, prtd_rel_diag2))
            self.rel_diag2 = nn.Embedding.from_pretrained(rel_diag2_embs, freeze=False)
            
        else:
            self.entity.weight.data = self.init_size * torch.randn((self.sizes[0], self.rank), dtype=self.data_type)       
            self.rel.weight.data = self.init_size * torch.randn((self.sizes[1], self.rank), dtype=self.data_type)
            self.rel_diag = nn.Embedding(self.sizes[1], self.rank)
            self.rel_diag1 = nn.Embedding(self.sizes[1], self.rank)
            self.rel_diag2 = nn.Embedding(self.sizes[1], self.rank)
            self.rel_diag.weight.data = 2 * torch.rand((self.sizes[1], self.rank), dtype=self.data_type) - 1.0
            self.rel_diag1.weight.data = 2 * torch.rand((self.sizes[1], self.rank), dtype=self.data_type) - 1.0
            self.rel_diag2.weight.data = 2 * torch.rand((self.sizes[1], self.rank), dtype=self.data_type) - 1.0
        
        self.multi_c = args.multi_c
        if self.multi_c:
            c_init = torch.ones((self.sizes[1], 1), dtype=self.data_type)
            c_init1 = torch.ones((self.sizes[1], 1), dtype=self.data_type)
            c_init2 = torch.ones((self.sizes[1], 1), dtype=self.data_type)
        else:
            c_init = torch.ones((1, 1), dtype=self.data_type)
            c_init1 = torch.ones((1, 1), dtype=self.data_type)
            c_init2 = torch.ones((1, 1), dtype=self.data_type)
        self.c = nn.Parameter(c_init, requires_grad=True)
        self.c1= nn.Parameter(c_init1, requires_grad=True)
        self.c2 = nn.Parameter(c_init2, requires_grad=True)
        
    def get_rhs(self, queries, eval_mode):
        if eval_mode:
            return self.entity.weight, self.bt.weight
        else:
            return self.entity(queries[:, 2]), self.bt(queries[:, 2])

    def similarity_score(self, lhs_e, rhs_e, eval_mode):
        lhs_e, c = lhs_e
        return - hyp_distance_multi_c(lhs_e, rhs_e, c, eval_mode) ** 2
    
class GeOKG(BaseH):

    def __init__(self, args):
        super(GeOKG, self).__init__(args)
        self.context_vec = nn.Embedding(self.sizes[1], self.rank)
        self.context_vec.weight.data = self.init_size * torch.randn((self.sizes[1], self.rank), dtype=self.data_type)
        self.act = nn.Softmax(dim=1)
        if args.dtype == "double":
            self.scale = torch.Tensor([1. / np.sqrt(self.rank)]).double().cuda()
        else:
            self.scale = torch.Tensor([1. / np.sqrt(self.rank)]).cuda()
            
    def score(self, lhs, rhs, eval_mode):
        if eval_mode:
            lhs_c, lhs_biases = lhs
            lhs_e, c_list = lhs_c
            rhs_e, rhs_biases = rhs
            score_list = []
            for i, c in enumerate(c_list):
                rhs_c = expmap0(rhs_e, c)
                lhs_c = (lhs_e[i], c)
                score_row = self.similarity_score(lhs_c, rhs_c, eval_mode)
                score_list.append(score_row)
            score = torch.stack(score_list, dim=0)
            score = score.squeeze(1)
            return lhs_biases + rhs_biases.t() + score
        else:
            lhs_e, lhs_biases = lhs
            rhs_e, rhs_biases = rhs
            score = self.similarity_score(lhs_e, rhs_e, eval_mode)
            if self.bias == 'constant':
                return self.gamma.item() + score
            elif self.bias == 'learn':
                return lhs_biases + rhs_biases + score
            else:
                return score
            
    def get_rhs(self, queries, eval_mode):
        if eval_mode:
            return self.entity.weight, self.bt.weight
        else:
            c = F.softplus(self.c[queries[:, 1]])
            tail = self.entity(queries[:, 2])
            rhs = expmap0(tail, c)
            return rhs, self.bt(queries[:, 2])

    def get_queries(self, queries):
        
        u = self.entity(queries[:, 0])
        r = queries[:, 1]

        c1 = F.softplus(self.c1[r])
        head1 = expmap0(u, c1)
        rel1 = self.rel(r)
        rel1 = expmap0(rel1, c1)
        head1 = project(mobius_add(head1, rel1, c1), c1)
        hyp1 = givens_rotations(self.rel_diag1(r), head1)

        c2 = F.softplus(self.c2[r])
        head2 = expmap0(u, c2)
        rel2= self.rel(r)
        rel2 = expmap0(rel2, c2)
        head2 = project(mobius_add(head2, rel2, c2), c2)
        hyp2 = givens_rotations(self.rel_diag2(r), head2)
        
        hyp1=logmap0(hyp1,c1).view((-1, 1, self.rank))
        hyp2=logmap0(hyp2,c2).view((-1, 1, self.rank))
        
        head3 = u
        rot_mat = self.rel_diag(r)
        euc = givens_rotations(rot_mat, head3).view((-1, 1, self.rank))
        
        c = F.softplus(self.c[r])
        cands = torch.cat([hyp1,hyp2,euc], dim=1)
        context_vec = self.context_vec(r).view((-1, 1, self.rank))
        att_weights = torch.sum(context_vec * cands * self.scale, dim=-1, keepdim=True)
        att_weights = self.act(att_weights)
        att_q = torch.sum(att_weights * cands, dim=1)
        
        lhs = expmap0(att_q, c)
        rel = self.rel(r)
        rel_c = expmap0(rel, c)
        res = project(mobius_add(lhs, rel_c, c), c)
        
        return (res, c), self.bh(queries[:, 0])
    
    def get_queries_lp(self, queries, r):
        
        u = self.entity(queries[:, 0])

        c1 = F.softplus(self.c1[r])
        head1 = expmap0(u, c1)
        rel1 = self.rel(r)
        rel1 = expmap0(rel1, c1)
        head1 = project(mobius_add(head1, rel1, c1), c1)
        hyp1 = givens_rotations(self.rel_diag1(r), head1)

        c2 = F.softplus(self.c2[r])
        head2 = expmap0(u, c2)
        rel2= self.rel(r)
        rel2 = expmap0(rel2, c2)
        head2 = project(mobius_add(head2, rel2, c2), c2)
        hyp2 = givens_rotations(self.rel_diag2(r), head2)
        
        hyp1=logmap0(hyp1,c1).view((-1, 1, self.rank))
        hyp2=logmap0(hyp2,c2).view((-1, 1, self.rank))
        
        head3 = u
        rot_mat = self.rel_diag(r)
        euc = givens_rotations(rot_mat, head3).view((-1, 1, self.rank))
        
        c = F.softplus(self.c[r])
        cands = torch.cat([hyp1,hyp2,euc], dim=1)
        context_vec = self.context_vec(r).view((-1, 1, self.rank))
        att_weights = torch.sum(context_vec * cands * self.scale, dim=-1, keepdim=True)
        att_weights = self.act(att_weights)
        att_q = torch.sum(att_weights * cands, dim=1)
        
        lhs = expmap0(att_q, c)
        rel = self.rel(r)
        rel_c = expmap0(rel, c)
        res = project(mobius_add(lhs, rel_c, c), c)
        
        return (res, c), self.bh(queries[:, 0])

    
class GeOKG_go(BaseH):

    def __init__(self, args):
        super(GeOKG_go, self).__init__(args)
        self.context_vec = nn.Embedding(self.sizes[1], self.rank)
        self.context_vec.weight.data = self.init_size * torch.randn((self.sizes[1], self.rank), dtype=self.data_type)
        self.act = nn.Softmax(dim=1)
        if args.dtype == "double":
            self.scale = torch.Tensor([1. / np.sqrt(self.rank)]).double().cuda()
        else:
            self.scale = torch.Tensor([1. / np.sqrt(self.rank)]).cuda()

    def get_queries(self, queries):
        
        u = self.entity(queries[:, 0])
        r = queries[:, 1]

        c1 = F.softplus(self.c1[r])
        head1 = expmap0(u, c1)
        rel1 = self.rel(r)
        rel1 = expmap0(rel1, c1)
        head1 = project(mobius_add(head1, rel1, c1), c1)
        hyp1 = givens_rotations(self.rel_diag1(r), head1)

        c2 = F.softplus(self.c2[r])
        head2 = expmap0(u, c2)
        rel2= self.rel(r)
        rel2 = expmap0(rel2, c2)
        head2 = project(mobius_add(head2, rel2, c2), c2)
        hyp2 = givens_rotations(self.rel_diag2(r), head2)
        
        hyp1=logmap0(hyp1,c1).view((-1, 1, self.rank))
        hyp2=logmap0(hyp2,c2).view((-1, 1, self.rank))
        
        head3 = u
        rot_mat = self.rel_diag(r)
        euc = givens_rotations(rot_mat, head3).view((-1, 1, self.rank))
        
        c = F.softplus(self.c[r])
        cands = torch.cat([hyp1,hyp2,euc], dim=1)
        context_vec = self.context_vec(r).view((-1, 1, self.rank))
        att_weights = torch.sum(context_vec * cands * self.scale, dim=-1, keepdim=True)
        att_weights = self.act(att_weights)
        att_q = torch.sum(att_weights * cands, dim=1)
        
        lhs = expmap0(att_q, c)
        rel = self.rel(r)
        rel_c = expmap0(rel, c)
        res = project(mobius_add(lhs, rel_c, c), c)
        
        return (res, c), self.bh(queries[:, 0])
    
    def get_queries_lp(self, queries, r):
        
        u = self.entity(queries[:, 0])

        c1 = F.softplus(self.c1[r])
        head1 = expmap0(u, c1)
        rel1 = self.rel(r)
        rel1 = expmap0(rel1, c1)
        head1 = project(mobius_add(head1, rel1, c1), c1)
        hyp1 = givens_rotations(self.rel_diag1(r), head1)

        c2 = F.softplus(self.c2[r])
        head2 = expmap0(u, c2)
        rel2= self.rel(r)
        rel2 = expmap0(rel2, c2)
        head2 = project(mobius_add(head2, rel2, c2), c2)
        hyp2 = givens_rotations(self.rel_diag2(r), head2)
        
        hyp1=logmap0(hyp1,c1).view((-1, 1, self.rank))
        hyp2=logmap0(hyp2,c2).view((-1, 1, self.rank))
        
        head3 = u
        rot_mat = self.rel_diag(r)
        euc = givens_rotations(rot_mat, head3).view((-1, 1, self.rank))
        
        c = F.softplus(self.c[r])
        cands = torch.cat([hyp1,hyp2,euc], dim=1)
        context_vec = self.context_vec(r).view((-1, 1, self.rank))
        att_weights = torch.sum(context_vec * cands * self.scale, dim=-1, keepdim=True)
        att_weights = self.act(att_weights)
        att_q = torch.sum(att_weights * cands, dim=1)
        
        lhs = expmap0(att_q, c)
        rel = self.rel(r)
        rel_c = expmap0(rel, c)
        res = project(mobius_add(lhs, rel_c, c), c)
        
        return (res, c), self.bh(queries[:, 0])