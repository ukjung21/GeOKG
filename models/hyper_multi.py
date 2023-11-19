
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from models.base import KGModel
from utils.euclidean import givens_rotations, givens_reflection, euc_sqdistance
from utils.hyperbolic import mobius_add, expmap0,logmap0, project, hyp_distance_multi_c

HYP_MODELS = ["GIE_rot", "GIE_ref", "GIE_trans", "TransH", "MurP", "ATT2_rot", "ATT2_ref", "ATT2_trans"]


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
    
class GIE_rot(BaseH):

    def __init__(self, args):
        super(GIE_rot, self).__init__(args)
        # self.rel_diag = nn.Embedding(self.sizes[1], self.rank)
        # self.rel_diag1 = nn.Embedding(self.sizes[1], self.rank)
        # self.rel_diag2 = nn.Embedding(self.sizes[1], self.rank)
        # self.rel_diag.weight.data = 2 * torch.rand((self.sizes[1], self.rank), dtype=self.data_type) - 1.0
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
        # rel2 = expmap0(rel2, c1)
        head1 = project(mobius_add(head1, rel1, c1), c1)
        res1 = givens_rotations(self.rel_diag1(r), head1)
        
        c2 = F.softplus(self.c2[r])
        head2 = expmap0(u, c2)
        rel2= self.rel(r)
        rel11 = expmap0(rel2, c2)
        # rel21= expmap0(rel2, c2)
        head2 = project(mobius_add(head2, rel11, c2), c2)
        res11 = givens_rotations(self.rel_diag2(r), head2)
        
        res1=logmap0(res1,c1)
        res11=logmap0(res11,c2)
        
        c = F.softplus(self.c[r])
        head = u
        rot_mat = self.rel_diag(r)
        rot_q = givens_rotations(rot_mat, head).view((-1, 1, self.rank))
        
        cands = torch.cat([res1.view(-1, 1, self.rank),res11.view(-1, 1, self.rank),rot_q], dim=1)
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
        # rel2 = expmap0(rel2, c1)
        head1 = project(mobius_add(head1, rel1, c1), c1)
        res1 = givens_rotations(self.rel_diag1(r), head1)
        
        c2 = F.softplus(self.c2[r])
        head2 = expmap0(u, c2)
        rel2= self.rel(r)
        rel11 = expmap0(rel2, c2)
        # rel21= expmap0(rel2, c2)
        head2 = project(mobius_add(head2, rel11, c2), c2)
        res11 = givens_rotations(self.rel_diag2(r), head2)
        
        res1=logmap0(res1,c1)
        res11=logmap0(res11,c2)
        
        c = F.softplus(self.c[r])
        head = u
        rot_mat = self.rel_diag(r)
        rot_q = givens_rotations(rot_mat, head).view((-1, 1, self.rank))
        
        cands = torch.cat([res1.view(-1, 1, self.rank),res11.view(-1, 1, self.rank),rot_q], dim=1)
        context_vec = self.context_vec(r).view((-1, 1, self.rank))
        att_weights = torch.sum(context_vec * cands * self.scale, dim=-1, keepdim=True)
        att_weights = self.act(att_weights)
        att_q = torch.sum(att_weights * cands, dim=1)
        
        lhs = expmap0(att_q, c)
        rel = self.rel(r)
        rel_c = expmap0(rel, c)
        res = project(mobius_add(lhs, rel_c, c), c)
        
        return (res, c), self.bh(queries[:, 0])
    
    # def get_rhs(self, queries, eval_mode):
    #     if eval_mode:
    #         return self.entity.weight, self.bt.weight
    #     else:
    #         c = F.softplus(self.c[queries[:, 1]])
    #         return expmap0(self.entity(queries[:, 2]), c), self.bt(queries[:, 2])

class GIE_ref(BaseH):

    def __init__(self, args):
        super(GIE_ref, self).__init__(args)
        # self.rel_diag = nn.Embedding(self.sizes[1], self.rank)
        # self.rel_diag1 = nn.Embedding(self.sizes[1], self.rank)
        # self.rel_diag2 = nn.Embedding(self.sizes[1], self.rank)
        # self.rel_diag.weight.data = 2 * torch.rand((self.sizes[1], self.rank), dtype=self.data_type) - 1.0
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
        # rel2 = expmap0(rel2, c1)
        lhs = project(mobius_add(head1, rel1, c1), c1)
        res1 = givens_reflection(self.rel_diag1(r), lhs)
        
        c2 = F.softplus(self.c2[r])
        head2 = expmap0(u, c2)
        rel2= self.rel(r)
        rel11 = expmap0(rel2, c2)
        # rel21= expmap0(rel2, c2)
        lhss = project(mobius_add(head2, rel11, c2), c2)
        res11 = givens_reflection(self.rel_diag2(r), lhss)
        
        res1=logmap0(res1,c1)
        res11=logmap0(res11,c2)
        
        c = F.softplus(self.c[r])
        head = u
        rot_mat = self.rel_diag(r)
        rot_q = givens_reflection(rot_mat, head).view((-1, 1, self.rank))
        
        cands = torch.cat([res1.view(-1, 1, self.rank),res11.view(-1, 1, self.rank),rot_q], dim=1)
        context_vec = self.context_vec(r).view((-1, 1, self.rank))
        att_weights = torch.sum(context_vec * cands * self.scale, dim=-1, keepdim=True)
        att_weights = self.act(att_weights)
        att_q = torch.sum(att_weights * cands, dim=1)
        
        lhs = expmap0(att_q, c)
        rel = self.rel(r)
        rel_c = expmap0(rel, c)
        res = project(mobius_add(lhs, rel_c, c), c)
        # res = project(lhs, c)
        
        # return ((res, c), self.bh(queries[:, 0])), att_weights
        return (res, c), self.bh(queries[:, 0])
    
    def get_queries_lp(self, queries, r):
        
        u = self.entity(queries[:, 0])

        c1 = F.softplus(self.c1[r])
        head1 = expmap0(u, c1)
        rel1 = self.rel(r)
        rel1 = expmap0(rel1, c1)
        # rel2 = expmap0(rel2, c1)
        lhs = project(mobius_add(head1, rel1, c1), c1)
        res1 = givens_reflection(self.rel_diag1(r), lhs)
        
        c2 = F.softplus(self.c2[r])
        head2 = expmap0(u, c2)
        rel2= self.rel(r)
        rel11 = expmap0(rel2, c2)
        # rel21= expmap0(rel2, c2)
        lhss = project(mobius_add(head2, rel11, c2), c2)
        res11 = givens_reflection(self.rel_diag2(r), lhss)
        
        res1=logmap0(res1,c1)
        res11=logmap0(res11,c2)
        
        c = F.softplus(self.c[r])
        head = u
        rot_mat = self.rel_diag(r)
        rot_q = givens_reflection(rot_mat, head).view((-1, 1, self.rank))
        
        cands = torch.cat([res1.view(-1, 1, self.rank),res11.view(-1, 1, self.rank),rot_q], dim=1)
        context_vec = self.context_vec(r).view((-1, 1, self.rank))
        att_weights = torch.sum(context_vec * cands * self.scale, dim=-1, keepdim=True)
        att_weights = self.act(att_weights)
        att_q = torch.sum(att_weights * cands, dim=1)
        
        lhs = expmap0(att_q, c)
        rel = self.rel(r)
        rel_c = expmap0(rel, c)
        res = project(mobius_add(lhs, rel_c, c), c)
        # res = project(lhs, c)
        
        # return ((res, c), self.bh(queries[:, 0])), att_weights
        return (res, c), self.bh(queries[:, 0])

class GIE_trans(BaseH):

    def __init__(self, args):
        super(GIE_trans, self).__init__(args)
        # self.rel_diag = nn.Embedding(self.sizes[1], self.rank)
        # self.rel_diag1 = nn.Embedding(self.sizes[1], self.rank)
        # self.rel_diag2 = nn.Embedding(self.sizes[1], self.rank)
        # self.rel_diag.weight.data = 2 * torch.rand((self.sizes[1], 2 * self.rank), dtype=self.data_type) - 1.0
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
        # rel2 = expmap0(rel2, c1)
        lhs = project(mobius_add(head1, rel1, c1), c1)
        # res1 = givens_reflection(self.rel_diag1(r), lhs)
        
        c2 = F.softplus(self.c2[r])
        head2 = expmap0(u, c2)
        rel2 = self.rel(r)
        rel11 = expmap0(rel2, c2)
        # rel21= expmap0(rel2, c2)
        lhss = project(mobius_add(head2, rel11, c2), c2)
        # res11 = givens_reflection(self.rel_diag2(r), lhss)
        
        res1=logmap0(lhs,c1)  
        res11=logmap0(lhss,c2)
        
        c = F.softplus(self.c[r])
        head = u
        # rot_mat, _ = torch.chunk(self.rel_diag(r), 2, dim=1)
        # rot_q = givens_reflection(rot_mat, head).view((-1, 1, self.rank))
        # rel1, rel2 = torch.chunk(self.rel(r), 2, dim=1)
        # rot_q = (head+rel1).view((-1, 1, self.rank))
        rot_q = (head).view((-1, 1, self.rank))
        
        cands = torch.cat([res1.view(-1, 1, self.rank),res11.view(-1, 1, self.rank),rot_q], dim=1)
        context_vec = self.context_vec(r).view((-1, 1, self.rank))
        att_weights = torch.sum(context_vec * cands * self.scale, dim=-1, keepdim=True)
        att_weights = self.act(att_weights)
        att_q = torch.sum(att_weights * cands, dim=1)
        
        lhs = expmap0(att_q, c)
        rel = self.rel(r)
        rel = expmap0(rel, c)
        res = project(mobius_add(lhs, rel, c), c)
        # res = mobius_add(lhs, rel, c)
        
        # return ((res, c), self.bh(queries[:, 0])), att_weights
        return (res, c), self.bh(queries[:, 0])
    
    def get_queries_lp(self, queries, r):
        
        u = self.entity(queries[:, 0])

        c1 = F.softplus(self.c1[r])
        head1 = expmap0(u, c1)
        rel1 = self.rel(r)
        rel1 = expmap0(rel1, c1)
        # rel2 = expmap0(rel2, c1)
        lhs = project(mobius_add(head1, rel1, c1), c1)
        # res1 = givens_reflection(self.rel_diag1(r), lhs)
        
        c2 = F.softplus(self.c2[r])
        head2 = expmap0(u, c2)
        rel2 = self.rel(r)
        rel11 = expmap0(rel2, c2)
        # rel21= expmap0(rel2, c2)
        lhss = project(mobius_add(head2, rel11, c2), c2)
        # res11 = givens_reflection(self.rel_diag2(r), lhss)
        
        res1=logmap0(lhs,c1)  
        res11=logmap0(lhss,c2)
        
        c = F.softplus(self.c[r])
        head = u
        # rot_mat, _ = torch.chunk(self.rel_diag(r), 2, dim=1)
        # rot_q = givens_reflection(rot_mat, head).view((-1, 1, self.rank))
        # rel1, rel2 = torch.chunk(self.rel(r), 2, dim=1)
        # rot_q = (head+rel1).view((-1, 1, self.rank))
        rot_q = (head).view((-1, 1, self.rank))
        
        cands = torch.cat([res1.view(-1, 1, self.rank),res11.view(-1, 1, self.rank),rot_q], dim=1)
        context_vec = self.context_vec(r).view((-1, 1, self.rank))
        att_weights = torch.sum(context_vec * cands * self.scale, dim=-1, keepdim=True)
        att_weights = self.act(att_weights)
        att_q = torch.sum(att_weights * cands, dim=1)
        
        lhs = expmap0(att_q, c)
        rel = self.rel(r)
        rel = expmap0(rel, c)
        res = project(mobius_add(lhs, rel, c), c)
        # res = mobius_add(lhs, rel, c)
        return (res, c), self.bh(queries[:, 0])
    
class TransH(BaseH):

    def __init__(self, args):
        super(TransH, self).__init__(args)

    def get_queries(self, queries):

        u = self.entity(queries[:, 0])
        r = queries[:, 1]
        
        rvh = self.rel(r)

        c = F.softplus(self.c[r])  
        
        u_e = logmap0(u, c)
        u_e = u_e + rvh
        u_h = expmap0(u_e, c)
        # res = project(u_h, c)
        res = u_h
        
        return (res, c), self.bh(queries[:, 0]) # lhs biases

    def get_queries_lp(self, queries, r):
        
        u = self.entity(queries[:, 0])
        
        rvh = self.rel(r)

        c = F.softplus(self.c[r])  
        
        u_e = logmap0(u, c)
        u_e = u_e + rvh
        u_h = expmap0(u_e, c)
        # res = project(u_h, c)
        res = u_h
        
        return (res, c), self.bh(queries[:, 0]) # lhs biases

class MurP(BaseH):

    def __init__(self, args):
        super(MurP, self).__init__(args)
        c_init = torch.ones((self.sizes[1], 1), dtype=self.data_type)
        self.c = nn.Parameter(c_init, requires_grad=False)        
        
    def get_queries(self, queries):

        u = self.entity(queries[:, 0])
        r = queries[:, 1]
        Ru = self.rel_diag(r)
        rvh = self.rel(r)

        c = F.softplus(self.c[r])  
        
        u_e = logmap0(u, c)
        u_W = u_e * Ru
        u_m = expmap0(u_W, c)
        rel1 = expmap0(rvh, c)
        res = project(mobius_add(u_m, rel1, c), c)
        
        return (res, c), self.bh(queries[:, 0]) # lhs biases

    def get_queries_lp(self, queries, r):
        
        u = self.entity(queries[:, 0])
        Ru = self.rel_diag(r)
        rvh = self.rel(r)

        c = F.softplus(self.c[r])  
        
        u_e = logmap0(u, c)
        u_W = u_e * Ru
        u_m = expmap0(u_W, c)
        rel1 = expmap0(rvh, c)
        res = project(mobius_add(u_m, rel1, c), c)
        
        return (res, c), self.bh(queries[:, 0]) # lhs biases
    
    # def get_rhs(self, queries, eval_mode):
    #     if eval_mode:
    #         return self.entity.weight, self.bt.weight
    #     else:
    #         c = F.softplus(self.c[queries[:, 1]])
    #         return expmap0(self.entity(queries[:, 2]), c), self.bt(queries[:, 2])
    
class ScaleH(BaseH):

    def __init__(self, args):
        super(ScaleH, self).__init__(args)
        
    def get_queries(self, queries):

        u = self.entity(queries[:, 0])
        r = queries[:, 1]
        Ru = self.rel_diag(r)
        rvh = self.rel(r)

        c = F.softplus(self.c[r])  
        
        u_W = u * Ru
        u_m = expmap0(u_W, c)
        rel1 = expmap0(rvh, c)
        res = project(mobius_add(u_m, rel1, c), c)
        
        return (res, c), self.bh(queries[:, 0]) # lhs biases

    def get_queries_lp(self, queries, r):
        
        u = self.entity(queries[:, 0])
        Ru = self.rel_diag(r)
        rvh = self.rel(r)

        c = F.softplus(self.c[r])  
        
        u_W = u * Ru
        u_m = expmap0(u_W, c)
        rel1 = expmap0(rvh, c)
        res = project(mobius_add(u_m, rel1, c), c)
        
        return (res, c), self.bh(queries[:, 0]) # lhs biases
    
class ATT2_rot(BaseH):

    def __init__(self, args):
        super(ATT2_rot, self).__init__(args)
        # self.rel_diag = nn.Embedding(self.sizes[1], self.rank)
        # self.rel_diag1 = nn.Embedding(self.sizes[1], self.rank)
        # self.rel_diag.weight.data = 2 * torch.rand((self.sizes[1], self.rank), dtype=self.data_type) - 1.0
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
        # rel2 = expmap0(rel2, c1)
        lhs = project(mobius_add(head1, rel1, c1), c1)
        res1 = givens_rotations(self.rel_diag1(r), lhs)
        
        res1=logmap0(res1,c1)
        
        c = F.softplus(self.c[r])
        head = u
        rot_mat = self.rel_diag(r)
        rot_q = givens_rotations(rot_mat, head).view((-1, 1, self.rank))
        
        cands = torch.cat([res1.view(-1, 1, self.rank),rot_q], dim=1)
        context_vec = self.context_vec(r).view((-1, 1, self.rank))
        att_weights = torch.sum(context_vec * cands * self.scale, dim=-1, keepdim=True)
        att_weights = self.act(att_weights)
        att_q = torch.sum(att_weights * cands, dim=1)
        
        lhs = expmap0(att_q, c)
        rel = self.rel(r)
        rel_c = expmap0(rel, c)
        res = project(mobius_add(lhs, rel_c, c), c)
        # res = project(lhs, c)
        
        return (res, c), self.bh(queries[:, 0])
    
    def get_queries_lp(self, queries, r):
        
        u = self.entity(queries[:, 0])

        c1 = F.softplus(self.c1[r])
        head1 = expmap0(u, c1)
        rel1 = self.rel(r)
        rel1 = expmap0(rel1, c1)
        # rel2 = expmap0(rel2, c1)
        lhs = project(mobius_add(head1, rel1, c1), c1)
        res1 = givens_rotations(self.rel_diag1(r), lhs)
        
        res1=logmap0(res1,c1)
        
        c = F.softplus(self.c[r])
        head = u
        rot_mat = self.rel_diag(r)
        rot_q = givens_rotations(rot_mat, head).view((-1, 1, self.rank))
        
        cands = torch.cat([res1.view(-1, 1, self.rank),rot_q], dim=1)
        context_vec = self.context_vec(r).view((-1, 1, self.rank))
        att_weights = torch.sum(context_vec * cands * self.scale, dim=-1, keepdim=True)
        att_weights = self.act(att_weights)
        att_q = torch.sum(att_weights * cands, dim=1)
        
        lhs = expmap0(att_q, c)
        rel = self.rel(r)
        rel_c = expmap0(rel, c)
        res = project(mobius_add(lhs, rel_c, c), c)
        # res = project(lhs, c)
        
        return (res, c), self.bh(queries[:, 0])
    
class ATT2_ref(BaseH):

    def __init__(self, args):
        super(ATT2_ref, self).__init__(args)
        # self.rel_diag = nn.Embedding(self.sizes[1], 2 * self.rank)
        # self.rel_diag1 = nn.Embedding(self.sizes[1], self.rank)
        # self.rel_diag.weight.data = 2 * torch.rand((self.sizes[1], 2 * self.rank), dtype=self.data_type) - 1.0
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
        # rel2 = expmap0(rel2, c1)
        lhs = project(mobius_add(head1, rel1, c1), c1)
        res1 = givens_reflection(self.rel_diag1(r), lhs)
        
        res1=logmap0(res1,c1)
        
        c = F.softplus(self.c[r])
        head = u
        rot_mat = self.rel_diag(r)
        rot_q = givens_reflection(rot_mat, head).view((-1, 1, self.rank))
        
        cands = torch.cat([res1.view(-1, 1, self.rank),rot_q], dim=1)
        context_vec = self.context_vec(r).view((-1, 1, self.rank))
        att_weights = torch.sum(context_vec * cands * self.scale, dim=-1, keepdim=True)
        att_weights = self.act(att_weights)
        att_q = torch.sum(att_weights * cands, dim=1)
        
        lhs = expmap0(att_q, c)
        rel = self.rel(r)
        rel_c = expmap0(rel, c)
        res = project(mobius_add(lhs, rel_c, c), c)
        # res = mobius_add(lhs, rel, c)
        return (res, c), self.bh(queries[:, 0])

    def get_queries_lp(self, queries, r):
        
        u = self.entity(queries[:, 0])
        
        c1 = F.softplus(self.c1[r])
        head1 = expmap0(u, c1)
        rel1 = self.rel(r)
        rel1 = expmap0(rel1, c1)
        # rel2 = expmap0(rel2, c1)
        lhs = project(mobius_add(head1, rel1, c1), c1)
        res1 = givens_reflection(self.rel_diag1(r), lhs)
        
        res1=logmap0(res1,c1)
        
        c = F.softplus(self.c[r])
        head = u
        rot_mat = self.rel_diag(r)
        rot_q = givens_reflection(rot_mat, head).view((-1, 1, self.rank))
        
        cands = torch.cat([res1.view(-1, 1, self.rank),rot_q], dim=1)
        context_vec = self.context_vec(r).view((-1, 1, self.rank))
        att_weights = torch.sum(context_vec * cands * self.scale, dim=-1, keepdim=True)
        att_weights = self.act(att_weights)
        att_q = torch.sum(att_weights * cands, dim=1)
        
        lhs = expmap0(att_q, c)
        rel = self.rel(r)
        rel_c = expmap0(rel, c)
        res = project(mobius_add(lhs, rel_c, c), c)
        # res = mobius_add(lhs, rel, c)
        return (res, c), self.bh(queries[:, 0])
    
class ATT2_trans(BaseH):

    def __init__(self, args):
        super(ATT2_trans, self).__init__(args)
        # self.rel_diag = nn.Embedding(self.sizes[1], 2 * self.rank)
        # self.rel_diag1 = nn.Embedding(self.sizes[1], self.rank)
        # self.rel_diag.weight.data = 2 * torch.rand((self.sizes[1], 2 * self.rank), dtype=self.data_type) - 1.0
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
        # rel2 = expmap0(rel2, c1)
        lhs = project(mobius_add(head1, rel1, c1), c1)
        # res1 = givens_reflection(self.rel_diag1(r), lhs)
        
        res1=logmap0(lhs,c1)
        
        c = F.softplus(self.c[r])
        head = u
        # rot_mat, _ = torch.chunk(self.rel_diag(r), 2, dim=1)
        # rot_q = givens_reflection(rot_mat, head).view((-1, 1, self.rank))
        # rot_q = (head+rel1).view((-1, 1, self.rank))
        rot_q = (head).view((-1, 1, self.rank))
        
        cands = torch.cat([res1.view(-1, 1, self.rank),rot_q], dim=1)
        context_vec = self.context_vec(r).view((-1, 1, self.rank))
        att_weights = torch.sum(context_vec * cands * self.scale, dim=-1, keepdim=True)
        att_weights = self.act(att_weights)
        att_q = torch.sum(att_weights * cands, dim=1)
        
        lhs = expmap0(att_q, c)
        rel = self.rel(r)
        rel = expmap0(rel, c)
        res = project(mobius_add(lhs, rel, c), c)
        # res = mobius_add(lhs, rel, c)
        return (res, c), self.bh(queries[:, 0])
    
    def get_queries_lp(self, queries, r):

        u = self.entity(queries[:, 0])

        c1 = F.softplus(self.c1[r])
        head1 = expmap0(u, c1)
        rel1 = self.rel(r)
        rel1 = expmap0(rel1, c1)
        # rel2 = expmap0(rel2, c1)
        lhs = project(mobius_add(head1, rel1, c1), c1)
        # res1 = givens_reflection(self.rel_diag1(r), lhs)
        
        res1=logmap0(lhs,c1)
        
        c = F.softplus(self.c[r])
        head = u
        # rot_mat, _ = torch.chunk(self.rel_diag(r), 2, dim=1)
        # rot_q = givens_reflection(rot_mat, head).view((-1, 1, self.rank))
        # rot_q = (head+rel1).view((-1, 1, self.rank))
        rot_q = (head).view((-1, 1, self.rank))
        
        cands = torch.cat([res1.view(-1, 1, self.rank),rot_q], dim=1)
        context_vec = self.context_vec(r).view((-1, 1, self.rank))
        att_weights = torch.sum(context_vec * cands * self.scale, dim=-1, keepdim=True)
        att_weights = self.act(att_weights)
        att_q = torch.sum(att_weights * cands, dim=1)
        
        lhs = expmap0(att_q, c)
        rel = self.rel(r)
        rel = expmap0(rel, c)
        res = project(mobius_add(lhs, rel, c), c)
        # res = mobius_add(lhs, rel, c)
        return (res, c), self.bh(queries[:, 0])
    
class HEM(BaseH):
    
    def __init__(self, args):
        super(HEM, self).__init__(args)
        self.entity.weight.data = self.init_size * torch.randn((self.sizes[0], self.rank), dtype=self.data_type)
        self.rel.weight.data = self.init_size * torch.randn((self.sizes[1], 2 * self.rank), dtype=self.data_type)
        self.rel_diag = nn.Embedding(self.sizes[1],
                                     self.rank)  # size:(entnum,relnum,entnum), rank:dim, rel_diag:(relnum,dim)
        self.rel_diag.weight.data = 2 * torch.rand((self.sizes[1], self.rank), dtype=self.data_type) - 1.0
        self.rel_diag_t = nn.Embedding(self.sizes[1],
                                       self.rank)  # size:(entnum,relnum,entnum), rank:dim, rel_diag:(relnum,dim)
        self.rel_diag_t.weight.data = 2 * torch.rand((self.sizes[1], self.rank), dtype=self.data_type) - 1.0
        self.multi_c = True  # args.multi_c
        t_init = torch.ones((self.sizes[2], 1), dtype=self.data_type)
        self.t = nn.Parameter(t_init, requires_grad=True)
        if self.multi_c:
            c_init = torch.ones((self.sizes[1], 1), dtype=self.data_type)
        else:
            c_init = torch.ones((1, 1), dtype=self.data_type)
        self.c = nn.Parameter(c_init, requires_grad=True)

    def get_rhs(self, queries, eval_mode):
        if eval_mode:
            return self.entity.weight, self.bt.weight
        else:
            c = F.softplus(self.c[queries[:, 1]] * self.t[queries[:, 2]])  # 可训曲率，每个关系都存在一个特定的曲率
            tail = self.entity(queries[:, 2])
            rel1 = self.rel_diag_t(queries[:, 1])
            rel1 = rel1 * tail
            rel1 = expmap0(rel1, c)
            return rel1, self.bt(queries[:, 2])

    def get_queries(self,queries):
        c = F.softplus(self.c[queries[:, 1]]*self.t[queries[:, 2]])
        head = self.entity(queries[:, 0])
        rel1 = self.rel_diag(queries[:, 1])
        rel, _ = torch.chunk(self.rel(queries[:, 1]), 2, dim=1)
        head1 = rel1*head
        head1 = expmap0(head1, c)
        rel2 = expmap0(rel, c)
        res2 = mobius_add(head1, rel2, c)
        return (res2, c), self.bh(queries[:, 0])
    
    def get_queries_lp(self,queries,r):
        c = F.softplus(self.c[r]*self.t[queries[:, 2]])
        head = self.entity(queries[:, 0])
        rel1 = self.rel_diag(r)
        rel, _ = torch.chunk(self.rel(r), 2, dim=1)
        head1 = rel1*head
        head1 = expmap0(head1, c)
        rel2 = expmap0(rel, c)
        res2 = mobius_add(head1, rel2, c)
        return (res2, c), self.bh(queries[:, 0])