
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from models.base import KGModel
from utils.euclidean import givens_rotations, givens_reflection, euc_sqdistance
from utils.hyperbolic import mobius_add, expmap0,logmap0, project, hyp_distance_multi_c

CUSTOM_MODELS = ["customGIEAtt1", "customGIEAtt", "customAttH", "customAttH2", "customGIE","customATT2","customGIE_rot","customGIE_ref","customGIE_trans","customATT2_rot","customATT2_ref","customATT2_trans", "customRotH"]


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
       
class customGIEAtt(BaseH):

    def __init__(self, args):
        super(customGIEAtt, self).__init__(args)
        self.att1 = nn.Embedding(self.sizes[1], self.rank)
        self.att1.weight.data = self.init_size * torch.randn((self.sizes[1], self.rank), dtype=self.data_type)
        self.att2 = nn.Embedding(self.sizes[1], self.rank)
        self.att2.weight.data = self.init_size * torch.randn((self.sizes[1], self.rank), dtype=self.data_type)
        self.att3 = nn.Embedding(self.sizes[1], self.rank)
        self.att3.weight.data = self.init_size * torch.randn((self.sizes[1], self.rank), dtype=self.data_type)
        self.context_vec = nn.Embedding(self.sizes[1], self.rank)
        self.rel_diag3 = nn.Embedding(self.sizes[1], self.rank)
        self.rel_diag3.weight.data = 2 * torch.rand((self.sizes[1], self.rank), dtype=self.data_type) - 1.0
        self.rel_diag4 = nn.Embedding(self.sizes[1], self.rank)
        self.rel_diag4.weight.data = 2 * torch.rand((self.sizes[1], self.rank), dtype=self.data_type) - 1.0
        self.rel_diag5 = nn.Embedding(self.sizes[1], self.rank)
        self.rel_diag5.weight.data = 2 * torch.rand((self.sizes[1], self.rank), dtype=self.data_type) - 1.0
        self.context_vec.weight.data = self.init_size * torch.randn((self.sizes[1], self.rank), dtype=self.data_type)
        self.act = nn.Softmax(dim=1)
        if args.dtype == "double":
            self.scale = torch.Tensor([1. / np.sqrt(self.rank)]).double().cuda()
        else:
            self.scale = torch.Tensor([1. / np.sqrt(self.rank)]).cuda()

    def get_queries(self, queries):
        
        u = self.entity(queries[:, 0])
        r = queries[:, 1]
        rvh = self.rel(r)
        
        rot_mat = self.rel_diag(r)
        ref_mat = self.rel_diag1(r)
        rot_mat1 = self.rel_diag2(r)
        ref_mat1 = self.rel_diag3(r)
        rot_mat2 = self.rel_diag4(r)
        ref_mat2 = self.rel_diag5(r)

        c1 = F.softplus(self.c1[r])
        head1 = expmap0(u, c1)
        head1 = logmap0(head1, c1)
        rot_1 = givens_rotations(rot_mat1, head1).view((-1, 1, self.rank))
        ref_1 = givens_reflection(ref_mat1, head1).view((-1, 1, self.rank))
        
        cands = torch.cat([ref_1, rot_1], dim=1)
        att_vec1 = self.att1(r).view((-1, 1, self.rank))
        att_weights = torch.sum(att_vec1 * cands * self.scale.to(cands.device), dim=-1, keepdim=True)
        att_weights = self.act(att_weights)
        att_1 = torch.sum(att_weights * cands, dim=1).view((-1, 1, self.rank))
        
        c2 = F.softplus(self.c2[r])
        head2 = expmap0(u, c2)
        head2 = logmap0(head2, c2)
        rot_2 = givens_rotations(rot_mat2, head2).view((-1, 1, self.rank))
        ref_2 = givens_reflection(ref_mat2, head2).view((-1, 1, self.rank))
        
        cands = torch.cat([ref_2, rot_2], dim=1)
        att_vec2 = self.att2(r).view((-1, 1, self.rank))
        att_weights = torch.sum(att_vec2 * cands * self.scale.to(cands.device), dim=-1, keepdim=True)
        att_weights = self.act(att_weights)
        att_2 = torch.sum(att_weights * cands, dim=1).view((-1, 1, self.rank))
        
        head3 = u
        rot_q = givens_rotations(rot_mat, head3).view((-1, 1, self.rank))
        ref_q = givens_reflection(ref_mat, head3).view((-1, 1, self.rank))
        
        cands = torch.cat([ref_q, rot_q], dim=1)
        att_vec = self.att3(r).view((-1, 1, self.rank))
        att_weights = torch.sum(att_vec * cands * self.scale.to(cands.device), dim=-1, keepdim=True)
        att_weights = self.act(att_weights)
        res3 = torch.sum(att_weights * cands, dim=1).view((-1, 1, self.rank))
        
        cands = torch.cat([att_1,att_2,res3], dim=1)
        context_vec = self.context_vec(r).view((-1, 1, self.rank))
        att_weights = torch.sum(context_vec * cands * self.scale.to(cands.device), dim=-1, keepdim=True)
        att_weights = self.act(att_weights)
        att_q = torch.sum(att_weights * cands, dim=1)
        
        c = F.softplus(self.c[r])
        lhs = expmap0(att_q, c)
        rel = self.rel(r)
        rel_c = expmap0(rel, c)
        res = project(mobius_add(lhs, rel_c, c), c)
        
        return (res, c), self.bh(queries[:, 0])

    def get_queries_lp(self, queries, r):
        
        u = self.entity(queries[:, 0])
        rvh = self.rel(r)
        
        rot_mat = self.rel_diag(r)
        ref_mat = self.rel_diag1(r)
        rot_mat1 = self.rel_diag2(r)
        ref_mat1 = self.rel_diag3(r)
        rot_mat2 = self.rel_diag4(r)
        ref_mat2 = self.rel_diag5(r)

        c1 = F.softplus(self.c1[r])
        head1 = expmap0(u, c1)
        head1 = logmap0(head1, c1)
        rot_1 = givens_rotations(rot_mat1, head1).view((-1, 1, self.rank))
        ref_1 = givens_reflection(ref_mat1, head1).view((-1, 1, self.rank))
        
        cands = torch.cat([ref_1, rot_1], dim=1)
        att_vec1 = self.att1(r).view((-1, 1, self.rank))
        att_weights = torch.sum(att_vec1 * cands * self.scale.to(cands.device), dim=-1, keepdim=True)
        att_weights = self.act(att_weights)
        att_1 = torch.sum(att_weights * cands, dim=1).view((-1, 1, self.rank))
        
        c2 = F.softplus(self.c2[r])
        head2 = expmap0(u, c2)
        head2 = logmap0(head2, c2)
        rot_2 = givens_rotations(rot_mat2, head2).view((-1, 1, self.rank))
        ref_2 = givens_reflection(ref_mat2, head2).view((-1, 1, self.rank))
        
        cands = torch.cat([ref_2, rot_2], dim=1)
        att_vec2 = self.att2(r).view((-1, 1, self.rank))
        att_weights = torch.sum(att_vec2 * cands * self.scale.to(cands.device), dim=-1, keepdim=True)
        att_weights = self.act(att_weights)
        att_2 = torch.sum(att_weights * cands, dim=1).view((-1, 1, self.rank))
        
        head3 = u
        rot_q = givens_rotations(rot_mat, head3).view((-1, 1, self.rank))
        ref_q = givens_reflection(ref_mat, head3).view((-1, 1, self.rank))
        
        cands = torch.cat([ref_q, rot_q], dim=1)
        att_vec = self.att3(r).view((-1, 1, self.rank))
        att_weights = torch.sum(att_vec * cands * self.scale.to(cands.device), dim=-1, keepdim=True)
        att_weights = self.act(att_weights)
        res3 = torch.sum(att_weights * cands, dim=1).view((-1, 1, self.rank))
        
        cands = torch.cat([att_1,att_2,res3], dim=1)
        context_vec = self.context_vec(r).view((-1, 1, self.rank))
        att_weights = torch.sum(context_vec * cands * self.scale.to(cands.device), dim=-1, keepdim=True)
        att_weights = self.act(att_weights)
        att_q = torch.sum(att_weights * cands, dim=1)
        
        c = F.softplus(self.c[r])
        lhs = expmap0(att_q, c)
        rel = self.rel(r)
        rel_c = expmap0(rel, c)
        res = project(mobius_add(lhs, rel_c, c), c)
        
        return (res, c), self.bh(queries[:, 0])
    
class customGIEAtt3(BaseH):

    def __init__(self, args):
        super(customGIEAtt3, self).__init__(args)
        self.att1 = nn.Embedding(self.sizes[1], self.rank)
        self.att1.weight.data = self.init_size * torch.randn((self.sizes[1], self.rank), dtype=self.data_type)
        self.att2 = nn.Embedding(self.sizes[1], self.rank)
        self.att2.weight.data = self.init_size * torch.randn((self.sizes[1], self.rank), dtype=self.data_type)
        self.att3 = nn.Embedding(self.sizes[1], self.rank)
        self.att3.weight.data = self.init_size * torch.randn((self.sizes[1], self.rank), dtype=self.data_type)
        
        self.rel_diag3 = nn.Embedding(self.sizes[1], self.rank)
        self.rel_diag3.weight.data = 2 * torch.rand((self.sizes[1], self.rank), dtype=self.data_type) - 1.0
        self.rel_diag4 = nn.Embedding(self.sizes[1], self.rank)
        self.rel_diag4.weight.data = 2 * torch.rand((self.sizes[1], self.rank), dtype=self.data_type) - 1.0
        self.rel_diag5 = nn.Embedding(self.sizes[1], self.rank)
        self.rel_diag5.weight.data = 2 * torch.rand((self.sizes[1], self.rank), dtype=self.data_type) - 1.0
        self.rel_diag6 = nn.Embedding(self.sizes[1], self.rank)
        self.rel_diag6.weight.data = 2 * torch.rand((self.sizes[1], self.rank), dtype=self.data_type) - 1.0
        self.rel_diag7 = nn.Embedding(self.sizes[1], self.rank)
        self.rel_diag7.weight.data = 2 * torch.rand((self.sizes[1], self.rank), dtype=self.data_type) - 1.0
        self.rel_diag8 = nn.Embedding(self.sizes[1], self.rank)
        self.rel_diag8.weight.data = 2 * torch.rand((self.sizes[1], self.rank), dtype=self.data_type) - 1.0
        
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
        rvh = self.rel(r)
        
        rot_mat = self.rel_diag(r)
        ref_mat = self.rel_diag1(r)
        scl_mat = self.rel_diag2(r)
        rot_mat1 = self.rel_diag3(r)
        ref_mat1 = self.rel_diag4(r)
        scl_mat1 = self.rel_diag5(r)
        rot_mat2 = self.rel_diag6(r)
        ref_mat2 = self.rel_diag7(r)
        scl_mat2 = self.rel_diag8(r)

        c1 = F.softplus(self.c1[r])
        head1 = expmap0(u, c1)
        head1 = logmap0(head1, c1)
        rot_1 = givens_rotations(rot_mat1, head1).view((-1, 1, self.rank))
        ref_1 = givens_reflection(ref_mat1, head1).view((-1, 1, self.rank))
        scl_1 = (scl_mat1*head1).view((-1, 1, self.rank))
        
        cands = torch.cat([ref_1, rot_1, scl_1], dim=1)
        att_vec1 = self.att1(r).view((-1, 1, self.rank))
        att_weights = torch.sum(att_vec1 * cands * self.scale.to(cands.device), dim=-1, keepdim=True)
        att_weights = self.act(att_weights)
        att_1 = torch.sum(att_weights * cands, dim=1).view((-1, 1, self.rank))
        
        c2 = F.softplus(self.c2[r])
        head2 = expmap0(u, c2)
        head2 = logmap0(head2, c2)
        rot_2 = givens_rotations(rot_mat2, head2).view((-1, 1, self.rank))
        ref_2 = givens_reflection(ref_mat2, head2).view((-1, 1, self.rank))
        scl_2 = (scl_mat2*head2).view((-1, 1, self.rank))
        
        cands = torch.cat([ref_2, rot_2, scl_2], dim=1)
        att_vec2 = self.att2(r).view((-1, 1, self.rank))
        att_weights = torch.sum(att_vec2 * cands * self.scale.to(cands.device), dim=-1, keepdim=True)
        att_weights = self.act(att_weights)
        att_2 = torch.sum(att_weights * cands, dim=1).view((-1, 1, self.rank))
        
        head3 = u
        rot_q = givens_rotations(rot_mat, head3).view((-1, 1, self.rank))
        ref_q = givens_reflection(ref_mat, head3).view((-1, 1, self.rank))
        scl_q = (scl_mat*head3).view((-1, 1, self.rank))
        
        cands = torch.cat([ref_q, rot_q, scl_q], dim=1)
        att_vec = self.att3(r).view((-1, 1, self.rank))
        att_weights = torch.sum(att_vec * cands * self.scale.to(cands.device), dim=-1, keepdim=True)
        att_weights = self.act(att_weights)
        res3 = torch.sum(att_weights * cands, dim=1).view((-1, 1, self.rank))
        
        cands = torch.cat([att_1,att_2,res3], dim=1)
        context_vec = self.context_vec(r).view((-1, 1, self.rank))
        att_weights = torch.sum(context_vec * cands * self.scale.to(cands.device), dim=-1, keepdim=True)
        att_weights = self.act(att_weights)
        att_q = torch.sum(att_weights * cands, dim=1)
        
        c = F.softplus(self.c[r])
        lhs = expmap0(att_q, c)
        rel = self.rel(r)
        rel_c = expmap0(rel, c)
        res = project(mobius_add(lhs, rel_c, c), c)
        
        return (res, c), self.bh(queries[:, 0])

    def get_queries_lp(self, queries, r):
        
        u = self.entity(queries[:, 0])
        rvh = self.rel(r)
        
        rot_mat = self.rel_diag(r)
        ref_mat = self.rel_diag1(r)
        scl_mat = self.rel_diag2(r)
        rot_mat1 = self.rel_diag3(r)
        ref_mat1 = self.rel_diag4(r)
        scl_mat1 = self.rel_diag5(r)
        rot_mat2 = self.rel_diag6(r)
        ref_mat2 = self.rel_diag7(r)
        scl_mat2 = self.rel_diag8(r)

        c1 = F.softplus(self.c1[r])
        head1 = expmap0(u, c1)
        head1 = logmap0(head1, c1)
        rot_1 = givens_rotations(rot_mat1, head1).view((-1, 1, self.rank))
        ref_1 = givens_reflection(ref_mat1, head1).view((-1, 1, self.rank))
        scl_1 = (scl_mat1*head1).view((-1, 1, self.rank))
        
        cands = torch.cat([ref_1, rot_1, scl_1], dim=1)
        att_vec1 = self.att1(r).view((-1, 1, self.rank))
        att_weights = torch.sum(att_vec1 * cands * self.scale.to(cands.device), dim=-1, keepdim=True)
        att_weights = self.act(att_weights)
        att_1 = torch.sum(att_weights * cands, dim=1).view((-1, 1, self.rank))
        
        c2 = F.softplus(self.c2[r])
        head2 = expmap0(u, c2)
        head2 = logmap0(head2, c2)
        rot_2 = givens_rotations(rot_mat2, head2).view((-1, 1, self.rank))
        ref_2 = givens_reflection(ref_mat2, head2).view((-1, 1, self.rank))
        scl_2 = (scl_mat2*head2).view((-1, 1, self.rank))
        
        cands = torch.cat([ref_2, rot_2, scl_2], dim=1)
        att_vec2 = self.att2(r).view((-1, 1, self.rank))
        att_weights = torch.sum(att_vec2 * cands * self.scale.to(cands.device), dim=-1, keepdim=True)
        att_weights = self.act(att_weights)
        att_2 = torch.sum(att_weights * cands, dim=1).view((-1, 1, self.rank))
        
        head3 = u
        rot_q = givens_rotations(rot_mat, head3).view((-1, 1, self.rank))
        ref_q = givens_reflection(ref_mat, head3).view((-1, 1, self.rank))
        scl_q = (scl_mat*head3).view((-1, 1, self.rank))
        
        cands = torch.cat([ref_q, rot_q, scl_q], dim=1)
        att_vec = self.att3(r).view((-1, 1, self.rank))
        att_weights = torch.sum(att_vec * cands * self.scale.to(cands.device), dim=-1, keepdim=True)
        att_weights = self.act(att_weights)
        res3 = torch.sum(att_weights * cands, dim=1).view((-1, 1, self.rank))
        
        cands = torch.cat([att_1,att_2,res3], dim=1)
        context_vec = self.context_vec(r).view((-1, 1, self.rank))
        att_weights = torch.sum(context_vec * cands * self.scale.to(cands.device), dim=-1, keepdim=True)
        att_weights = self.act(att_weights)
        att_q = torch.sum(att_weights * cands, dim=1)
        
        c = F.softplus(self.c[r])
        lhs = expmap0(att_q, c)
        rel = self.rel(r)
        rel_c = expmap0(rel, c)
        res = project(mobius_add(lhs, rel_c, c), c)
        
        return (res, c), self.bh(queries[:, 0])
    
class customATT2Att3(BaseH):

    def __init__(self, args):
        super(customATT2Att3, self).__init__(args)
        self.att1 = nn.Embedding(self.sizes[1], self.rank)
        self.att1.weight.data = self.init_size * torch.randn((self.sizes[1], self.rank), dtype=self.data_type)
        self.att3 = nn.Embedding(self.sizes[1], self.rank)
        self.att3.weight.data = self.init_size * torch.randn((self.sizes[1], self.rank), dtype=self.data_type)
        
        self.rel_diag3 = nn.Embedding(self.sizes[1], self.rank)
        self.rel_diag3.weight.data = 2 * torch.rand((self.sizes[1], self.rank), dtype=self.data_type) - 1.0
        self.rel_diag4 = nn.Embedding(self.sizes[1], self.rank)
        self.rel_diag4.weight.data = 2 * torch.rand((self.sizes[1], self.rank), dtype=self.data_type) - 1.0
        self.rel_diag5 = nn.Embedding(self.sizes[1], self.rank)
        self.rel_diag5.weight.data = 2 * torch.rand((self.sizes[1], self.rank), dtype=self.data_type) - 1.0
        
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
        rvh = self.rel(r)
        
        rot_mat = self.rel_diag(r)
        ref_mat = self.rel_diag1(r)
        scl_mat = self.rel_diag2(r)
        rot_mat1 = self.rel_diag3(r)
        ref_mat1 = self.rel_diag4(r)
        scl_mat1 = self.rel_diag5(r)

        c1 = F.softplus(self.c1[r])
        head1 = expmap0(u, c1)
        head1 = logmap0(head1, c1)
        rot_1 = givens_rotations(rot_mat1, head1).view((-1, 1, self.rank))
        ref_1 = givens_reflection(ref_mat1, head1).view((-1, 1, self.rank))
        scl_1 = (scl_mat1 * head1).view((-1, 1, self.rank))
        
        cands = torch.cat([ref_1, rot_1, scl_1], dim=1)
        att_vec1 = self.att1(r).view((-1, 1, self.rank))
        att_weights = torch.sum(att_vec1 * cands * self.scale.to(cands.device), dim=-1, keepdim=True)
        att_weights = self.act(att_weights)
        att_1 = torch.sum(att_weights * cands, dim=1).view((-1, 1, self.rank))
        
        head3 = u
        rot_q = givens_rotations(rot_mat, head3).view((-1, 1, self.rank))
        ref_q = givens_reflection(ref_mat, head3).view((-1, 1, self.rank))
        scl_q = (scl_mat * head3).view((-1, 1, self.rank))
        
        cands = torch.cat([ref_q, rot_q, scl_q], dim=1)
        att_vec = self.att3(r).view((-1, 1, self.rank))
        att_weights = torch.sum(att_vec * cands * self.scale.to(cands.device), dim=-1, keepdim=True)
        att_weights = self.act(att_weights)
        res3 = torch.sum(att_weights * cands, dim=1).view((-1, 1, self.rank))
        
        cands = torch.cat([att_1,res3], dim=1)
        context_vec = self.context_vec(r).view((-1, 1, self.rank))
        att_weights = torch.sum(context_vec * cands * self.scale.to(cands.device), dim=-1, keepdim=True)
        att_weights = self.act(att_weights)
        att_q = torch.sum(att_weights * cands, dim=1)
        
        c = F.softplus(self.c[r])
        lhs = expmap0(att_q, c)
        rel = self.rel(r)
        rel_c = expmap0(rel, c)
        res = project(mobius_add(lhs, rel_c, c), c)
        
        return (res, c), self.bh(queries[:, 0])

    def get_queries_lp(self, queries, r):
        
        u = self.entity(queries[:, 0])
        rvh = self.rel(r)
        
        rot_mat = self.rel_diag(r)
        ref_mat = self.rel_diag1(r)
        scl_mat = self.rel_diag2(r)
        rot_mat1 = self.rel_diag3(r)
        ref_mat1 = self.rel_diag4(r)
        scl_mat1 = self.rel_diag5(r)

        c1 = F.softplus(self.c1[r])
        head1 = expmap0(u, c1)
        head1 = logmap0(head1, c1)
        rot_1 = givens_rotations(rot_mat1, head1).view((-1, 1, self.rank))
        ref_1 = givens_reflection(ref_mat1, head1).view((-1, 1, self.rank))
        scl_1 = (scl_mat1 * head1).view((-1, 1, self.rank))
        
        cands = torch.cat([ref_1, rot_1, scl_1], dim=1)
        att_vec1 = self.att1(r).view((-1, 1, self.rank))
        att_weights = torch.sum(att_vec1 * cands * self.scale.to(cands.device), dim=-1, keepdim=True)
        att_weights = self.act(att_weights)
        att_1 = torch.sum(att_weights * cands, dim=1).view((-1, 1, self.rank))
        
        head3 = u
        rot_q = givens_rotations(rot_mat, head3).view((-1, 1, self.rank))
        ref_q = givens_reflection(ref_mat, head3).view((-1, 1, self.rank))
        scl_q = (scl_mat * head3).view((-1, 1, self.rank))
        
        cands = torch.cat([ref_q, rot_q, scl_q], dim=1)
        att_vec = self.att3(r).view((-1, 1, self.rank))
        att_weights = torch.sum(att_vec * cands * self.scale.to(cands.device), dim=-1, keepdim=True)
        att_weights = self.act(att_weights)
        res3 = torch.sum(att_weights * cands, dim=1).view((-1, 1, self.rank))
        
        cands = torch.cat([att_1,res3], dim=1)
        context_vec = self.context_vec(r).view((-1, 1, self.rank))
        att_weights = torch.sum(context_vec * cands * self.scale.to(cands.device), dim=-1, keepdim=True)
        att_weights = self.act(att_weights)
        att_q = torch.sum(att_weights * cands, dim=1)
        
        c = F.softplus(self.c[r])
        lhs = expmap0(att_q, c)
        rel = self.rel(r)
        rel_c = expmap0(rel, c)
        res = project(mobius_add(lhs, rel_c, c), c)
        
        return (res, c), self.bh(queries[:, 0])
    
class customATT2Att(BaseH):

    def __init__(self, args):
        super(customATT2Att, self).__init__(args)
        self.att1 = nn.Embedding(self.sizes[1], self.rank)
        self.att1.weight.data = self.init_size * torch.randn((self.sizes[1], self.rank), dtype=self.data_type)
        self.att2 = nn.Embedding(self.sizes[1], self.rank)
        self.att2.weight.data = self.init_size * torch.randn((self.sizes[1], self.rank), dtype=self.data_type)

        self.rel_diag3 = nn.Embedding(self.sizes[1], self.rank)
        self.rel_diag3.weight.data = 2 * torch.rand((self.sizes[1], self.rank), dtype=self.data_type) - 1.0
        
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
        
        rot_mat = self.rel_diag(r)
        ref_mat = self.rel_diag1(r)
        rot_mat1 = self.rel_diag2(r)
        ref_mat1 = self.rel_diag3(r)

        c1 = F.softplus(self.c1[r])
        head1 = expmap0(u, c1)
        head1 = logmap0(head1, c1)
        rot_1 = givens_rotations(rot_mat1, head1).view((-1, 1, self.rank))
        ref_1 = givens_reflection(ref_mat1, head1).view((-1, 1, self.rank))
        
        cands = torch.cat([ref_1, rot_1], dim=1)
        att_vec1 = self.att1(r).view((-1, 1, self.rank))
        att_weights = torch.sum(att_vec1 * cands * self.scale.to(cands.device), dim=-1, keepdim=True)
        att_weights = self.act(att_weights)
        att_1 = torch.sum(att_weights * cands, dim=1).view((-1, 1, self.rank))
        
        head2 = u
        rot_q = givens_rotations(rot_mat, head2).view((-1, 1, self.rank))
        ref_q = givens_reflection(ref_mat, head2).view((-1, 1, self.rank))
        
        cands = torch.cat([ref_q, rot_q], dim=1)
        att_vec = self.att2(r).view((-1, 1, self.rank))
        att_weights = torch.sum(att_vec * cands * self.scale.to(cands.device), dim=-1, keepdim=True)
        att_weights = self.act(att_weights)
        res2 = torch.sum(att_weights * cands, dim=1).view((-1, 1, self.rank))
        
        cands = torch.cat([att_1,res2], dim=1)
        context_vec = self.context_vec(r).view((-1, 1, self.rank))
        att_weights = torch.sum(context_vec * cands * self.scale.to(cands.device), dim=-1, keepdim=True)
        att_weights = self.act(att_weights)
        att_q = torch.sum(att_weights * cands, dim=1)
        
        rvh = self.rel(r)
        c = F.softplus(self.c[r])
        lhs = expmap0(att_q, c)
        rel_c = expmap0(rvh, c)
        res = project(mobius_add(lhs, rel_c, c), c)
        
        return (res, c), self.bh(queries[:, 0])

    def get_queries_lp(self, queries, r):
        
        u = self.entity(queries[:, 0])
        
        rot_mat = self.rel_diag(r)
        ref_mat = self.rel_diag1(r)
        rot_mat1 = self.rel_diag2(r)
        ref_mat1 = self.rel_diag3(r)

        c1 = F.softplus(self.c1[r])
        head1 = expmap0(u, c1)
        head1 = logmap0(head1, c1)
        rot_1 = givens_rotations(rot_mat1, head1).view((-1, 1, self.rank))
        ref_1 = givens_reflection(ref_mat1, head1).view((-1, 1, self.rank))
        
        cands = torch.cat([ref_1, rot_1], dim=1)
        att_vec1 = self.att1(r).view((-1, 1, self.rank))
        att_weights = torch.sum(att_vec1 * cands * self.scale.to(cands.device), dim=-1, keepdim=True)
        att_weights = self.act(att_weights)
        att_1 = torch.sum(att_weights * cands, dim=1).view((-1, 1, self.rank))
        
        head2 = u
        rot_q = givens_rotations(rot_mat, head2).view((-1, 1, self.rank))
        ref_q = givens_reflection(ref_mat, head2).view((-1, 1, self.rank))
        
        cands = torch.cat([ref_q, rot_q], dim=1)
        att_vec = self.att2(r).view((-1, 1, self.rank))
        att_weights = torch.sum(att_vec * cands * self.scale.to(cands.device), dim=-1, keepdim=True)
        att_weights = self.act(att_weights)
        res2 = torch.sum(att_weights * cands, dim=1).view((-1, 1, self.rank))
        
        cands = torch.cat([att_1,res2], dim=1)
        context_vec = self.context_vec(r).view((-1, 1, self.rank))
        att_weights = torch.sum(context_vec * cands * self.scale.to(cands.device), dim=-1, keepdim=True)
        att_weights = self.act(att_weights)
        att_q = torch.sum(att_weights * cands, dim=1)
        
        rvh = self.rel(r)
        c = F.softplus(self.c[r])
        lhs = expmap0(att_q, c)
        rel_c = expmap0(rvh, c)
        res = project(mobius_add(lhs, rel_c, c), c)
        
        return (res, c), self.bh(queries[:, 0])
    
class customGIEAtt_1(BaseH):

    def __init__(self, args):
        super(customGIEAtt_1, self).__init__(args)
        self.att1 = nn.Embedding(self.sizes[1], self.rank)
        self.att1.weight.data = self.init_size * torch.randn((self.sizes[1], self.rank), dtype=self.data_type)
        self.att2 = nn.Embedding(self.sizes[1], self.rank)
        self.att2.weight.data = self.init_size * torch.randn((self.sizes[1], self.rank), dtype=self.data_type)
        self.att3 = nn.Embedding(self.sizes[1], self.rank)
        self.att3.weight.data = self.init_size * torch.randn((self.sizes[1], self.rank), dtype=self.data_type)
        self.context_vec = nn.Embedding(self.sizes[1], self.rank)
        self.rel_diag3 = nn.Embedding(self.sizes[1], self.rank)
        self.rel_diag3.weight.data = 2 * torch.rand((self.sizes[1], self.rank), dtype=self.data_type) - 1.0
        self.rel_diag4 = nn.Embedding(self.sizes[1], self.rank)
        self.rel_diag4.weight.data = 2 * torch.rand((self.sizes[1], self.rank), dtype=self.data_type) - 1.0
        self.rel_diag5 = nn.Embedding(self.sizes[1], self.rank)
        self.rel_diag5.weight.data = 2 * torch.rand((self.sizes[1], self.rank), dtype=self.data_type) - 1.0
        self.context_vec.weight.data = self.init_size * torch.randn((self.sizes[1], self.rank), dtype=self.data_type)
        self.act = nn.Softmax(dim=1)
        if args.dtype == "double":
            self.scale = torch.Tensor([1. / np.sqrt(self.rank)]).double().cuda()
        else:
            self.scale = torch.Tensor([1. / np.sqrt(self.rank)]).cuda()

    def get_queries(self, queries):
        
        u = self.entity(queries[:, 0])
        r = queries[:, 1]
        rvh = self.rel(r)
        
        rot_mat = self.rel_diag(r)
        ref_mat = self.rel_diag1(r)
        rot_mat1 = self.rel_diag2(r)
        ref_mat1 = self.rel_diag3(r)
        rot_mat2 = self.rel_diag4(r)
        ref_mat2 = self.rel_diag5(r)

        c1 = F.softplus(self.c1[r])
        head1 = expmap0(u, c1)
        rot_1 = givens_rotations(rot_mat1, head1).view((-1, 1, self.rank))
        ref_1 = givens_reflection(ref_mat1, head1).view((-1, 1, self.rank))
        
        cands = torch.cat([ref_1, rot_1], dim=1)
        att_vec1 = self.att1(r).view((-1, 1, self.rank))
        att_weights = torch.sum(att_vec1 * cands * self.scale.to(cands.device), dim=-1, keepdim=True)
        att_weights = self.act(att_weights)
        att_1 = torch.sum(att_weights * cands, dim=1)
        
        c2 = F.softplus(self.c2[r])
        head2 = expmap0(u, c2)
        rot_2 = givens_rotations(rot_mat2, head2).view((-1, 1, self.rank))
        ref_2 = givens_reflection(ref_mat2, head2).view((-1, 1, self.rank))
        
        cands = torch.cat([ref_2, rot_2], dim=1)
        att_vec2 = self.att2(r).view((-1, 1, self.rank))
        att_weights = torch.sum(att_vec2 * cands * self.scale.to(cands.device), dim=-1, keepdim=True)
        att_weights = self.act(att_weights)
        att_2 = torch.sum(att_weights * cands, dim=1)
        
        res1=logmap0(att_1,c1).view((-1, 1, self.rank))
        res2=logmap0(att_2,c2).view((-1, 1, self.rank))
        
        rot_mat = self.rel_diag(r)
        head3 = u
        rot_q = givens_rotations(rot_mat, head3).view((-1, 1, self.rank))
        ref_q = givens_reflection(ref_mat, head3).view((-1, 1, self.rank))
        
        cands = torch.cat([ref_q, rot_q], dim=1)
        att_vec = self.att3(r).view((-1, 1, self.rank))
        att_weights = torch.sum(att_vec * cands * self.scale.to(cands.device), dim=-1, keepdim=True)
        att_weights = self.act(att_weights)
        res3 = torch.sum(att_weights * cands, dim=1).view((-1, 1, self.rank))
        
        cands = torch.cat([res1,res2,res3], dim=1)
        context_vec = self.context_vec(r).view((-1, 1, self.rank))
        att_weights = torch.sum(context_vec * cands * self.scale.to(cands.device), dim=-1, keepdim=True)
        att_weights = self.act(att_weights)
        att_q = torch.sum(att_weights * cands, dim=1)
        
        c = F.softplus(self.c[r])
        lhs = expmap0(att_q, c)
        rel = self.rel(r)
        rel_c = expmap0(rel, c)
        res = project(mobius_add(lhs, rel_c, c), c)
        
        return (res, c), self.bh(queries[:, 0])

    def get_queries_lp(self, queries, r):
        
        u = self.entity(queries[:, 0])
        rvh = self.rel(r)
        
        rot_mat = self.rel_diag(r)
        ref_mat = self.rel_diag1(r)
        rot_mat1 = self.rel_diag2(r)
        ref_mat1 = self.rel_diag3(r)
        rot_mat2 = self.rel_diag4(r)
        ref_mat2 = self.rel_diag5(r)

        c1 = F.softplus(self.c1[r])
        head1 = expmap0(u, c1)
        rot_1 = givens_rotations(rot_mat1, head1).view((-1, 1, self.rank))
        ref_1 = givens_reflection(ref_mat1, head1).view((-1, 1, self.rank))
        
        cands = torch.cat([ref_1, rot_1], dim=1)
        att_vec1 = self.att1(r).view((-1, 1, self.rank))
        att_weights = torch.sum(att_vec1 * cands * self.scale.to(cands.device), dim=-1, keepdim=True)
        att_weights = self.act(att_weights)
        att_1 = torch.sum(att_weights * cands, dim=1)
        
        c2 = F.softplus(self.c2[r])
        head2 = expmap0(u, c2)
        rot_2 = givens_rotations(rot_mat2, head2).view((-1, 1, self.rank))
        ref_2 = givens_reflection(ref_mat2, head2).view((-1, 1, self.rank))
        
        cands = torch.cat([ref_2, rot_2], dim=1)
        att_vec2 = self.att2(r).view((-1, 1, self.rank))
        att_weights = torch.sum(att_vec2 * cands * self.scale.to(cands.device), dim=-1, keepdim=True)
        att_weights = self.act(att_weights)
        att_2 = torch.sum(att_weights * cands, dim=1)
        
        res1=logmap0(att_1,c1).view((-1, 1, self.rank))
        res2=logmap0(att_2,c2).view((-1, 1, self.rank))
        
        rot_mat = self.rel_diag(r)
        head3 = u
        rot_q = givens_rotations(rot_mat, head3).view((-1, 1, self.rank))
        ref_q = givens_reflection(ref_mat, head3).view((-1, 1, self.rank))
        
        cands = torch.cat([ref_q, rot_q], dim=1)
        att_vec = self.att3(r).view((-1, 1, self.rank))
        att_weights = torch.sum(att_vec * cands * self.scale.to(cands.device), dim=-1, keepdim=True)
        att_weights = self.act(att_weights)
        res3 = torch.sum(att_weights * cands, dim=1).view((-1, 1, self.rank))
        
        cands = torch.cat([res1,res2,res3], dim=1)
        context_vec = self.context_vec(r).view((-1, 1, self.rank))
        att_weights = torch.sum(context_vec * cands * self.scale.to(cands.device), dim=-1, keepdim=True)
        att_weights = self.act(att_weights)
        att_q = torch.sum(att_weights * cands, dim=1)
        
        c = F.softplus(self.c[r])
        lhs = expmap0(att_q, c)
        rel = self.rel(r)
        rel_c = expmap0(rel, c)
        res = project(mobius_add(lhs, rel_c, c), c)
        
        return (res, c), self.bh(queries[:, 0])
    
class customATT2Att_1(BaseH):

    def __init__(self, args):
        super(customATT2Att_1, self).__init__(args)
        self.att1 = nn.Embedding(self.sizes[1], self.rank)
        self.att1.weight.data = self.init_size * torch.randn((self.sizes[1], self.rank), dtype=self.data_type)
        self.att3 = nn.Embedding(self.sizes[1], self.rank)
        self.att3.weight.data = self.init_size * torch.randn((self.sizes[1], self.rank), dtype=self.data_type)
        
        self.rel_diag3 = nn.Embedding(self.sizes[1], self.rank)
        self.rel_diag3.weight.data = 2 * torch.rand((self.sizes[1], self.rank), dtype=self.data_type) - 1.0

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
        rvh = self.rel(r)
        
        rot_mat = self.rel_diag(r)
        ref_mat = self.rel_diag1(r)
        rot_mat1 = self.rel_diag2(r)
        ref_mat1 = self.rel_diag3(r)

        c1 = F.softplus(self.c1[r])
        head1 = expmap0(u, c1)
        rot_1 = givens_rotations(rot_mat1, head1).view((-1, 1, self.rank))
        ref_1 = givens_reflection(ref_mat1, head1).view((-1, 1, self.rank))
        
        cands = torch.cat([ref_1, rot_1], dim=1)
        att_vec1 = self.att1(r).view((-1, 1, self.rank))
        att_weights = torch.sum(att_vec1 * cands * self.scale.to(cands.device), dim=-1, keepdim=True)
        att_weights = self.act(att_weights)
        att_1 = torch.sum(att_weights * cands, dim=1)
        
        res1=logmap0(att_1,c1).view((-1, 1, self.rank))
        
        rot_mat = self.rel_diag(r)
        head3 = u
        rot_q = givens_rotations(rot_mat, head3).view((-1, 1, self.rank))
        ref_q = givens_reflection(ref_mat, head3).view((-1, 1, self.rank))
        
        cands = torch.cat([ref_q, rot_q], dim=1)
        att_vec = self.att3(r).view((-1, 1, self.rank))
        att_weights = torch.sum(att_vec * cands * self.scale.to(cands.device), dim=-1, keepdim=True)
        att_weights = self.act(att_weights)
        res3 = torch.sum(att_weights * cands, dim=1).view((-1, 1, self.rank))
        
        cands = torch.cat([res1,res3], dim=1)
        context_vec = self.context_vec(r).view((-1, 1, self.rank))
        att_weights = torch.sum(context_vec * cands * self.scale.to(cands.device), dim=-1, keepdim=True)
        att_weights = self.act(att_weights)
        att_q = torch.sum(att_weights * cands, dim=1)
        
        c = F.softplus(self.c[r])
        lhs = expmap0(att_q, c)
        rel = self.rel(r)
        rel_c = expmap0(rel, c)
        res = project(mobius_add(lhs, rel_c, c), c)
        
        return (res, c), self.bh(queries[:, 0])

    def get_queries_lp(self, queries, r):
        
        u = self.entity(queries[:, 0])
        rvh = self.rel(r)
        
        rot_mat = self.rel_diag(r)
        ref_mat = self.rel_diag1(r)
        rot_mat1 = self.rel_diag2(r)
        ref_mat1 = self.rel_diag3(r)

        c1 = F.softplus(self.c1[r])
        head1 = expmap0(u, c1)
        rot_1 = givens_rotations(rot_mat1, head1).view((-1, 1, self.rank))
        ref_1 = givens_reflection(ref_mat1, head1).view((-1, 1, self.rank))
        
        cands = torch.cat([ref_1, rot_1], dim=1)
        att_vec1 = self.att1(r).view((-1, 1, self.rank))
        att_weights = torch.sum(att_vec1 * cands * self.scale.to(cands.device), dim=-1, keepdim=True)
        att_weights = self.act(att_weights)
        att_1 = torch.sum(att_weights * cands, dim=1)
        
        res1=logmap0(att_1,c1).view((-1, 1, self.rank))
        
        rot_mat = self.rel_diag(r)
        head3 = u
        rot_q = givens_rotations(rot_mat, head3).view((-1, 1, self.rank))
        ref_q = givens_reflection(ref_mat, head3).view((-1, 1, self.rank))
        
        cands = torch.cat([ref_q, rot_q], dim=1)
        att_vec = self.att3(r).view((-1, 1, self.rank))
        att_weights = torch.sum(att_vec * cands * self.scale.to(cands.device), dim=-1, keepdim=True)
        att_weights = self.act(att_weights)
        res3 = torch.sum(att_weights * cands, dim=1).view((-1, 1, self.rank))
        
        cands = torch.cat([res1,res3], dim=1)
        context_vec = self.context_vec(r).view((-1, 1, self.rank))
        att_weights = torch.sum(context_vec * cands * self.scale.to(cands.device), dim=-1, keepdim=True)
        att_weights = self.act(att_weights)
        att_q = torch.sum(att_weights * cands, dim=1)
        
        c = F.softplus(self.c[r])
        lhs = expmap0(att_q, c)
        rel = self.rel(r)
        rel_c = expmap0(rel, c)
        res = project(mobius_add(lhs, rel_c, c), c)
        
        return (res, c), self.bh(queries[:, 0])
    
class customGIEAtt_2(BaseH):

    def __init__(self, args):
        super(customGIEAtt_2, self).__init__(args)
        self.att1 = nn.Embedding(self.sizes[1], self.rank)
        self.att1.weight.data = self.init_size * torch.randn((self.sizes[1], self.rank), dtype=self.data_type)
        self.att2 = nn.Embedding(self.sizes[1], self.rank)
        self.att2.weight.data = self.init_size * torch.randn((self.sizes[1], self.rank), dtype=self.data_type)
        self.att3 = nn.Embedding(self.sizes[1], self.rank)
        self.att3.weight.data = self.init_size * torch.randn((self.sizes[1], self.rank), dtype=self.data_type)
        self.context_vec = nn.Embedding(self.sizes[1], self.rank)
        self.rel_diag3 = nn.Embedding(self.sizes[1], self.rank)
        self.rel_diag3.weight.data = 2 * torch.rand((self.sizes[1], self.rank), dtype=self.data_type) - 1.0
        self.rel_diag4 = nn.Embedding(self.sizes[1], self.rank)
        self.rel_diag4.weight.data = 2 * torch.rand((self.sizes[1], self.rank), dtype=self.data_type) - 1.0
        self.rel_diag5 = nn.Embedding(self.sizes[1], self.rank)
        self.rel_diag5.weight.data = 2 * torch.rand((self.sizes[1], self.rank), dtype=self.data_type) - 1.0
        self.context_vec.weight.data = self.init_size * torch.randn((self.sizes[1], self.rank), dtype=self.data_type)
        self.act = nn.Softmax(dim=1)
        if args.dtype == "double":
            self.scale = torch.Tensor([1. / np.sqrt(self.rank)]).double().cuda()
        else:
            self.scale = torch.Tensor([1. / np.sqrt(self.rank)]).cuda()

    def get_queries(self, queries):
        
        u = self.entity(queries[:, 0])
        r = queries[:, 1]
        rvh = self.rel(r)
        
        rot_mat = self.rel_diag(r)
        ref_mat = self.rel_diag1(r)
        rot_mat1 = self.rel_diag2(r)
        ref_mat1 = self.rel_diag3(r)
        rot_mat2 = self.rel_diag4(r)
        ref_mat2 = self.rel_diag5(r)

        c1 = F.softplus(self.c1[r])
        head1 = expmap0(u, c1)
        rel1 = expmap0(rvh, c1)
        head1 = mobius_add(head1, rel1, c1)
        head1 = logmap0(head1,c1)
        rot_1 = givens_rotations(rot_mat1, head1).view((-1, 1, self.rank))
        ref_1 = givens_reflection(ref_mat1, head1).view((-1, 1, self.rank))
        
        cands = torch.cat([ref_1, rot_1], dim=1)
        att_vec1 = self.att1(r).view((-1, 1, self.rank))
        att_weights = torch.sum(att_vec1 * cands * self.scale.to(cands.device), dim=-1, keepdim=True)
        att_weights = self.act(att_weights)
        att_1 = torch.sum(att_weights * cands, dim=1)
        
        c2 = F.softplus(self.c2[r])
        head2 = expmap0(u, c2)
        rel2 = expmap0(rvh, c2)
        head2 = mobius_add(head2, rel2, c2)
        head2 = logmap0(head2,c2)
        rot_2 = givens_rotations(rot_mat2, head2).view((-1, 1, self.rank))
        ref_2 = givens_reflection(ref_mat2, head2).view((-1, 1, self.rank))
        
        cands = torch.cat([ref_2, rot_2], dim=1)
        att_vec2 = self.att2(r).view((-1, 1, self.rank))
        att_weights = torch.sum(att_vec2 * cands * self.scale.to(cands.device), dim=-1, keepdim=True)
        att_weights = self.act(att_weights)
        att_2 = torch.sum(att_weights * cands, dim=1)
        
        res1=att_1.view((-1, 1, self.rank))
        res2=att_2.view((-1, 1, self.rank))
        
        rot_mat = self.rel_diag(r)
        head3 = u
        rot_q = givens_rotations(rot_mat, head3).view((-1, 1, self.rank))
        ref_q = givens_reflection(ref_mat, head3).view((-1, 1, self.rank))
        
        cands = torch.cat([ref_q, rot_q], dim=1)
        att_vec = self.att3(r).view((-1, 1, self.rank))
        att_weights = torch.sum(att_vec * cands * self.scale.to(cands.device), dim=-1, keepdim=True)
        att_weights = self.act(att_weights)
        res3 = torch.sum(att_weights * cands, dim=1).view((-1, 1, self.rank))
        
        cands = torch.cat([res1,res2,res3], dim=1)
        context_vec = self.context_vec(r).view((-1, 1, self.rank))
        att_weights = torch.sum(context_vec * cands * self.scale.to(cands.device), dim=-1, keepdim=True)
        att_weights = self.act(att_weights)
        att_q = torch.sum(att_weights * cands, dim=1)
        
        c = F.softplus(self.c[r])
        lhs = expmap0(att_q, c)
        rel = self.rel(r)
        rel_c = expmap0(rel, c)
        res = project(mobius_add(lhs, rel_c, c), c)
        
        return (res, c), self.bh(queries[:, 0])

    def get_queries_lp(self, queries, r):
        
        u = self.entity(queries[:, 0])
        rvh = self.rel(r)
        
        rot_mat = self.rel_diag(r)
        ref_mat = self.rel_diag1(r)
        rot_mat1 = self.rel_diag2(r)
        ref_mat1 = self.rel_diag3(r)
        rot_mat2 = self.rel_diag4(r)
        ref_mat2 = self.rel_diag5(r)

        c1 = F.softplus(self.c1[r])
        head1 = expmap0(u, c1)
        rel1 = expmap0(rvh, c1)
        head1 = mobius_add(head1, rel1, c1)
        head1 = logmap0(head1,c1)
        rot_1 = givens_rotations(rot_mat1, head1).view((-1, 1, self.rank))
        ref_1 = givens_reflection(ref_mat1, head1).view((-1, 1, self.rank))
        
        cands = torch.cat([ref_1, rot_1], dim=1)
        att_vec1 = self.att1(r).view((-1, 1, self.rank))
        att_weights = torch.sum(att_vec1 * cands * self.scale.to(cands.device), dim=-1, keepdim=True)
        att_weights = self.act(att_weights)
        att_1 = torch.sum(att_weights * cands, dim=1)
        
        c2 = F.softplus(self.c2[r])
        head2 = expmap0(u, c2)
        rel2 = expmap0(rvh, c2)
        head2 = mobius_add(head2, rel2, c2)
        head2 = logmap0(head2,c2)
        rot_2 = givens_rotations(rot_mat2, head2).view((-1, 1, self.rank))
        ref_2 = givens_reflection(ref_mat2, head2).view((-1, 1, self.rank))
        
        cands = torch.cat([ref_2, rot_2], dim=1)
        att_vec2 = self.att2(r).view((-1, 1, self.rank))
        att_weights = torch.sum(att_vec2 * cands * self.scale.to(cands.device), dim=-1, keepdim=True)
        att_weights = self.act(att_weights)
        att_2 = torch.sum(att_weights * cands, dim=1)
        
        res1=att_1.view((-1, 1, self.rank))
        res2=att_2.view((-1, 1, self.rank))
        
        rot_mat = self.rel_diag(r)
        head3 = u
        rot_q = givens_rotations(rot_mat, head3).view((-1, 1, self.rank))
        ref_q = givens_reflection(ref_mat, head3).view((-1, 1, self.rank))
        
        cands = torch.cat([ref_q, rot_q], dim=1)
        att_vec = self.att3(r).view((-1, 1, self.rank))
        att_weights = torch.sum(att_vec * cands * self.scale.to(cands.device), dim=-1, keepdim=True)
        att_weights = self.act(att_weights)
        res3 = torch.sum(att_weights * cands, dim=1).view((-1, 1, self.rank))
        
        cands = torch.cat([res1,res2,res3], dim=1)
        context_vec = self.context_vec(r).view((-1, 1, self.rank))
        att_weights = torch.sum(context_vec * cands * self.scale.to(cands.device), dim=-1, keepdim=True)
        att_weights = self.act(att_weights)
        att_q = torch.sum(att_weights * cands, dim=1)
        
        c = F.softplus(self.c[r])
        lhs = expmap0(att_q, c)
        rel = self.rel(r)
        rel_c = expmap0(rel, c)
        res = project(mobius_add(lhs, rel_c, c), c)
        
        return (res, c), self.bh(queries[:, 0])
    
class customATT2Att_2(BaseH):

    def __init__(self, args):
        super(customATT2Att_2, self).__init__(args)
        self.att1 = nn.Embedding(self.sizes[1], self.rank)
        self.att1.weight.data = self.init_size * torch.randn((self.sizes[1], self.rank), dtype=self.data_type)
        self.att3 = nn.Embedding(self.sizes[1], self.rank)
        self.att3.weight.data = self.init_size * torch.randn((self.sizes[1], self.rank), dtype=self.data_type)
        
        self.rel_diag3 = nn.Embedding(self.sizes[1], self.rank)
        self.rel_diag3.weight.data = 2 * torch.rand((self.sizes[1], self.rank), dtype=self.data_type) - 1.0

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
        rvh = self.rel(r)
        
        rot_mat = self.rel_diag(r)
        ref_mat = self.rel_diag1(r)
        rot_mat1 = self.rel_diag2(r)
        ref_mat1 = self.rel_diag3(r)

        c1 = F.softplus(self.c1[r])
        head1 = expmap0(u, c1)
        rel1 = expmap0(rvh, c1)
        head1 = mobius_add(head1, rel1, c1)
        head1 = logmap0(head1,c1)
        rot_1 = givens_rotations(rot_mat1, head1).view((-1, 1, self.rank))
        ref_1 = givens_reflection(ref_mat1, head1).view((-1, 1, self.rank))
        
        cands = torch.cat([ref_1, rot_1], dim=1)
        att_vec1 = self.att1(r).view((-1, 1, self.rank))
        att_weights = torch.sum(att_vec1 * cands * self.scale.to(cands.device), dim=-1, keepdim=True)
        att_weights = self.act(att_weights)
        att_1 = torch.sum(att_weights * cands, dim=1)
        
        res1=att_1.view((-1, 1, self.rank))
        
        rot_mat = self.rel_diag(r)
        head3 = u
        rot_q = givens_rotations(rot_mat, head3).view((-1, 1, self.rank))
        ref_q = givens_reflection(ref_mat, head3).view((-1, 1, self.rank))
        
        cands = torch.cat([ref_q, rot_q], dim=1)
        att_vec = self.att3(r).view((-1, 1, self.rank))
        att_weights = torch.sum(att_vec * cands * self.scale.to(cands.device), dim=-1, keepdim=True)
        att_weights = self.act(att_weights)
        res3 = torch.sum(att_weights * cands, dim=1).view((-1, 1, self.rank))
        
        cands = torch.cat([res1,res3], dim=1)
        context_vec = self.context_vec(r).view((-1, 1, self.rank))
        att_weights = torch.sum(context_vec * cands * self.scale.to(cands.device), dim=-1, keepdim=True)
        att_weights = self.act(att_weights)
        att_q = torch.sum(att_weights * cands, dim=1)
        
        c = F.softplus(self.c[r])
        lhs = expmap0(att_q, c)
        rel = self.rel(r)
        rel_c = expmap0(rel, c)
        res = project(mobius_add(lhs, rel_c, c), c)
        
        return (res, c), self.bh(queries[:, 0])

    def get_queries_lp(self, queries, r):
        
        u = self.entity(queries[:, 0])
        rvh = self.rel(r)
        
        rot_mat = self.rel_diag(r)
        ref_mat = self.rel_diag1(r)
        rot_mat1 = self.rel_diag2(r)
        ref_mat1 = self.rel_diag3(r)

        c1 = F.softplus(self.c1[r])
        head1 = expmap0(u, c1)
        rel1 = expmap0(rvh, c1)
        head1 = mobius_add(head1, rel1, c1)
        head1 = logmap0(head1,c1)
        rot_1 = givens_rotations(rot_mat1, head1).view((-1, 1, self.rank))
        ref_1 = givens_reflection(ref_mat1, head1).view((-1, 1, self.rank))
        
        cands = torch.cat([ref_1, rot_1], dim=1)
        att_vec1 = self.att1(r).view((-1, 1, self.rank))
        att_weights = torch.sum(att_vec1 * cands * self.scale.to(cands.device), dim=-1, keepdim=True)
        att_weights = self.act(att_weights)
        att_1 = torch.sum(att_weights * cands, dim=1)
        
        res1=att_1.view((-1, 1, self.rank))
        
        rot_mat = self.rel_diag(r)
        head3 = u
        rot_q = givens_rotations(rot_mat, head3).view((-1, 1, self.rank))
        ref_q = givens_reflection(ref_mat, head3).view((-1, 1, self.rank))
        
        cands = torch.cat([ref_q, rot_q], dim=1)
        att_vec = self.att3(r).view((-1, 1, self.rank))
        att_weights = torch.sum(att_vec * cands * self.scale.to(cands.device), dim=-1, keepdim=True)
        att_weights = self.act(att_weights)
        res3 = torch.sum(att_weights * cands, dim=1).view((-1, 1, self.rank))
        
        cands = torch.cat([res1,res3], dim=1)
        context_vec = self.context_vec(r).view((-1, 1, self.rank))
        att_weights = torch.sum(context_vec * cands * self.scale.to(cands.device), dim=-1, keepdim=True)
        att_weights = self.act(att_weights)
        att_q = torch.sum(att_weights * cands, dim=1)
        
        c = F.softplus(self.c[r])
        lhs = expmap0(att_q, c)
        rel = self.rel(r)
        rel_c = expmap0(rel, c)
        res = project(mobius_add(lhs, rel_c, c), c)
        
        return (res, c), self.bh(queries[:, 0])
    
class customGIEAtt_3(BaseH):

    def __init__(self, args):
        super(customGIEAtt_3, self).__init__(args)
        self.att1 = nn.Embedding(self.sizes[1], self.rank)
        self.att1.weight.data = self.init_size * torch.randn((self.sizes[1], self.rank), dtype=self.data_type)
        self.att2 = nn.Embedding(self.sizes[1], self.rank)
        self.att2.weight.data = self.init_size * torch.randn((self.sizes[1], self.rank), dtype=self.data_type)
        self.att3 = nn.Embedding(self.sizes[1], self.rank)
        self.att3.weight.data = self.init_size * torch.randn((self.sizes[1], self.rank), dtype=self.data_type)
        self.context_vec = nn.Embedding(self.sizes[1], self.rank)
        self.rel_diag3 = nn.Embedding(self.sizes[1], self.rank)
        self.rel_diag3.weight.data = 2 * torch.rand((self.sizes[1], self.rank), dtype=self.data_type) - 1.0
        self.rel_diag4 = nn.Embedding(self.sizes[1], self.rank)
        self.rel_diag4.weight.data = 2 * torch.rand((self.sizes[1], self.rank), dtype=self.data_type) - 1.0
        self.rel_diag5 = nn.Embedding(self.sizes[1], self.rank)
        self.rel_diag5.weight.data = 2 * torch.rand((self.sizes[1], self.rank), dtype=self.data_type) - 1.0
        self.context_vec.weight.data = self.init_size * torch.randn((self.sizes[1], self.rank), dtype=self.data_type)
        self.act = nn.Softmax(dim=1)
        if args.dtype == "double":
            self.scale = torch.Tensor([1. / np.sqrt(self.rank)]).double().cuda()
        else:
            self.scale = torch.Tensor([1. / np.sqrt(self.rank)]).cuda()

    def get_queries(self, queries):
        
        u = self.entity(queries[:, 0])
        r = queries[:, 1]
        rvh = self.rel(r)
        
        rot_mat = self.rel_diag(r)
        ref_mat = self.rel_diag1(r)
        rot_mat1 = self.rel_diag2(r)
        ref_mat1 = self.rel_diag3(r)
        rot_mat2 = self.rel_diag4(r)
        ref_mat2 = self.rel_diag5(r)

        c1 = F.softplus(self.c1[r])
        head1 = expmap0(u, c1)
        rel1 = expmap0(rvh, c1)
        head1 = project(mobius_add(head1, rel1, c1), c1)
        head1 = logmap0(head1,c1)
        rot_1 = givens_rotations(rot_mat1, head1).view((-1, 1, self.rank))
        ref_1 = givens_reflection(ref_mat1, head1).view((-1, 1, self.rank))
        
        cands = torch.cat([ref_1, rot_1], dim=1)
        att_vec1 = self.att1(r).view((-1, 1, self.rank))
        att_weights = torch.sum(att_vec1 * cands * self.scale.to(cands.device), dim=-1, keepdim=True)
        att_weights = self.act(att_weights)
        att_1 = torch.sum(att_weights * cands, dim=1)
        
        c2 = F.softplus(self.c2[r])
        head2 = expmap0(u, c2)
        rel2 = expmap0(rvh, c2)
        head2 = project(mobius_add(head2, rel2, c2), c2)
        head2 = logmap0(head2,c2)
        rot_2 = givens_rotations(rot_mat2, head2).view((-1, 1, self.rank))
        ref_2 = givens_reflection(ref_mat2, head2).view((-1, 1, self.rank))
        
        cands = torch.cat([ref_2, rot_2], dim=1)
        att_vec2 = self.att2(r).view((-1, 1, self.rank))
        att_weights = torch.sum(att_vec2 * cands * self.scale.to(cands.device), dim=-1, keepdim=True)
        att_weights = self.act(att_weights)
        att_2 = torch.sum(att_weights * cands, dim=1)
        
        res1=att_1.view((-1, 1, self.rank))
        res2=att_2.view((-1, 1, self.rank))
        
        rot_mat = self.rel_diag(r)
        head3 = u
        rot_q = givens_rotations(rot_mat, head3).view((-1, 1, self.rank))
        ref_q = givens_reflection(ref_mat, head3).view((-1, 1, self.rank))
        
        cands = torch.cat([ref_q, rot_q], dim=1)
        att_vec = self.att3(r).view((-1, 1, self.rank))
        att_weights = torch.sum(att_vec * cands * self.scale.to(cands.device), dim=-1, keepdim=True)
        att_weights = self.act(att_weights)
        res3 = torch.sum(att_weights * cands, dim=1).view((-1, 1, self.rank))
        
        cands = torch.cat([res1,res2,res3], dim=1)
        context_vec = self.context_vec(r).view((-1, 1, self.rank))
        att_weights = torch.sum(context_vec * cands * self.scale.to(cands.device), dim=-1, keepdim=True)
        att_weights = self.act(att_weights)
        att_q = torch.sum(att_weights * cands, dim=1)
        
        c = F.softplus(self.c[r])
        lhs = expmap0(att_q, c)
        rel = self.rel(r)
        rel_c = expmap0(rel, c)
        res = project(mobius_add(lhs, rel_c, c), c)
        
        return (res, c), self.bh(queries[:, 0])

    def get_queries_lp(self, queries, r):
        
        u = self.entity(queries[:, 0])
        rvh = self.rel(r)
        
        rot_mat = self.rel_diag(r)
        ref_mat = self.rel_diag1(r)
        rot_mat1 = self.rel_diag2(r)
        ref_mat1 = self.rel_diag3(r)
        rot_mat2 = self.rel_diag4(r)
        ref_mat2 = self.rel_diag5(r)

        c1 = F.softplus(self.c1[r])
        head1 = expmap0(u, c1)
        rel1 = expmap0(rvh, c1)
        head1 = project(mobius_add(head1, rel1, c1), c1)
        head1 = logmap0(head1,c1)
        rot_1 = givens_rotations(rot_mat1, head1).view((-1, 1, self.rank))
        ref_1 = givens_reflection(ref_mat1, head1).view((-1, 1, self.rank))
        
        cands = torch.cat([ref_1, rot_1], dim=1)
        att_vec1 = self.att1(r).view((-1, 1, self.rank))
        att_weights = torch.sum(att_vec1 * cands * self.scale.to(cands.device), dim=-1, keepdim=True)
        att_weights = self.act(att_weights)
        att_1 = torch.sum(att_weights * cands, dim=1)
        
        c2 = F.softplus(self.c2[r])
        head2 = expmap0(u, c2)
        rel2 = expmap0(rvh, c2)
        head2 = project(mobius_add(head2, rel2, c2), c2)
        head2 = logmap0(head2,c2)
        rot_2 = givens_rotations(rot_mat2, head2).view((-1, 1, self.rank))
        ref_2 = givens_reflection(ref_mat2, head2).view((-1, 1, self.rank))
        
        cands = torch.cat([ref_2, rot_2], dim=1)
        att_vec2 = self.att2(r).view((-1, 1, self.rank))
        att_weights = torch.sum(att_vec2 * cands * self.scale.to(cands.device), dim=-1, keepdim=True)
        att_weights = self.act(att_weights)
        att_2 = torch.sum(att_weights * cands, dim=1)
        
        res1=att_1.view((-1, 1, self.rank))
        res2=att_2.view((-1, 1, self.rank))
        
        rot_mat = self.rel_diag(r)
        head3 = u
        rot_q = givens_rotations(rot_mat, head3).view((-1, 1, self.rank))
        ref_q = givens_reflection(ref_mat, head3).view((-1, 1, self.rank))
        
        cands = torch.cat([ref_q, rot_q], dim=1)
        att_vec = self.att3(r).view((-1, 1, self.rank))
        att_weights = torch.sum(att_vec * cands * self.scale.to(cands.device), dim=-1, keepdim=True)
        att_weights = self.act(att_weights)
        res3 = torch.sum(att_weights * cands, dim=1).view((-1, 1, self.rank))
        
        cands = torch.cat([res1,res2,res3], dim=1)
        context_vec = self.context_vec(r).view((-1, 1, self.rank))
        att_weights = torch.sum(context_vec * cands * self.scale.to(cands.device), dim=-1, keepdim=True)
        att_weights = self.act(att_weights)
        att_q = torch.sum(att_weights * cands, dim=1)
        
        c = F.softplus(self.c[r])
        lhs = expmap0(att_q, c)
        rel = self.rel(r)
        rel_c = expmap0(rel, c)
        res = project(mobius_add(lhs, rel_c, c), c)
        
        return (res, c), self.bh(queries[:, 0])
    
class customATT2Att_3(BaseH):

    def __init__(self, args):
        super(customGIEAtt_3, self).__init__(args)
        self.att1 = nn.Embedding(self.sizes[1], self.rank)
        self.att1.weight.data = self.init_size * torch.randn((self.sizes[1], self.rank), dtype=self.data_type)
        self.att3 = nn.Embedding(self.sizes[1], self.rank)
        self.att3.weight.data = self.init_size * torch.randn((self.sizes[1], self.rank), dtype=self.data_type)
        self.context_vec = nn.Embedding(self.sizes[1], self.rank)
        self.rel_diag3 = nn.Embedding(self.sizes[1], self.rank)
        self.rel_diag3.weight.data = 2 * torch.rand((self.sizes[1], self.rank), dtype=self.data_type) - 1.0
        
        self.context_vec.weight.data = self.init_size * torch.randn((self.sizes[1], self.rank), dtype=self.data_type)
        self.act = nn.Softmax(dim=1)
        if args.dtype == "double":
            self.scale = torch.Tensor([1. / np.sqrt(self.rank)]).double().cuda()
        else:
            self.scale = torch.Tensor([1. / np.sqrt(self.rank)]).cuda()

    def get_queries(self, queries):
        
        u = self.entity(queries[:, 0])
        r = queries[:, 1]
        rvh = self.rel(r)
        
        rot_mat = self.rel_diag(r)
        ref_mat = self.rel_diag1(r)
        rot_mat1 = self.rel_diag2(r)
        ref_mat1 = self.rel_diag3(r)
        rot_mat2 = self.rel_diag4(r)
        ref_mat2 = self.rel_diag5(r)

        c1 = F.softplus(self.c1[r])
        head1 = expmap0(u, c1)
        rel1 = expmap0(rvh, c1)
        head1 = project(mobius_add(head1, rel1, c1), c1)
        head1 = logmap0(head1,c1)
        rot_1 = givens_rotations(rot_mat1, head1).view((-1, 1, self.rank))
        ref_1 = givens_reflection(ref_mat1, head1).view((-1, 1, self.rank))
        
        cands = torch.cat([ref_1, rot_1], dim=1)
        att_vec1 = self.att1(r).view((-1, 1, self.rank))
        att_weights = torch.sum(att_vec1 * cands * self.scale.to(cands.device), dim=-1, keepdim=True)
        att_weights = self.act(att_weights)
        att_1 = torch.sum(att_weights * cands, dim=1)
        
        res1=att_1.view((-1, 1, self.rank))
        
        rot_mat = self.rel_diag(r)
        head3 = u
        rot_q = givens_rotations(rot_mat, head3).view((-1, 1, self.rank))
        ref_q = givens_reflection(ref_mat, head3).view((-1, 1, self.rank))
        
        cands = torch.cat([ref_q, rot_q], dim=1)
        att_vec = self.att3(r).view((-1, 1, self.rank))
        att_weights = torch.sum(att_vec * cands * self.scale.to(cands.device), dim=-1, keepdim=True)
        att_weights = self.act(att_weights)
        res3 = torch.sum(att_weights * cands, dim=1).view((-1, 1, self.rank))
        
        cands = torch.cat([res1,res3], dim=1)
        context_vec = self.context_vec(r).view((-1, 1, self.rank))
        att_weights = torch.sum(context_vec * cands * self.scale.to(cands.device), dim=-1, keepdim=True)
        att_weights = self.act(att_weights)
        att_q = torch.sum(att_weights * cands, dim=1)
        
        c = F.softplus(self.c[r])
        lhs = expmap0(att_q, c)
        rel = self.rel(r)
        rel_c = expmap0(rel, c)
        res = project(mobius_add(lhs, rel_c, c), c)
        
        return (res, c), self.bh(queries[:, 0])

    def get_queries_lp(self, queries, r):
        
        u = self.entity(queries[:, 0])
        rvh = self.rel(r)
        
        rot_mat = self.rel_diag(r)
        ref_mat = self.rel_diag1(r)
        rot_mat1 = self.rel_diag2(r)
        ref_mat1 = self.rel_diag3(r)
        rot_mat2 = self.rel_diag4(r)
        ref_mat2 = self.rel_diag5(r)

        c1 = F.softplus(self.c1[r])
        head1 = expmap0(u, c1)
        rel1 = expmap0(rvh, c1)
        head1 = project(mobius_add(head1, rel1, c1), c1)
        head1 = logmap0(head1,c1)
        rot_1 = givens_rotations(rot_mat1, head1).view((-1, 1, self.rank))
        ref_1 = givens_reflection(ref_mat1, head1).view((-1, 1, self.rank))
        
        cands = torch.cat([ref_1, rot_1], dim=1)
        att_vec1 = self.att1(r).view((-1, 1, self.rank))
        att_weights = torch.sum(att_vec1 * cands * self.scale.to(cands.device), dim=-1, keepdim=True)
        att_weights = self.act(att_weights)
        att_1 = torch.sum(att_weights * cands, dim=1)
        
        res1=att_1.view((-1, 1, self.rank))
        
        rot_mat = self.rel_diag(r)
        head3 = u
        rot_q = givens_rotations(rot_mat, head3).view((-1, 1, self.rank))
        ref_q = givens_reflection(ref_mat, head3).view((-1, 1, self.rank))
        
        cands = torch.cat([ref_q, rot_q], dim=1)
        att_vec = self.att3(r).view((-1, 1, self.rank))
        att_weights = torch.sum(att_vec * cands * self.scale.to(cands.device), dim=-1, keepdim=True)
        att_weights = self.act(att_weights)
        res3 = torch.sum(att_weights * cands, dim=1).view((-1, 1, self.rank))
        
        cands = torch.cat([res1,res3], dim=1)
        context_vec = self.context_vec(r).view((-1, 1, self.rank))
        att_weights = torch.sum(context_vec * cands * self.scale.to(cands.device), dim=-1, keepdim=True)
        att_weights = self.act(att_weights)
        att_q = torch.sum(att_weights * cands, dim=1)
        
        c = F.softplus(self.c[r])
        lhs = expmap0(att_q, c)
        rel = self.rel(r)
        rel_c = expmap0(rel, c)
        res = project(mobius_add(lhs, rel_c, c), c)
        
        return (res, c), self.bh(queries[:, 0])
    
class customGIEAtt_4(BaseH):

    def __init__(self, args):
        super(customGIEAtt_4, self).__init__(args)
        self.att1 = nn.Embedding(self.sizes[1], self.rank)
        self.att1.weight.data = self.init_size * torch.randn((self.sizes[1], self.rank), dtype=self.data_type)
        self.att2 = nn.Embedding(self.sizes[1], self.rank)
        self.att2.weight.data = self.init_size * torch.randn((self.sizes[1], self.rank), dtype=self.data_type)
        self.att3 = nn.Embedding(self.sizes[1], self.rank)
        self.att3.weight.data = self.init_size * torch.randn((self.sizes[1], self.rank), dtype=self.data_type)
        self.context_vec = nn.Embedding(self.sizes[1], self.rank)
        self.rel_diag3 = nn.Embedding(self.sizes[1], self.rank)
        self.rel_diag3.weight.data = 2 * torch.rand((self.sizes[1], self.rank), dtype=self.data_type) - 1.0
        self.rel_diag4 = nn.Embedding(self.sizes[1], self.rank)
        self.rel_diag4.weight.data = 2 * torch.rand((self.sizes[1], self.rank), dtype=self.data_type) - 1.0
        self.rel_diag5 = nn.Embedding(self.sizes[1], self.rank)
        self.rel_diag5.weight.data = 2 * torch.rand((self.sizes[1], self.rank), dtype=self.data_type) - 1.0
        self.context_vec.weight.data = self.init_size * torch.randn((self.sizes[1], self.rank), dtype=self.data_type)
        self.act = nn.Softmax(dim=1)
        if args.dtype == "double":
            self.scale = torch.Tensor([1. / np.sqrt(self.rank)]).double().cuda()
        else:
            self.scale = torch.Tensor([1. / np.sqrt(self.rank)]).cuda()

    def get_queries(self, queries):
        
        u = self.entity(queries[:, 0])
        r = queries[:, 1]
        rvh = self.rel(r)
        
        rot_mat = self.rel_diag(r)
        ref_mat = self.rel_diag1(r)
        rot_mat1 = self.rel_diag2(r)
        ref_mat1 = self.rel_diag3(r)
        rot_mat2 = self.rel_diag4(r)
        ref_mat2 = self.rel_diag5(r)

        c1 = F.softplus(self.c1[r])
        head1 = expmap0(u, c1)
        rel1 = expmap0(rvh, c1)
        head1 = project(mobius_add(head1, rel1, c1), c1)
        head1 = logmap0(head1,c1)
        rot_1 = givens_rotations(rot_mat1, head1).view((-1, 1, self.rank))
        ref_1 = givens_reflection(ref_mat1, head1).view((-1, 1, self.rank))
        
        cands = torch.cat([ref_1, rot_1], dim=1)
        att_vec1 = self.att1(r).view((-1, 1, self.rank))
        att_weights = torch.sum(att_vec1 * cands * self.scale.to(cands.device), dim=-1, keepdim=True)
        att_weights = self.act(att_weights)
        att_1 = torch.sum(att_weights * cands, dim=1)
        
        c2 = F.softplus(self.c2[r])
        head2 = expmap0(u, c2)
        rel2 = expmap0(rvh, c2)
        head2 = project(mobius_add(head2, rel2, c2), c2)
        head2 = logmap0(head2,c2)
        rot_2 = givens_rotations(rot_mat2, head2).view((-1, 1, self.rank))
        ref_2 = givens_reflection(ref_mat2, head2).view((-1, 1, self.rank))
        
        cands = torch.cat([ref_2, rot_2], dim=1)
        att_vec2 = self.att2(r).view((-1, 1, self.rank))
        att_weights = torch.sum(att_vec2 * cands * self.scale.to(cands.device), dim=-1, keepdim=True)
        att_weights = self.act(att_weights)
        att_2 = torch.sum(att_weights * cands, dim=1)
        
        res1=att_1.view((-1, 1, self.rank))
        res2=att_2.view((-1, 1, self.rank))
        
        rot_mat = self.rel_diag(r)
        head3 = u+rvh
        rot_q = givens_rotations(rot_mat, head3).view((-1, 1, self.rank))
        ref_q = givens_reflection(ref_mat, head3).view((-1, 1, self.rank))
        
        cands = torch.cat([ref_q, rot_q], dim=1)
        att_vec = self.att3(r).view((-1, 1, self.rank))
        att_weights = torch.sum(att_vec * cands * self.scale.to(cands.device), dim=-1, keepdim=True)
        att_weights = self.act(att_weights)
        res3 = torch.sum(att_weights * cands, dim=1).view((-1, 1, self.rank))
        
        cands = torch.cat([res1,res2,res3], dim=1)
        context_vec = self.context_vec(r).view((-1, 1, self.rank))
        att_weights = torch.sum(context_vec * cands * self.scale.to(cands.device), dim=-1, keepdim=True)
        att_weights = self.act(att_weights)
        att_q = torch.sum(att_weights * cands, dim=1)
        
        c = F.softplus(self.c[r])
        lhs = expmap0(att_q, c)
        rel = self.rel(r)
        rel_c = expmap0(rel, c)
        res = project(mobius_add(lhs, rel_c, c), c)
        
        return (res, c), self.bh(queries[:, 0])

    def get_queries_lp(self, queries, r):
        
        u = self.entity(queries[:, 0])
        rvh = self.rel(r)
        
        rot_mat = self.rel_diag(r)
        ref_mat = self.rel_diag1(r)
        rot_mat1 = self.rel_diag2(r)
        ref_mat1 = self.rel_diag3(r)
        rot_mat2 = self.rel_diag4(r)
        ref_mat2 = self.rel_diag5(r)

        c1 = F.softplus(self.c1[r])
        head1 = expmap0(u, c1)
        rel1 = expmap0(rvh, c1)
        head1 = project(mobius_add(head1, rel1, c1), c1)
        head1 = logmap0(head1,c1)
        rot_1 = givens_rotations(rot_mat1, head1).view((-1, 1, self.rank))
        ref_1 = givens_reflection(ref_mat1, head1).view((-1, 1, self.rank))
        
        cands = torch.cat([ref_1, rot_1], dim=1)
        att_vec1 = self.att1(r).view((-1, 1, self.rank))
        att_weights = torch.sum(att_vec1 * cands * self.scale.to(cands.device), dim=-1, keepdim=True)
        att_weights = self.act(att_weights)
        att_1 = torch.sum(att_weights * cands, dim=1)
        
        c2 = F.softplus(self.c2[r])
        head2 = expmap0(u, c2)
        rel2 = expmap0(rvh, c2)
        head2 = project(mobius_add(head2, rel2, c2), c2)
        head2 = logmap0(head2,c2)
        rot_2 = givens_rotations(rot_mat2, head2).view((-1, 1, self.rank))
        ref_2 = givens_reflection(ref_mat2, head2).view((-1, 1, self.rank))
        
        cands = torch.cat([ref_2, rot_2], dim=1)
        att_vec2 = self.att2(r).view((-1, 1, self.rank))
        att_weights = torch.sum(att_vec2 * cands * self.scale.to(cands.device), dim=-1, keepdim=True)
        att_weights = self.act(att_weights)
        att_2 = torch.sum(att_weights * cands, dim=1)
        
        res1=att_1.view((-1, 1, self.rank))
        res2=att_2.view((-1, 1, self.rank))
        
        rot_mat = self.rel_diag(r)
        head3 = u+rvh
        rot_q = givens_rotations(rot_mat, head3).view((-1, 1, self.rank))
        ref_q = givens_reflection(ref_mat, head3).view((-1, 1, self.rank))
        
        cands = torch.cat([ref_q, rot_q], dim=1)
        att_vec = self.att3(r).view((-1, 1, self.rank))
        att_weights = torch.sum(att_vec * cands * self.scale.to(cands.device), dim=-1, keepdim=True)
        att_weights = self.act(att_weights)
        res3 = torch.sum(att_weights * cands, dim=1).view((-1, 1, self.rank))
        
        cands = torch.cat([res1,res2,res3], dim=1)
        context_vec = self.context_vec(r).view((-1, 1, self.rank))
        att_weights = torch.sum(context_vec * cands * self.scale.to(cands.device), dim=-1, keepdim=True)
        att_weights = self.act(att_weights)
        att_q = torch.sum(att_weights * cands, dim=1)
        
        c = F.softplus(self.c[r])
        lhs = expmap0(att_q, c)
        rel = self.rel(r)
        rel_c = expmap0(rel, c)
        res = project(mobius_add(lhs, rel_c, c), c)
        
        return (res, c), self.bh(queries[:, 0])
    
class customATT2Att_4(BaseH):

    def __init__(self, args):
        super(customGIEAtt_4, self).__init__(args)
        self.att1 = nn.Embedding(self.sizes[1], self.rank)
        self.att1.weight.data = self.init_size * torch.randn((self.sizes[1], self.rank), dtype=self.data_type)
        self.att3 = nn.Embedding(self.sizes[1], self.rank)
        self.att3.weight.data = self.init_size * torch.randn((self.sizes[1], self.rank), dtype=self.data_type)
        self.context_vec = nn.Embedding(self.sizes[1], self.rank)
        self.rel_diag3 = nn.Embedding(self.sizes[1], self.rank)
        self.rel_diag3.weight.data = 2 * torch.rand((self.sizes[1], self.rank), dtype=self.data_type) - 1.0
        
        self.context_vec.weight.data = self.init_size * torch.randn((self.sizes[1], self.rank), dtype=self.data_type)
        self.act = nn.Softmax(dim=1)
        if args.dtype == "double":
            self.scale = torch.Tensor([1. / np.sqrt(self.rank)]).double().cuda()
        else:
            self.scale = torch.Tensor([1. / np.sqrt(self.rank)]).cuda()

    def get_queries(self, queries):
        
        u = self.entity(queries[:, 0])
        r = queries[:, 1]
        rvh = self.rel(r)
        
        rot_mat = self.rel_diag(r)
        ref_mat = self.rel_diag1(r)
        rot_mat1 = self.rel_diag2(r)
        ref_mat1 = self.rel_diag3(r)

        c1 = F.softplus(self.c1[r])
        head1 = expmap0(u, c1)
        rel1 = expmap0(rvh, c1)
        head1 = project(mobius_add(head1, rel1, c1), c1)
        head1 = logmap0(head1,c1)
        rot_1 = givens_rotations(rot_mat1, head1).view((-1, 1, self.rank))
        ref_1 = givens_reflection(ref_mat1, head1).view((-1, 1, self.rank))
        
        cands = torch.cat([ref_1, rot_1], dim=1)
        att_vec1 = self.att1(r).view((-1, 1, self.rank))
        att_weights = torch.sum(att_vec1 * cands * self.scale.to(cands.device), dim=-1, keepdim=True)
        att_weights = self.act(att_weights)
        att_1 = torch.sum(att_weights * cands, dim=1)
        
        res1=att_1.view((-1, 1, self.rank))
        
        rot_mat = self.rel_diag(r)
        head3 = u+rvh
        rot_q = givens_rotations(rot_mat, head3).view((-1, 1, self.rank))
        ref_q = givens_reflection(ref_mat, head3).view((-1, 1, self.rank))
        
        cands = torch.cat([ref_q, rot_q], dim=1)
        att_vec = self.att3(r).view((-1, 1, self.rank))
        att_weights = torch.sum(att_vec * cands * self.scale.to(cands.device), dim=-1, keepdim=True)
        att_weights = self.act(att_weights)
        res3 = torch.sum(att_weights * cands, dim=1).view((-1, 1, self.rank))
        
        cands = torch.cat([res1,res3], dim=1)
        context_vec = self.context_vec(r).view((-1, 1, self.rank))
        att_weights = torch.sum(context_vec * cands * self.scale.to(cands.device), dim=-1, keepdim=True)
        att_weights = self.act(att_weights)
        att_q = torch.sum(att_weights * cands, dim=1)
        
        c = F.softplus(self.c[r])
        lhs = expmap0(att_q, c)
        rel = self.rel(r)
        rel_c = expmap0(rel, c)
        res = project(mobius_add(lhs, rel_c, c), c)
        
        return (res, c), self.bh(queries[:, 0])

    def get_queries_lp(self, queries, r):
        
        u = self.entity(queries[:, 0])
        rvh = self.rel(r)
        
        rot_mat = self.rel_diag(r)
        ref_mat = self.rel_diag1(r)
        rot_mat1 = self.rel_diag2(r)
        ref_mat1 = self.rel_diag3(r)

        c1 = F.softplus(self.c1[r])
        head1 = expmap0(u, c1)
        rel1 = expmap0(rvh, c1)
        head1 = project(mobius_add(head1, rel1, c1), c1)
        head1 = logmap0(head1,c1)
        rot_1 = givens_rotations(rot_mat1, head1).view((-1, 1, self.rank))
        ref_1 = givens_reflection(ref_mat1, head1).view((-1, 1, self.rank))
        
        cands = torch.cat([ref_1, rot_1], dim=1)
        att_vec1 = self.att1(r).view((-1, 1, self.rank))
        att_weights = torch.sum(att_vec1 * cands * self.scale.to(cands.device), dim=-1, keepdim=True)
        att_weights = self.act(att_weights)
        att_1 = torch.sum(att_weights * cands, dim=1)
        
        res1=att_1.view((-1, 1, self.rank))
        
        rot_mat = self.rel_diag(r)
        head3 = u+rvh
        rot_q = givens_rotations(rot_mat, head3).view((-1, 1, self.rank))
        ref_q = givens_reflection(ref_mat, head3).view((-1, 1, self.rank))
        
        cands = torch.cat([ref_q, rot_q], dim=1)
        att_vec = self.att3(r).view((-1, 1, self.rank))
        att_weights = torch.sum(att_vec * cands * self.scale.to(cands.device), dim=-1, keepdim=True)
        att_weights = self.act(att_weights)
        res3 = torch.sum(att_weights * cands, dim=1).view((-1, 1, self.rank))
        
        cands = torch.cat([res1,res3], dim=1)
        context_vec = self.context_vec(r).view((-1, 1, self.rank))
        att_weights = torch.sum(context_vec * cands * self.scale.to(cands.device), dim=-1, keepdim=True)
        att_weights = self.act(att_weights)
        att_q = torch.sum(att_weights * cands, dim=1)
        
        c = F.softplus(self.c[r])
        lhs = expmap0(att_q, c)
        rel = self.rel(r)
        rel_c = expmap0(rel, c)
        res = project(mobius_add(lhs, rel_c, c), c)
        
        return (res, c), self.bh(queries[:, 0])
    
class customAttH(BaseH): # AttH on curvature c

    def __init__(self, args):
        super(customAttH, self).__init__(args)
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
        rvh = self.rel(r)
        c = F.softplus(self.c[r])

        u_e = logmap0(u, c)
        rot_mat = self.rel_diag(r)
        ref_mat = self.rel_diag1(r)
        rot_1 = givens_rotations(rot_mat, u_e).view((-1, 1, self.rank))
        ref_1 = givens_reflection(ref_mat, u_e).view((-1, 1, self.rank))
        
        cands = torch.cat([ref_1, rot_1], dim=1)
        context_vec = self.context_vec(r).view((-1, 1, self.rank))
        att_weights = torch.sum(context_vec * cands * self.scale, dim=-1, keepdim=True)
        att_weights = self.act(att_weights)
        att_q = torch.sum(att_weights * cands, dim=1)
        
        u_m = expmap0(att_q, c)
        rel1 = expmap0(rvh, c)
        res = project(mobius_add(u_m, rel1, c), c)
        
        return (res, c), self.bh(queries[:, 0])
    
    def get_queries_lp(self, queries, r):
        
        u = self.entity(queries[:, 0])
        rvh = self.rel(r)
        c = F.softplus(self.c[r])

        u_e = logmap0(u, c)
        rot_mat = self.rel_diag(r)
        ref_mat = self.rel_diag1(r)
        rot_1 = givens_rotations(rot_mat, u_e).view((-1, 1, self.rank))
        ref_1 = givens_reflection(ref_mat, u_e).view((-1, 1, self.rank))
        
        cands = torch.cat([ref_1, rot_1], dim=1)
        context_vec = self.context_vec(r).view((-1, 1, self.rank))
        att_weights = torch.sum(context_vec * cands * self.scale, dim=-1, keepdim=True)
        att_weights = self.act(att_weights)
        att_q = torch.sum(att_weights * cands, dim=1)
        
        u_m = expmap0(att_q, c)
        rel1 = expmap0(rvh, c)
        res = project(mobius_add(u_m, rel1, c), c)
        
        return (res, c), self.bh(queries[:, 0])
    
    # def get_rhs(self, queries, eval_mode):
    #     if eval_mode:
    #         c = F.softplus(self.c[queries[:, 1]])
    #         return expmap0(self.entity.weight, c), self.bt.weight
    #     else:
    #         c = F.softplus(self.c[queries[:, 1]])
    #         return expmap0(self.entity(queries[:, 2]), c), self.bt(queries[:, 2])