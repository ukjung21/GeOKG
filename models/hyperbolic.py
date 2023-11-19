"""Hyperbolic Knowledge Graph embedding models where all parameters are defined in tangent spaces."""
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from models.base import KGModel
from utils.euclidean import givens_rotations, givens_reflection
from utils.hyperbolic import mobius_add, expmap0, project, hyp_distance_multi_c

ATT_MODELS = ["RotH", "RefH", "AttH"]


class BaseH(KGModel):
    """Trainable curvature for each relationship."""

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
        else:
            c_init = torch.ones((1, 1), dtype=self.data_type)
        self.c = nn.Parameter(c_init, requires_grad=True)

    def get_rhs(self, queries, eval_mode):
        """Get embeddings and biases of target entities."""
        if eval_mode:
            return self.entity.weight, self.bt.weight
        else:
            return self.entity(queries[:, 2]), self.bt(queries[:, 2])

    def similarity_score(self, lhs_e, rhs_e, eval_mode):
        """Compute similarity scores or queries against targets in embedding space."""
        lhs_e, c = lhs_e
        return - hyp_distance_multi_c(lhs_e, rhs_e, c, eval_mode) ** 2


class RotH(BaseH):
    """Hyperbolic 2x2 Givens rotations"""

    def get_queries(self, queries):
        """Compute embedding and biases of queries."""
        c = F.softplus(self.c[queries[:, 1]])
        head = expmap0(self.entity(queries[:, 0]), c)
        rel1 = self.rel(queries[:, 1])
        rel1 = expmap0(rel1, c)
        lhs = project(mobius_add(head, rel1, c), c)
        res1 = givens_rotations(self.rel_diag(queries[:, 1]), lhs)
        res2 = mobius_add(res1, rel1, c)
        return (res2, c), self.bh(queries[:, 0])
    
    def get_queries_lp(self, queries, r):
        """Compute embedding and biases of queries."""
        c = F.softplus(self.c[r])
        head = expmap0(self.entity(queries[:, 0]), c)
        rel1 = self.rel(r)
        rel1 = expmap0(rel1, c)
        lhs = project(mobius_add(head, rel1, c), c)
        res1 = givens_rotations(self.rel_diag(r), lhs)
        res2 = mobius_add(res1, rel1, c)
        return (res2, c), self.bh(queries[:, 0])


class RefH(BaseH):
    """Hyperbolic 2x2 Givens reflections"""

    def get_queries(self, queries):
        """Compute embedding and biases of queries."""
        c = F.softplus(self.c[queries[:, 1]])
        rel = self.rel(queries[:, 1])
        rel = expmap0(rel, c)
        lhs = givens_reflection(self.rel_diag(queries[:, 1]), self.entity(queries[:, 0]))
        lhs = expmap0(lhs, c)
        res = project(mobius_add(lhs, rel, c), c)
        return (res, c), self.bh(queries[:, 0])
    
    def get_queries_lp(self, queries, r):
        """Compute embedding and biases of queries."""
        c = F.softplus(self.c[r])
        rel = self.rel(r)
        rel = expmap0(rel, c)
        lhs = givens_reflection(self.rel_diag(r), self.entity(queries[:, 0]))
        lhs = expmap0(lhs, c)
        res = project(mobius_add(lhs, rel, c), c)
        return (res, c), self.bh(queries[:, 0])


class AttH(BaseH):
    """Hyperbolic attention model combining translations, reflections and rotations"""

    def __init__(self, args):
        super(AttH, self).__init__(args)
        # self.rel_diag = nn.Embedding(self.sizes[1], 2 * self.rank)
        # self.rel_diag.weight.data = 2 * torch.rand((self.sizes[1], 2 * self.rank), dtype=self.data_type) - 1.0
        self.context_vec = nn.Embedding(self.sizes[1], self.rank)
        self.context_vec.weight.data = self.init_size * torch.randn((self.sizes[1], self.rank), dtype=self.data_type)
        self.act = nn.Softmax(dim=1)
        if args.dtype == "double":
            self.scale = torch.Tensor([1. / np.sqrt(self.rank)]).double().cuda()
        else:
            self.scale = torch.Tensor([1. / np.sqrt(self.rank)]).cuda()

    def get_queries(self, queries):
        """Compute embedding and biases of queries."""
        c = F.softplus(self.c[queries[:, 1]])
        head = self.entity(queries[:, 0])
        rot_mat = self.rel_diag(queries[:, 1])
        ref_mat = self.rel_diag1(queries[:, 1])
        rot_q = givens_rotations(rot_mat, head).view((-1, 1, self.rank))
        ref_q = givens_reflection(ref_mat, head).view((-1, 1, self.rank))
        
        cands = torch.cat([ref_q, rot_q], dim=1)
        context_vec = self.context_vec(queries[:, 1]).view((-1, 1, self.rank))
        att_weights = torch.sum(context_vec * cands * self.scale, dim=-1, keepdim=True)
        att_weights = self.act(att_weights)
        att_q = torch.sum(att_weights * cands, dim=1)
        
        lhs = expmap0(att_q, c)
        rel = self.rel(queries[:, 1])
        rel = expmap0(rel, c)
        res = project(mobius_add(lhs, rel, c), c)
        
        return (res, c), self.bh(queries[:, 0])
    
    def get_queries_lp(self, queries, r):
        """Compute embedding and biases of queries."""
        c = F.softplus(self.c[r])
        head = self.entity(queries[:, 0])
        rot_mat = self.rel_diag(r)
        ref_mat = self.rel_diag1(r)
        rot_q = givens_rotations(rot_mat, head).view((-1, 1, self.rank))
        ref_q = givens_reflection(ref_mat, head).view((-1, 1, self.rank))
        
        cands = torch.cat([ref_q, rot_q], dim=1)
        context_vec = self.context_vec(r).view((-1, 1, self.rank))
        att_weights = torch.sum(context_vec * cands * self.scale, dim=-1, keepdim=True)
        att_weights = self.act(att_weights)
        att_q = torch.sum(att_weights * cands, dim=1)
        
        lhs = expmap0(att_q, c)
        rel = self.rel(queries[:, 1])
        rel = expmap0(rel, c)
        res = project(mobius_add(lhs, rel, c), c)
        
        return (res, c), self.bh(queries[:, 0])