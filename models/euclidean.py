import numpy as np
import torch
from torch import nn

from models.base import KGModel
from utils.euclidean import euc_sqdistance, givens_rotations, givens_reflection

EUC_MODELS = ["TransE", "CP", "MurE"]


class BaseE(KGModel):

    def __init__(self, args):
        super(BaseE, self).__init__(args.sizes, args.rank, args.dropout, args.gamma, args.dtype, args.bias,
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
            
        else:
            self.entity.weight.data = self.init_size * torch.randn((self.sizes[0], self.rank), dtype=self.data_type)       
            self.rel.weight.data = self.init_size * torch.randn((self.sizes[1], self.rank), dtype=self.data_type)
            self.rel_diag = nn.Embedding(self.sizes[1], self.rank)
            self.rel_diag.weight.data = 2 * torch.rand((self.sizes[1], self.rank), dtype=self.data_type) - 1.0

    def get_rhs(self, queries, eval_mode):
        if eval_mode:
            return self.entity.weight, self.bt.weight
        else:
            return self.entity(queries[:, 2]), self.bt(queries[:, 2])

    def similarity_score(self, lhs_e, rhs_e, eval_mode):
        if self.sim == "dot":
            if eval_mode:
                score = lhs_e @ rhs_e.transpose(0, 1)
            else:
                score = torch.sum(lhs_e * rhs_e, dim=-1, keepdim=True)
        else:
            score = - euc_sqdistance(lhs_e, rhs_e, eval_mode)
        return score


class TransE(BaseE):

    def __init__(self, args):
        super(TransE, self).__init__(args)
        self.sim = "dist"

    def get_queries(self, queries):
        head_e = self.entity(queries[:, 0])
        rel_e = self.rel(queries[:, 1])
        lhs_e = head_e + rel_e
        lhs_biases = self.bh(queries[:, 0])
        return lhs_e, lhs_biases
    
    def get_queries_lp(self, queries, r):
        head_e = self.entity(queries[:, 0])
        rel_e = self.rel(r)
        lhs_e = head_e + rel_e
        lhs_biases = self.bh(queries[:, 0])
        return lhs_e, lhs_biases


class CP(BaseE):

    def __init__(self, args):
        super(CP, self).__init__(args)
        self.sim = "dot"

    def get_queries(self, queries: torch.Tensor):
        return self.entity(queries[:, 0]) * self.rel(queries[:, 1]), self.bh(queries[:, 0])
    
    def get_queries_lp(self, queries: torch.Tensor, r):
        return self.entity(queries[:, 0]) * self.rel(r), self.bh(queries[:, 0])


class MurE(BaseE):

    def __init__(self, args):
        super(MurE, self).__init__(args)
        self.sim = "dist"

    def get_queries(self, queries: torch.Tensor):
        lhs_e = self.rel_diag(queries[:, 1]) * self.entity(queries[:, 0]) + self.rel(queries[:, 1])
        lhs_biases = self.bh(queries[:, 0])
        return lhs_e, lhs_biases
    
    def get_queries_lp(self, queries: torch.Tensor, r):
        lhs_e = self.rel_diag(r) * self.entity(queries[:, 0]) + self.rel(r)
        lhs_biases = self.bh(queries[:, 0])
        return lhs_e, lhs_biases

class RotE(BaseE):
    """Euclidean 2x2 Givens rotations"""

    def __init__(self, args):
        super(RotE, self).__init__(args)
        # self.rel_diag = nn.Embedding(self.sizes[1], self.rank)
        # self.rel_diag.weight.data = 2 * torch.rand((self.sizes[1], self.rank), dtype=self.data_type) - 1.0
        self.sim = "dist"

    def get_queries(self, queries: torch.Tensor):
        """Compute embedding and biases of queries."""
        lhs_e = givens_rotations(self.rel_diag(queries[:, 1]), self.entity(queries[:, 0])) + self.rel(queries[:, 1])
        lhs_biases = self.bh(queries[:, 0])
        return lhs_e, lhs_biases
    
    def get_queries_lp(self, queries: torch.Tensor, r):
        """Compute embedding and biases of queries."""
        lhs_e = givens_rotations(self.rel_diag(r), self.entity(queries[:, 0])) + self.rel(r)
        lhs_biases = self.bh(queries[:, 0])
        return lhs_e, lhs_biases

class RefE(BaseE):
    """Euclidean 2x2 Givens reflections"""

    def __init__(self, args):
        super(RefE, self).__init__(args)
        # self.rel_diag = nn.Embedding(self.sizes[1], self.rank)
        # self.rel_diag.weight.data = 2 * torch.rand((self.sizes[1], self.rank), dtype=self.data_type) - 1.0
        self.sim = "dist"

    def get_queries(self, queries):
        """Compute embedding and biases of queries."""
        lhs = givens_reflection(self.rel_diag(queries[:, 1]), self.entity(queries[:, 0]))
        rel = self.rel(queries[:, 1])
        lhs_biases = self.bh(queries[:, 0])
        return lhs + rel, lhs_biases
    
    def get_queries_lp(self, queries, r):
        """Compute embedding and biases of queries."""
        lhs = givens_reflection(self.rel_diag(r), self.entity(queries[:, 0]))
        rel = self.rel(r)
        lhs_biases = self.bh(queries[:, 0])
        return lhs + rel, lhs_biases


class AttE(BaseE):
    """Euclidean attention model combining translations, reflections and rotations"""

    def __init__(self, args):
        super(AttE, self).__init__(args)
        self.sim = "dist"

        # reflection
        self.rel_diag = nn.Embedding(self.sizes[1], self.rank)
        self.rel_diag.weight.data = 2 * torch.rand((self.sizes[1], self.rank), dtype=self.data_type) - 1.0

        # rotation
        self.rel_diag1 = nn.Embedding(self.sizes[1], self.rank)
        self.rel_diag1.weight.data = 2 * torch.rand((self.sizes[1], self.rank), dtype=self.data_type) - 1.0
        
        self.rel_diag2 = nn.Embedding(self.sizes[2], self.rank)
        self.rel_diag2.weight.data = 2 * torch.rand((self.sizes[1], self.rank), dtype=self.data_type) - 1.0

        # attention
        self.context_vec = nn.Embedding(self.sizes[1], self.rank)
        self.act = nn.Softmax(dim=1)
        self.scale = torch.Tensor([1. / np.sqrt(self.rank)]).cuda()
        
        if args.prtd:
            prtd_rel_diag1 = torch.tensor(np.load(args.prtd+'/diag1_relation_embedding.npy'), dtype=self.data_type)
            prot = 2 * torch.randn((self.sizes[1]-prtd_rel_diag1.shape[0], self.rank), dtype=self.data_type) - 1.0
            rel_diag1_embs = torch.vstack((prot, prtd_rel_diag1))
            self.rel_diag1 = nn.Embedding.from_pretrained(rel_diag1_embs, freeze=False)

    def get_reflection_queries(self, queries):
        lhs_ref_e = givens_reflection(
            self.rel_diag(queries[:, 1]), self.entity(queries[:, 0])
        )
        return lhs_ref_e

    def get_rotation_queries(self, queries):
        lhs_rot_e = givens_rotations(
            self.rel_diag1(queries[:, 1]), self.entity(queries[:, 0])
        )
        return lhs_rot_e

    def get_queries(self, queries):
        """Compute embedding and biases of queries."""
        lhs_ref_e = self.get_reflection_queries(queries).view((-1, 1, self.rank))
        lhs_rot_e = self.get_rotation_queries(queries).view((-1, 1, self.rank))

        # self-attention mechanism
        cands = torch.cat([lhs_ref_e, lhs_rot_e], dim=1)
        context_vec = self.context_vec(queries[:, 1]).view((-1, 1, self.rank))
        att_weights = torch.sum(context_vec * cands * self.scale, dim=-1, keepdim=True)
        att_weights = self.act(att_weights)
        lhs_e = torch.sum(att_weights * cands, dim=1) + self.rel(queries[:, 1])
        return lhs_e, self.bh(queries[:, 0])
    
    def get_queries_lp(self, queries, r):
        """Compute embedding and biases of queries."""
        lhs_ref_e = self.get_reflection_queries(queries).view((-1, 1, self.rank))
        lhs_rot_e = self.get_rotation_queries(queries).view((-1, 1, self.rank))

        # self-attention mechanism
        cands = torch.cat([lhs_ref_e, lhs_rot_e], dim=1)
        context_vec = self.context_vec(r).view((-1, 1, self.rank))
        att_weights = torch.sum(context_vec * cands * self.scale, dim=-1, keepdim=True)
        att_weights = self.act(att_weights)
        lhs_e = torch.sum(att_weights * cands, dim=1) + self.rel(r)
        return lhs_e, self.bh(queries[:, 0])
    
class customAttCompE(BaseE):

    def __init__(self, args):
        super(customAttCompE, self).__init__(args)
        self.sim = "dist"

        # reflection
        self.ref = nn.Embedding(self.sizes[1], self.rank)
        self.ref.weight.data = 2 * torch.rand((self.sizes[1], self.rank), dtype=self.data_type) - 1.0

        # rotation
        self.rot = nn.Embedding(self.sizes[1], self.rank)
        self.rot.weight.data = 2 * torch.rand((self.sizes[1], self.rank), dtype=self.data_type) - 1.0
        
        # translation
        self.trs = nn.Embedding(self.sizes[1], self.rank)
        self.trs.weight.data = 2 * torch.rand((self.sizes[1], self.rank), dtype=self.data_type) - 1.0
        
        # scale
        self.scl = nn.Embedding(self.sizes[1], self.rank)
        self.scl.weight.data = 2 * torch.rand((self.sizes[1], self.rank), dtype=self.data_type) - 1.0

        # attention
        self.context_vec = nn.Embedding(self.sizes[1], self.rank)
        self.act = nn.Softmax(dim=1)
        self.scale = torch.Tensor([1. / np.sqrt(self.rank)]).cuda()

    def get_reflection_queries(self, queries):
        lhs_ref_e = givens_reflection(
            self.ref(queries[:, 1]), self.entity(queries[:, 0])
        )
        return lhs_ref_e

    def get_rotation_queries(self, queries):
        lhs_rot_e = givens_rotations(
            self.rot(queries[:, 1]), self.entity(queries[:, 0])
        )
        return lhs_rot_e

    def get_queries(self, queries):
        """Compute embedding and biases of queries."""
        lhs_ref_e = self.get_reflection_queries(queries).view((-1, 1, self.rank))
        lhs_rot_e = self.get_rotation_queries(queries).view((-1, 1, self.rank))

        # self-attention mechanism
        cands = torch.cat([lhs_ref_e, lhs_rot_e], dim=1)
        context_vec = self.context_vec(queries[:, 1]).view((-1, 1, self.rank))
        att_weights = torch.sum(context_vec * cands * self.scale, dim=-1, keepdim=True)
        att_weights = self.act(att_weights)
        lhs_e = torch.sum(att_weights * cands, dim=1) + self.rel(queries[:, 1])
        
        # lhs_e += self.trs(queries[:, 1])
        lhs_e *= self.scl(queries[:, 1])
        
        return lhs_e, self.bh(queries[:, 0])
    
    def get_queries_lp(self, queries, r):
        """Compute embedding and biases of queries."""
        lhs_ref_e = self.get_reflection_queries(queries).view((-1, 1, self.rank))
        lhs_rot_e = self.get_rotation_queries(queries).view((-1, 1, self.rank))

        # self-attention mechanism
        cands = torch.cat([lhs_ref_e, lhs_rot_e], dim=1)
        context_vec = self.context_vec(r).view((-1, 1, self.rank))
        att_weights = torch.sum(context_vec * cands * self.scale, dim=-1, keepdim=True)
        att_weights = self.act(att_weights)
        lhs_e = torch.sum(att_weights * cands, dim=1) + self.rel(queries[:, 1])
        
        # lhs_e += self.trs(queries[:, 1])
        lhs_e *= self.scl(queries[:, 1])
        
        return lhs_e, self.bh(queries[:, 0])