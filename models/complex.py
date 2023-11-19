
import torch
from torch import nn

from models.base import KGModel
from utils.euclidean import euc_sqdistance

COMPLEX_MODELS = ["ComplEx", "RotatE"]


class BaseC(KGModel):

    def __init__(self, args):
        super(BaseC, self).__init__(args.sizes, args.rank, args.dropout, args.gamma, args.dtype, args.bias,
                                    args.init_size)
        assert self.rank % 2 == 0, "Complex models require even embedding dimension"
        # self.rank = self.rank // 2
        self.embeddings = nn.ModuleList([
            nn.Embedding(s, 2 * self.rank, sparse=True)
            for s in self.sizes[:2]
        ])
        self.embeddings[0].weight.data = self.init_size * self.embeddings[0].weight.to(self.data_type)
        self.embeddings[1].weight.data = self.init_size * self.embeddings[1].weight.to(self.data_type)

    def get_rhs(self, queries, eval_mode):
        if eval_mode:
            return self.embeddings[0].weight, self.bt.weight
        else:
            return self.embeddings[0](queries[:, 2]), self.bt(queries[:, 2])
        
    def similarity_score(self, lhs_e, rhs_e, eval_mode):
        """Compute similarity scores or queries against targets in embedding space."""
        lhs_e = lhs_e[:, :self.rank], lhs_e[:, self.rank:]
        rhs_e = rhs_e[:, :self.rank], rhs_e[:, self.rank:]
        if eval_mode:
            return lhs_e[0] @ rhs_e[0].transpose(0, 1) + lhs_e[1] @ rhs_e[1].transpose(0, 1)
        else:
            return torch.sum(
                lhs_e[0] * rhs_e[0] + lhs_e[1] * rhs_e[1],
                1, keepdim=True
            )

    # def similarity_score(self, lhs_e, rhs_e, eval_mode):
    #     re_head, im_head = lhs_e[:, :self.rank], lhs_e[:, self.rank:]
    #     re_tail, im_tail = rhs_e[:, :self.rank], rhs_e[:, self.rank:]
    #     if eval_mode:
    #         return lhs_e[0] @ rhs_e[0].transpose(0, 1) + lhs_e[1] @ rhs_e[1].transpose(0, 1)
    #     else:
    #         re_score = re_head - re_tail
    #         im_score = im_head - im_tail
    #         score = torch.stack([re_score, im_score], dim=0)
    #         score = score.norm(dim=0)
    #         return score
        #     return torch.sum(
        #         lhs_e[0] * rhs_e[0] + lhs_e[1] * rhs_e[1],
        #         1, keepdim=True
        #     )
        # h1, h2 = lhs_e[:, :self.rank], lhs_e[:, self.rank:]
        # t1, t2 = rhs_e[:, :self.rank], rhs_e[:, self.rank:]
        # score = - euc_sqdistance(h1, t1, eval_mode) - euc_sqdistance(h2, t2, eval_mode)
        # return score

    def get_complex_embeddings(self, queries):
        head_e = self.embeddings[0](queries[:, 0])
        rel_e = self.embeddings[1](queries[:, 1])
        rhs_e = self.embeddings[0](queries[:, 2])
        head_e = head_e[:, :self.rank], head_e[:, self.rank:]
        rel_e = rel_e[:, :self.rank], rel_e[:, self.rank:]
        rhs_e = rhs_e[:, :self.rank], rhs_e[:, self.rank:]
        return head_e, rel_e, rhs_e

    def get_factors(self, queries):
        head_e, rel_e, rhs_e = self.get_complex_embeddings(queries)
        head_f = torch.sqrt(head_e[0] ** 2 + head_e[1] ** 2)
        rel_f = torch.sqrt(rel_e[0] ** 2 + rel_e[1] ** 2)
        rhs_f = torch.sqrt(rhs_e[0] ** 2 + rhs_e[1] ** 2)
        return head_f, rel_f, rhs_f


class ComplEx(BaseC):

    def get_queries(self, queries):
        head_e, rel_e, _ = self.get_complex_embeddings(queries)
        lhs_e = torch.cat([
            head_e[0] * rel_e[0] - head_e[1] * rel_e[1],
            head_e[0] * rel_e[1] + head_e[1] * rel_e[0]
        ], 1)
        return lhs_e, self.bh(queries[:, 0])


class RotatE(BaseC):

    def get_queries(self, queries):
        head_e, rel_e, _ = self.get_complex_embeddings(queries)
        rel_norm = torch.sqrt(rel_e[0] ** 2 + rel_e[1] ** 2)
        cos = rel_e[0] / rel_norm # re_relation
        sin = rel_e[1] / rel_norm # im_relation
        lhs_e = torch.cat([
            head_e[0] * cos - head_e[1] * sin, # re_head
            head_e[0] * sin + head_e[1] * cos # im_head
        ], 1)
        return lhs_e, self.bh(queries[:, 0])
    
    def get_queries_lp(self, queries, r):
        head_e, _, _ = self.get_complex_embeddings(queries)
        rel_e = self.embeddings[1](r)
        rel_e = rel_e[:, :self.rank], rel_e[:, self.rank:]
        rel_norm = torch.sqrt(rel_e[0] ** 2 + rel_e[1] ** 2)
        cos = rel_e[0] / rel_norm
        sin = rel_e[1] / rel_norm
        lhs_e = torch.cat([
            head_e[0] * cos - head_e[1] * sin,
            head_e[0] * sin + head_e[1] * cos
        ], 1)
        return lhs_e, self.bh(queries[:, 0])

class CompoundE(BaseC):
    def get_queries(self, queries):
        head_e, rel_e, _ = self.get_complex_embeddings(queries)
        rel_norm = torch.sqrt(rel_e[0] ** 2 + rel_e[1] ** 2)
        cos = rel_e[0] / rel_norm # re_rotation
        sin = rel_e[1] / rel_norm # im_rotation
        lhs_e = torch.cat([
            head_e[0] * cos - head_e[1] * sin, # re_head
            head_e[0] * sin + head_e[1] * cos # im_head
        ], 1)
        return lhs_e, self.bh(queries[:, 0])
    
    def get_queries_lp(self, queries, r):
        head_e, _, _ = self.get_complex_embeddings(queries)
        rel_e = self.embeddings[1](r)
        rel_e = rel_e[:, :self.rank], rel_e[:, self.rank:]
        rel_norm = torch.sqrt(rel_e[0] ** 2 + rel_e[1] ** 2)
        cos = rel_e[0] / rel_norm
        sin = rel_e[1] / rel_norm
        lhs_e = torch.cat([
            head_e[0] * cos - head_e[1] * sin,
            head_e[0] * sin + head_e[1] * cos
        ], 1)
        return lhs_e, self.bh(queries[:, 0])