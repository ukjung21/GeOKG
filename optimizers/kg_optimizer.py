"""Knowledge Graph embedding model optimizer."""
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from torch import nn
import math

class KGOptimizer(object):

    def __init__(
            self, args, model, regularizer, optimizer, batch_size, neg_sample_size, double_neg, verbose=True):
        self.args = args
        self.model = model
        self.regularizer = regularizer
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.verbose = verbose
        self.double_neg = double_neg
        self.loss_fn = nn.CrossEntropyLoss(reduction='mean')
        self.neg_sample_size = neg_sample_size
        self.n_entities = model.sizes[0]

    def reduce_lr(self, factor=0.8):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= factor

    def get_neg_samples(self, input_batch):
        negative_batch = input_batch.repeat(self.neg_sample_size, 1)
        batch_size = input_batch.shape[0]
        negsamples = torch.Tensor(np.random.randint(
            self.n_entities,
            size=batch_size * self.neg_sample_size)
        ).to(input_batch.dtype)
        negative_batch[:, 2] = negsamples
        if self.double_neg:
            negsamples = torch.Tensor(np.random.randint(
                self.n_entities,
                size=batch_size * self.neg_sample_size)
            ).to(input_batch.dtype)
            negative_batch[:, 0] = negsamples
        
        # negative_batch = input_batch.repeat(1, 1)
        # batch_size = input_batch.shape[0]
        # negsamples = torch.Tensor(np.random.randint(
        #     self.n_entities,
        #     size=batch_size)
        # ).to(input_batch.dtype)
        # negative_batch[:, 2] = negsamples
        # if self.double_neg:
        #     negsamples = torch.Tensor(np.random.randint(
        #         self.n_entities,
        #         size=batch_size)
        #     ).to(input_batch.dtype)
        #     negative_batch[:, 0] = negsamples
        return negative_batch

    def neg_sampling_loss(self, input_batch):
        positive_score, factors = self.model(input_batch)
        positive_score = F.logsigmoid(-positive_score)

        neg_samples = self.get_neg_samples(input_batch)
        negative_score, _ = self.model(neg_samples)
        negative_score = F.logsigmoid(negative_score)
        loss = - torch.cat([positive_score, negative_score], dim=0).mean()
        
#         m=self.args.margin
        # u1=self.args.u1
        # u2=self.args.u2
#         lam=self.args.lam
        
        # self.act = nn.Sigmoid()
        # self.criterion = nn.Softplus()
        # self.relu = nn.ReLU()
        
        # p_score, factors = self.model(input_batch)
        # neg_samples = self.get_neg_samples(input_batch)
        # n_score, _ = self.model(neg_samples)        
        
        # zeros_neg = torch.zeros_like(n_score)
        # zeros_pos = torch.zeros_like(p_score)
        
        # SSLoss
        # loss = torch.max(p_score - u1, zeros_pos).mean() + lam * torch.max(u2 - n_score, zeros_neg).mean()
        
        # logistic SSLoss
        # positive_score = F.logsigmoid(u1-p_score)
        # negative_score = F.logsigmoid(n_score-u2)
        # loss = - torch.cat([positive_score, negative_score], dim=0).mean()
        
        # softplusLoss
        # positive_score = self.criterion(-p_score)
        # negative_score = self.criterion(n_score)
        # loss = torch.cat([positive_score, negative_score], dim=0).mean()
        
        # softplus SSLoss
        # log_p = -torch.log(self.act(p_score))
        # log_n = -torch.log(self.act(n_score))
        # p_loss = self.relu(u1 - log_p)
        # n_loss = self.relu(log_n - u2)
        # loss = torch.cat([p_loss, n_loss], dim=0).mean()
        
        # total = torch.cat((p_score, n_score), 1)
        # t_soft = self.act(total)
        # p_s, n_s = torch.split(t_soft, [1, 10], dim=1)
        # log_p = -torch.log(p_s)
        # log_n = -torch.log(n_s)
        # p_loss = self.relu(log_p+math.log(u1)).mean()
        # n_loss = self.relu(-math.log(u2)-log_n).mean()
        # loss = p_loss + lam * n_loss

        return loss, factors

    def no_neg_sampling_loss(self, input_batch):
        predictions, factors = self.model(input_batch, eval_mode=True)
        truth = input_batch[:, 2]
        log_prob = F.logsigmoid(-predictions)
        idx = torch.arange(0, truth.shape[0], dtype=truth.dtype)
        pos_scores = F.logsigmoid(predictions[idx, truth]) - F.logsigmoid(-predictions[idx, truth])
        log_prob[idx, truth] += pos_scores
        loss = - log_prob.mean()
        loss += self.regularizer.forward(factors)
        return loss, factors

    def calculate_loss(self, input_batch):
        if self.neg_sample_size > 0:
            loss, factors = self.neg_sampling_loss(input_batch)
        else:
            predictions, factors = self.model(input_batch, eval_mode=True)
            truth = input_batch[:, 2]
            loss = self.loss_fn(predictions, truth)
        loss += self.regularizer.forward(factors)
        return loss

    def calculate_valid_loss(self, examples):
        b_begin = 0
        loss = 0.0
        counter = 0
        with torch.no_grad():
            while b_begin < examples.shape[0]:
                input_batch = examples[
                              b_begin:b_begin + self.batch_size
                              ].cuda()
                b_begin += self.batch_size
                loss += self.calculate_loss(input_batch)
                counter += 1
        loss /= counter
        return loss

    def epoch(self, examples):
        actual_examples = examples[torch.randperm(examples.shape[0]), :]
        with tqdm.tqdm(total=examples.shape[0], unit='ex', disable=not self.verbose) as bar:
            bar.set_description(f'train loss')
            b_begin = 0
            total_loss = 0.0
            counter = 0
            while b_begin < examples.shape[0]:
                input_batch = actual_examples[
                              b_begin:b_begin + self.batch_size
                              ].cuda()

                l = self.calculate_loss(input_batch)
                self.optimizer.zero_grad()
                l.backward()
                self.optimizer.step()

                b_begin += self.batch_size
                total_loss += l
                counter += 1
                bar.update(input_batch.shape[0])
                bar.set_postfix(loss=f'{l.item():.4f}')
        total_loss /= counter
        return total_loss
