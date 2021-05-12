# -*- coding: utf-8 -*-
from package.model import KEModel
from package.metrics import evaluate_link_prediction
from tqdm import tqdm

import torch


class Trainer:

    def __init__(self, model: KEModel, optimizer, weight_decay, criterion,
                 train_dataloader, test_dataloader):

        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader

        self.weight_decay = weight_decay
        self.device = next(model.parameters()).device

    def train_reciprocal_epoch(self, epoch_idx, tensorboard=None):
        self.model.train()

        tq = tqdm(self.train_dataloader, desc=f'Epoch {epoch_idx}', total=len(self.train_dataloader))
        loss = 0.

        for step, batch in enumerate(tq):
            self.optimizer.zero_grad()
            batch.to(self.device)

            score = self.model(batch.lhs, batch.rel, batch.rhs)
            loss = self.criterion(score, batch.target, batch.weight)

            # reg loss
            if self.weight_decay != 0.0:
                reg_loss = self.model.get_regularization_loss(batch.lhs, batch.rel, batch.target)
                loss += self.weight_decay * reg_loss / batch.lhs.size(0)

            loss.backward()
            self.optimizer.step()
            tq.set_postfix(loss=float(loss))

            if tensorboard:
                tensorboard.add_scalar('Train/Step/Loss', float(loss), epoch_idx * len(self.train_dataloader) + step)

        if tensorboard:
            tensorboard.add_scalar('Train/Epoch/Loss', float(loss), epoch_idx)

    def train_standard_epoch(self, epoch_idx, tensorboard=None):
        self.model.train()

        tq = tqdm(self.train_dataloader, desc=f'Epoch {epoch_idx}', total=len(self.train_dataloader))
        loss = 0.

        for step, batch in enumerate(tq):
            self.optimizer.zero_grad()
            batch.to(self.device)

            score = self.model(batch.lhs, batch.rel, batch.rhs)
            loss = self.criterion(score, batch.target, batch.weight)

            # reg loss
            if self.weight_decay != 0.0:

                if step % 2 == 0:  # 'head'
                    reg_loss = self.model.get_regularization_loss(batch.target, batch.rel, batch.rhs)
                else:  # 'tail'
                    reg_loss = self.model.get_regularization_loss(batch.lhs, batch.rel, batch.target)

                loss += self.weight_decay * reg_loss / batch.rel.size(0)



            loss.backward()
            self.optimizer.step()
            tq.set_postfix(loss=float(loss))

            if tensorboard:
                tensorboard.add_scalar('Train/Step/Loss', float(loss), epoch_idx * len(self.train_dataloader) + step)

        if tensorboard:
            tensorboard.add_scalar('Train/Epoch/Loss', float(loss), epoch_idx)

    @torch.no_grad()
    def test_reciprocal(self, fwd):
        self.model.eval()

        tq = tqdm(self.test_dataloader, desc=f'Evaluating', total=len(self.test_dataloader))
        rank = []
        for step, batch in enumerate(tq):
            batch.to(self.device)

            score = self.model(batch.lhs, batch.rel, batch.rhs)
            candidate_score = torch.gather(score, 1, batch.target)

            for i, (h, r) in enumerate(zip(batch.lhs, batch.rel)):
                score[i, fwd[int(h), int(r)]] = -float('inf')

            rank.append(torch.sum(score > candidate_score, dim=1).float() + 1)
        rank = torch.cat(rank)
        return {k: float(v) for k, v in evaluate_link_prediction(rank).items()}

    @torch.no_grad()
    def test_standard(self, fwd, fwd_inv):
        self.model.eval()

        tq = tqdm(self.test_dataloader, desc=f'Evaluating', total=len(self.test_dataloader))
        rank = []
        for step, batch in enumerate(tq):
            batch.to(self.device)

            score = self.model(batch.lhs, batch.rel, batch.rhs)
            candidate_score = torch.gather(score, 1, batch.target)

            if step % 2 == 0:  # 'head'
                for i, (t, r) in enumerate(zip(batch.rhs, batch.rel)):
                    score[i, fwd_inv[int(t), int(r)]] = -float('inf')
            else:  # 'tail'
                for i, (h, r) in enumerate(zip(batch.lhs, batch.rel)):
                    score[i, fwd[int(h), int(r)]] = -float('inf')

            rank.append(torch.sum(score > candidate_score, dim=1).float() + 1)
        rank = torch.cat(rank)
        return {k: float(v) for k, v in evaluate_link_prediction(rank).items()}
