# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F


class NegativeSamplingLoss(torch.nn.Module):

    def __init__(self, gamma, adversarial_sampling=False, adversarial_temperature=1.):
        super(NegativeSamplingLoss, self).__init__()
        self.gamma = gamma
        self.adversarial_sampling = adversarial_sampling
        self.adversarial_temperature = adversarial_temperature

    def forward(self, input, target, weights=None):
        b = input.size(0)
        mask = torch.gt(target, 0.5)
        positive_score = input[mask].view(b, -1)  # positive sample only at first position in NS mode
        negative_score = input[~mask].view(b, -1)

        negative_score = F.logsigmoid(-self.gamma - negative_score)

        if self.adversarial_sampling:
            negative_score *= F.softmax(negative_score * self.adversarial_temperature, dim=1).detach()
            negative_score = negative_score.sum(dim=1)
        else:
            negative_score = negative_score.mean(dim=1)

        positive_score = F.logsigmoid(self.gamma + positive_score)

        if weights is None:
            positive_score = positive_score.mean()
            negative_score = negative_score.mean()
        else:
            positive_score = (weights * positive_score).sum() / weights.sum()
            negative_score = (weights * negative_score).sum() / weights.sum()

        return - (positive_score + negative_score) / 2.


class LabelSmoothingCELoss(torch.nn.Module):

    def __init__(self, epsilon=0.):
        super(LabelSmoothingCELoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, input, target, weights=None):
        n = input.size(-1)
        target = target[:, 0]
        log_x = F.log_softmax(input, dim=-1)

        if weights is None:
            loss = -log_x.sum(dim=-1).mean()
            nll = F.nll_loss(log_x, target)
        else:
            loss = -(log_x.sum(dim=-1) * weights).sum() / weights.sum()
            nll = (F.nll_loss(log_x, target, reduction='none') * weights).sum() / weights.sum()

        x = self.epsilon * loss / n + (1. - self.epsilon) * nll
        return x


class LabelSmoothingBCELoss(torch.nn.Module):

    def __init__(self, gamma=0., epsilon=0.):
        super(LabelSmoothingBCELoss, self).__init__()
        self.loss = torch.nn.BCEWithLogitsLoss(reduction='none')
        self.gamma = gamma
        self.epsilon = epsilon

    def forward(self, input, target, weights=None):
        n = input.size(-1)

        target = (1. - self.epsilon) * target + self.epsilon / n
        x = self.loss(input + self.gamma, target)
        if weights is None:
            x = x.mean()
        else:
            x = (x * weights).sum() / weights.sum()
        return x
