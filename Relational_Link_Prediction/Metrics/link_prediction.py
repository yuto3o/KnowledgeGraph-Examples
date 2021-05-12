# -*- coding: utf-8 -*-
import torch


def evaluate_link_prediction(y_pred: torch.FloatTensor):
    # mean rank
    mr = torch.mean(y_pred)

    # mean reciprocal rank
    mrr = torch.mean(1. / y_pred)

    # hits@1
    hits_at_1 = torch.mean(torch.le(y_pred, 1).float())

    # hits@3
    hits_at_3 = torch.mean(torch.le(y_pred, 3).float())

    # hits@10
    hits_at_10 = torch.mean(torch.le(y_pred, 10).float())

    return dict(mr=mr, mrr=mrr, hits_at_1=hits_at_1, hits_at_3=hits_at_3, hits_at_10=hits_at_10)
