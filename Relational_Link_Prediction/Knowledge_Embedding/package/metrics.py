# -*- coding: utf-8 -*-
import torch


def evaluate_link_prediction(input: torch.Tensor):
    # mean rank
    mr = torch.mean(input)

    # mean reciprocal rank
    mrr = torch.mean(1. / input)

    # hits@1
    hits_at_1 = torch.mean(torch.le(input, 1).float())

    # hits@3
    hits_at_3 = torch.mean(torch.le(input, 3).float())

    # hits@10
    hits_at_10 = torch.mean(torch.le(input, 10).float())

    return dict(mr=mr, mrr=mrr, hits_at_1=hits_at_1, hits_at_3=hits_at_3, hits_at_10=hits_at_10)
