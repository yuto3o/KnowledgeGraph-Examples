# -*- coding: utf-8 -*-
from collections import defaultdict

import torch


def ndim(x):
    def _func(x, n):
        if not isinstance(x, str) and hasattr(x, '__len__'):
            n = _func(x[0], n + 1)
        return n

    return _func(x, 0)


def coordinate_mask(n, candidate):
    mask = torch.zeros(n)
    mask[candidate] = 1.
    return mask


def single_hop(triplets, inverse_return=False):
    if inverse_return:
        fwd = defaultdict(list)
        fwd_inversed = defaultdict(list)
        for _1, _2, _3 in triplets:
            fwd[_1, _2].append(_3)
            fwd_inversed[_3, _2].append(_1)
        return dict(fwd), dict(fwd_inversed)
    else:
        fwd = defaultdict(list)
        for _1, _2, _3 in triplets:
            fwd[_1, _2].append(_3)
        return dict(fwd)
