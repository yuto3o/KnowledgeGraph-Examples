# -*- coding: utf-8 -*-
from package.dataset.fb15k import FB15k
from package.dataset.fb15k237 import FB15k237
from package.dataset.wn18 import WN18
from package.dataset.wn18rr import WN18RR

HOME_DIR = '.'


def benchmark(name: str):
    name = name.lower()

    if name == 'fb15k':
        bm = FB15k

    elif name == 'fb15k-237':
        bm = FB15k237

    elif name == 'wn18':
        bm = WN18

    elif name == 'wn18rr':
        bm = WN18RR

    else:
        raise NotImplementedError(name)

    return bm(HOME_DIR)
