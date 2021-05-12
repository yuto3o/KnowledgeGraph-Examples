# -*- coding: utf-8 -*-
from package.utils import coordinate_mask, single_hop
from torch import LongTensor, tensor, sqrt, device, Tensor
from torch.utils.data import Dataset as _Dataset, DataLoader as _DataLoader

import numpy as np


class Data(dict):

    def __init__(self, *args, **kwargs):
        super(Data, self).__init__(*args, **kwargs)

    def __getattr__(self, item):
        try:
            return self[item]
        except AttributeError:
            return None

    def __setattr__(self, key, value):
        self[key] = value

    def to(self, x: device):

        for k, v in self.items():
            if isinstance(v, Tensor):
                self[k] = v.to(x)


class Dataset(_Dataset):
    SAMPLING_MODE_NS = 'ns'
    SAMPLING_MODE_1VN = '1vn'
    SAMPLING_MODE_KVN = 'kvn'

    CORRUPT_MODE_HEAD = 'head'
    CORRUPT_MODE_TAIL = 'tail'

    def __init__(self, triplets, num_entities: int,
                 sampling_mode: str, corrupt_mode: str, num_negative_samples: int = 256):
        """
        Dataset Module

        Parameters
        ----------
        triplets: [(lhs, rel, rhs), ...]
        num_entities: int
            number of the unique entities in triplets
        sampling_mode: str
            sampling mode, SAMPLING_MODE_NS, SAMPLING_MODE_1VN or SAMPLING_MODE_KVN
        corrupt_mode: str
            corrupt mode, CORRUPT_MODE_HEAD or CORRUPT_MODE_TAIL
        num_negative_samples: int, Default: 256
            number of the negative samples per positive sample
        """

        assert sampling_mode in (Dataset.SAMPLING_MODE_NS, Dataset.SAMPLING_MODE_1VN, Dataset.SAMPLING_MODE_KVN)
        assert corrupt_mode in (Dataset.CORRUPT_MODE_HEAD, Dataset.CORRUPT_MODE_TAIL)

        self.triplets = triplets
        self.num_entities = num_entities

        self.corrupt_mode = corrupt_mode
        self.sampling_mode = sampling_mode
        self.num_negative_samples = num_negative_samples

        self.fwd, self.fwd_inv = single_hop(self.triplets, inverse_return=True)

        if self.sampling_mode == Dataset.SAMPLING_MODE_KVN:
            if self.corrupt_mode == Dataset.CORRUPT_MODE_TAIL:
                self.triplets = list(self.fwd.keys())
            else:  # HEAD
                self.triplets = list(self.fwd_inv.keys())

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, item):

        e = np.arange(self.num_entities)

        if self.sampling_mode == Dataset.SAMPLING_MODE_KVN:

            if self.corrupt_mode == Dataset.CORRUPT_MODE_TAIL:
                lhs, rel = self.triplets[item]
                target = coordinate_mask(self.num_entities, self.fwd[lhs, rel])
                return LongTensor([lhs]), LongTensor([rel]), LongTensor(e), target, tensor(1.)
            else:  # HEAD
                rhs, rel = self.triplets[item]
                target = coordinate_mask(self.num_entities, self.fwd_inv[rhs, rel])
                return LongTensor(e), LongTensor([rel]), LongTensor([rhs]), target, tensor(1.)

        elif self.sampling_mode == Dataset.SAMPLING_MODE_1VN:
            lhs, rel, rhs = self.triplets[item]

            if self.corrupt_mode == Dataset.CORRUPT_MODE_TAIL:
                return LongTensor([lhs]), LongTensor([rel]), LongTensor(e), LongTensor([rhs]), tensor(1.)
            else:  # HEAD
                return LongTensor(e), LongTensor([rel]), LongTensor([rhs]), LongTensor([lhs]), tensor(1.)

        else:  # NS
            lhs, rel, rhs = self.triplets[item]

            weights = sum(self.fwd[lhs, rel]) + sum(self.fwd_inv[rhs, rel]) + 4
            weights = sqrt(1. / tensor([weights]))

            if self.corrupt_mode == Dataset.CORRUPT_MODE_TAIL:
                n = np.random.choice(np.delete(e, self.fwd[(lhs, rel)]), size=self.num_negative_samples)

                lhs = [lhs]
                rel = [rel]
                rhs = [rhs] + list(n)
                target = coordinate_mask(self.num_entities, [0])

            else:  # HEAD
                n = np.random.choice(np.delete(e, self.fwd_inv[(rhs, rel)]), size=self.num_negative_samples)

                lhs = [lhs] + list(n)
                rel = [rel]
                rhs = [rhs]
                target = coordinate_mask(self.num_entities, [0])

            return LongTensor(lhs), LongTensor(rel), LongTensor(rhs), LongTensor(target), weights


class AlternateDataLoader:

    def __init__(self, datasets, *args, **kwargs):
        """
        Parameters
        ----------
        datasets: [Dataset, ...]
            get batch from the lists in turn
        *args, **kwargs:
            arguments for torch.util.data.DataLoader
        """
        self.dataloaders = [_DataLoader(d, *args, **kwargs) for d in datasets]
        self.length = sum([len(d) for d in self.dataloaders])

    def __iter__(self):
        iterators = [self.iterable(d) for d in self.dataloaders]

        for step in range(self.length):
            yield next(iterators[step % len(iterators)].__iter__())

    def __len__(self):
        return self.length

    @staticmethod
    def iterable(dataloader):
        for lhs, rel, rhs, target, weights in dataloader:
            if dataloader.dataset.sampling_mode == Dataset.SAMPLING_MODE_NS:
                yield Data(lhs=lhs, rel=rel, rhs=rhs, target=target, weight=weights)
            else:
                if dataloader.dataset.corrupt_mode == Dataset.CORRUPT_MODE_HEAD:
                    yield Data(lhs=lhs[0:1], rel=rel, rhs=rhs, target=target, weight=weights)
                else:  # TAIL
                    yield Data(lhs=lhs, rel=rel, rhs=rhs[0:1], target=target, weight=weights)
