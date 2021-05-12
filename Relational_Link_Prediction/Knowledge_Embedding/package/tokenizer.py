# -*- coding: utf-8 -*-
from package.utils import ndim

from os.path import join


class Tokenizer:

    def __init__(self, entity2id, relation2id):
        self.entity2id = entity2id
        self.relation2id = relation2id

        self.id2entity = {v: k for k, v in entity2id.items()}
        self.id2relation = {v: k for k, v in relation2id.items()}

    @property
    def num_entities(self):
        return len(self.entity2id)

    @property
    def num_relations(self):
        return len(self.relation2id)

    def tokenize(self, xs):

        if ndim(xs) > 1:
            return [self.tokenize(x) for x in xs]
        else:
            lhs, rel, rhs = xs
            return self.entity2id[lhs], self.relation2id[rel], self.entity2id[rhs]

    def tokenize_inv(self, xs):

        if ndim(xs) > 1:
            return [self.tokenize_inv(x) for x in xs]
        else:
            lhs, rel, rhs = xs
            return self.id2entity[lhs], self.id2relation[rel], self.id2entity[rhs]

    def save(self, path: str, entity_file='entity.txt', relation_file='relation.txt'):

        with open(join(path, entity_file), 'w', encoding='utf-8') as f:
            for _ in range(len(self.id2entity)):
                f.write(f"{self.id2entity[_]}\n")

        with open(join(path, relation_file), 'w', encoding='utf-8') as f:
            for _ in range(len(self.id2relation)):
                f.write(f"{self.id2relation[_]}\n")

    @staticmethod
    def load(path: str, entity_file='entity.txt', relation_file='relation.txt'):

        entity2id = {}
        with open(join(path, entity_file), 'r', encoding='utf-8') as f:

            while True:
                vocab = f.readline().strip()
                if not vocab:
                    break
                entity2id[vocab] = len(entity2id)

        relation2id = {}
        with open(join(path, relation_file), 'r', encoding='utf-8') as f:
            while True:
                vocab = f.readline().strip()
                if not vocab:
                    break
                relation2id[vocab] = len(relation2id)

        return Tokenizer(entity2id, relation2id)
