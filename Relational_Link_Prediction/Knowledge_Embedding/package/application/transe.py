# -*- coding: utf-8 -*-
from package.model import KEModel, KEEncoder, KEDecoder

import torch


class TransEModel(KEModel):

    def __init__(self, num_entities, num_relations, embedding_dim, p):
        super(TransEModel, self).__init__()

        self.encoder = TransEEncoder(num_entities, num_relations, embedding_dim)
        self.decoder = TransEDecoder(p)

    def forward(self, lhs, rel, rhs):
        lhs, rel, rhs = self.encoder(lhs, rel, rhs)
        score = self.decoder(lhs, rel, rhs)
        return score

    def get_regularization_loss(self, lhs, rel, rhs):
        lhs = self.encoder.get_entity_embedding(lhs)
        rel = self.encoder.get_relation_embedding(rel)
        rhs = self.encoder.get_entity_embedding(rhs)

        return lhs.pow(2).sum() + rel.pow(2).sum() + rhs.pow(2).sum()


class TransEEncoder(KEEncoder):

    def __init__(self, num_entities, num_relations, embedding_dim):
        super(TransEEncoder, self).__init__()

        self.E = torch.nn.Embedding(num_entities, embedding_dim)
        self.R = torch.nn.Embedding(num_relations, embedding_dim)

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.E.weight.data)
        torch.nn.init.xavier_uniform_(self.R.weight.data)

    @property
    def entities_embedding(self):
        return self.E.weight

    @property
    def relations_embedding(self):
        return self.R.weight

    def get_entity_embedding(self, token):
        return self.E(token)

    def get_relation_embedding(self, token):
        return self.R(token)

    def forward(self, lhs, rel, rhs):
        return self.E(lhs), self.R(rel), self.E(rhs)


class TransEDecoder(KEDecoder):

    def __init__(self, p):
        super(TransEDecoder, self).__init__()
        self.p = p

    def reset_parameters(self):
        pass

    def forward(self, lhs, rel, rhs):
        # batch_size, num_sample, embedding_dim

        if lhs.size(1) != 1:  # head
            x = lhs + (rel - rhs)
        else:  # tail
            x = (lhs + rel) - rhs

        x = x.norm(self.p, dim=2)

        return - x
