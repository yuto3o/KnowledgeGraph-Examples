# -*- coding: utf-8 -*-
from package.model import KEModel, KEEncoder, KEDecoder

import torch


class RotatEModel(KEModel):

    def __init__(self, num_entities, num_relations, embedding_dim, p):
        super(RotatEModel, self).__init__()

        self.encoder = RotatEEncoder(num_entities, num_relations, embedding_dim)
        self.decoder = RotatEDecoder(p)

    def forward(self, lhs, rel, rhs):
        lhs, rel, rhs = self.encoder(lhs, rel, rhs)
        score = self.decoder(lhs, rel, rhs)
        return score

    def get_regularization_loss(self, lhs, rel, rhs):
        lhs = self.encoder.get_entity_embedding(lhs)
        rel = self.encoder.get_relation_embedding(rel)
        rhs = self.encoder.get_entity_embedding(rhs)

        loss = lhs.pow(2).sum() + \
               rel.pow(2).sum() + \
               rhs.pow(2).sum()

        return loss


class RotatEEncoder(KEEncoder):

    def __init__(self, num_entities, num_relations, embedding_dim):
        super(RotatEEncoder, self).__init__()

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


class RotatEDecoder(KEDecoder):

    def __init__(self, p):
        super(RotatEDecoder, self).__init__()
        self.p = p

    def reset_parameters(self):
        pass

    def forward(self, lhs, rel, rhs):
        # batch_size, num_sample, embedding_dim

        lhs_r, lhs_i = torch.chunk(lhs, 2, dim=2)
        rhs_r, rhs_i = torch.chunk(rhs, 2, dim=2)

        rel_r = torch.cos(rel)
        rel_i = torch.sin(rel)

        if lhs_r.size(1) != 1:  # head
            xr = (rhs_r * rel_r + rhs_i * rel_i) - lhs_r
            xi = (rhs_i * rel_r - rhs_r * rel_i) - lhs_i

        else:  # tail
            xr = (lhs_r * rel_r - lhs_i * rel_i) - rhs_r
            xi = (lhs_r * rel_i + lhs_i * rel_r) - rhs_i

        x = torch.stack([xr, xi], dim=0).norm(p=self.p, dim=0).sum(dim=2)

        return - x
