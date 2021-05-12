# -*- coding: utf-8 -*-
from package.model import KEModel, KEEncoder, KEDecoder

import torch


class TuckERModel(KEModel):

    def __init__(self, num_entities, num_relations, embedding_dim_entity, embedding_dim_relation,
                 input_dropout_rate, hidden0_dropout_rate, hidden1_dropout_rate):
        super(TuckERModel, self).__init__()

        self.encoder = TuckEREncoder(num_entities, num_relations, embedding_dim_entity, embedding_dim_relation)
        self.decoder = TuckERDecoder(embedding_dim_entity, embedding_dim_relation,
                                     input_dropout_rate, hidden0_dropout_rate, hidden1_dropout_rate)

    def forward(self, lhs, rel, rhs):
        lhs, rel, rhs = self.encoder(lhs, rel, rhs)
        score = self.decoder(lhs, rel, rhs)
        return score

    def get_regularization_loss(self, lhs, rel, rhs):
        lhs = self.encoder.get_entity_embedding(lhs)
        rel = self.encoder.get_relation_embedding(rel)
        rhs = self.encoder.get_entity_embedding(rhs)

        loss = lhs.pow(3).sum() + \
               rel.pow(3).sum() + \
               rhs.pow(3).sum()

        return loss


class TuckEREncoder(KEEncoder):

    def __init__(self, num_entities, num_relations, embedding_dim_entity, embedding_dim_relation):
        super(TuckEREncoder, self).__init__()

        self.E = torch.nn.Embedding(num_entities, embedding_dim_entity)
        self.R = torch.nn.Embedding(num_relations, embedding_dim_relation)

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


class TuckERDecoder(KEDecoder):

    def __init__(self, embedding_dim_entity, embedding_dim_relation,
                 input_dropout_rate, hidden0_dropout_rate, hidden1_dropout_rate):
        super(TuckERDecoder, self).__init__()

        w = torch.nn.Parameter(torch.Tensor(embedding_dim_relation,
                                            embedding_dim_entity,
                                            embedding_dim_entity), requires_grad=True)
        self.register_parameter('W', w)

        self.inp_drop = torch.nn.Dropout(input_dropout_rate)
        self.hidden0_drop = torch.nn.Dropout(hidden0_dropout_rate)
        self.hidden1_drop = torch.nn.Dropout(hidden1_dropout_rate)

        self.bn0 = torch.nn.BatchNorm1d(self.embedding_dim_entity)
        self.bn1 = torch.nn.BatchNorm1d(self.embedding_dim_entity)

    def reset_parameters(self):
        torch.nn.init.uniform_(self.W, -1, 1)

    def forward(self, lhs, rel, rhs):
        assert lhs.size(1) == 1, 'Only support TAIL mode'

        lhs = lhs[:, 0]
        rel = rel[:, 0]

        lhs = self.bn0(lhs)
        lhs = self.inp_drop(lhs)

        x = torch.einsum('bi, ijk->bjk', rel, self.W)
        x = self.hidden0_drop(x)
        x = torch.einsum('bj, bjk->bk', lhs, x)
        x = self.bn1(x)
        x = self.hidden1_drop(x)
        x = torch.einsum('bk,bnk->bn', x, rhs)

        return x
