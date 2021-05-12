# -*- coding: utf-8 -*-
from package.model import KEModel, KEEncoder, KEDecoder
from math import floor
import torch


class ConvEModel(KEModel):

    def __init__(self, num_entities, num_relations, embedding_dim,
                 embedding_shape, hidden_unit,
                 input_dropout, feature_map_dropout, hidden_dropout):
        super(ConvEModel, self).__init__()

        self.encoder = ConvEEncoder(num_entities, num_relations, embedding_dim)
        self.decoder = ConvEDecoder(num_entities, embedding_dim,
                                    embedding_shape, hidden_unit,
                                    input_dropout, feature_map_dropout, hidden_dropout)

    def forward(self, lhs, rel, rhs):
        rhs_idx = rhs
        lhs, rel, rhs = self.encoder(lhs, rel, rhs)
        score = self.decoder(lhs, rel, rhs, rhs_idx)
        return score

    def get_regularization_loss(self, lhs, rel, rhs):
        return 0.


class ConvEEncoder(KEEncoder):

    def __init__(self, num_entities, num_relations, embedding_dim):
        super(ConvEEncoder, self).__init__()

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


class ConvEDecoder(KEDecoder):

    def __init__(self,
                 num_entities, embedding_dim,
                 embedding_shape, hidden_unit,
                 input_dropout, feature_map_dropout, hidden_dropout):
        super(ConvEDecoder, self).__init__()

        self.input_dropout = torch.nn.Dropout(input_dropout)
        self.feature_map_dropout = torch.nn.Dropout2d(feature_map_dropout)
        self.hidden_dropout = torch.nn.Dropout(hidden_dropout)
        self.embedding_shape0 = embedding_shape
        self.embedding_shape1 = embedding_dim // embedding_shape

        self.conv1 = torch.nn.Conv2d(1, hidden_unit, (3, 3))
        self.bn0 = torch.nn.BatchNorm2d(1)
        self.bn1 = torch.nn.BatchNorm2d(hidden_unit)
        self.bn2 = torch.nn.BatchNorm1d(embedding_dim)

        self.register_parameter('b', torch.nn.Parameter(torch.zeros(num_entities)))
        hidden_unit = floor((self.embedding_shape0 * 2 - 2) * (self.embedding_shape1 - 2) * hidden_unit)
        self.fc = torch.nn.Linear(hidden_unit, embedding_dim)

    def reset_parameters(self):
        pass

    def forward(self, lhs, rel, rhs, rhs_idx):
        assert lhs.size(1) == 1, 'Only support TAIL mode'

        h = lhs.view(-1, 1, self.embedding_shape0, self.embedding_shape1)
        r = rel.view(-1, 1, self.embedding_shape0, self.embedding_shape1)

        x = torch.cat([h, r], dim=2)
        x = self.bn0(x)
        x = self.input_dropout(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.feature_map_dropout(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.hidden_dropout(x)
        x = self.bn2(x)
        x = torch.relu(x).view(x.size(0), -1)
        x = torch.einsum('ik, ijk->ij', x, rhs)
        x += self.b[rhs_idx].expand_as(x)

        return x
