# -*- coding: utf-8 -*-
from package.model import KEModel, KEEncoder, KEDecoder
from torch_geometric.nn import RGCNConv

import torch


class RGCNModel(KEModel):

    def __init__(self, A, num_entities, num_relations, embedding_dim,
                 hidden_uint, hidden_dropout, num_bases):
        super(RGCNModel, self).__init__()

        self.encoder = RGCNEncoder(A, num_entities, num_relations, embedding_dim,
                                   hidden_uint, hidden_dropout, num_bases)
        self.decoder = RGCNDecoder()

    def forward(self, lhs, rel, rhs):
        lhs, rel, rhs = self.encoder(lhs, rel, rhs)
        score = self.decoder(lhs, rel, rhs)
        return score

    def get_regularization_loss(self, lhs, rel, rhs):
        return 0.


class RGCNEncoder(KEEncoder):

    def __init__(self, A,
                 num_entities, num_relations, embedding_dim,
                 hidden_uint, hidden_dropout, num_bases):
        super(RGCNEncoder, self).__init__()

        edge1_index, edge_type, edge2_index = A.T

        self.edge_type = edge_type
        self.edge_index = torch.stack([edge1_index, edge2_index], dim=0)

        self.num_entities = num_entities
        self.num_relations = num_relations

        self.E = torch.nn.Embedding(num_entities, embedding_dim)
        self.R = torch.nn.Embedding(num_relations, embedding_dim)

        self.conv1 = RGCNConv(embedding_dim, hidden_uint, num_relations=num_relations, num_bases=num_bases)
        self.conv2 = RGCNConv(hidden_uint, embedding_dim, num_relations=num_relations, num_bases=num_bases)

        self.hidden_dropout = torch.nn.Dropout(hidden_dropout)

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


class RGCNDecoder(KEDecoder):

    def __init__(self):
        super(RGCNDecoder, self).__init__()

    def reset_parameters(self):
        pass

    def forward(self, lhs, rel, rhs):
        # batch_size, num_sample, embedding_dim

        if lhs.size(1) != 1:  # head
            x = torch.einsum('ijk, ijk->ij', (rhs * rel), lhs)
        else:  # tail
            x = torch.einsum('ijk, ijk->ij', (lhs * rel), rhs)

        return x
