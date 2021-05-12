# -*- coding: utf-8 -*-
from torch import LongTensor, Tensor
from torch.nn import Module
from typing import Union


class KEModel(Module):

    def __init__(self):
        super(KEModel, self).__init__()

        self.encoder: Union[KEEncoder, None] = None
        self.decoder: Union[KEDecoder, None] = None

    def reset_parameters(self):
        if self.encoder is not None:
            self.encoder.reset_parameters()
        if self.decoder is not None:
            self.decoder.reset_parameters()

    @property
    def entities_embedding(self):
        return self.encoder.entities_embedding

    @property
    def relations_embedding(self):
        return self.encoder.relations_embedding

    def get_regularization_loss(self, lhs: LongTensor, rel: LongTensor, rhs: LongTensor):
        raise NotImplemented()

    def forward(self, lhs: LongTensor, rel: LongTensor, rhs: LongTensor):
        raise NotImplemented()


class KEEncoder(Module):

    def __init__(self):
        super(KEEncoder, self).__init__()

    def reset_parameters(self):
        pass

    @property
    def entities_embedding(self):
        raise NotImplemented()

    @property
    def relations_embedding(self):
        raise NotImplemented()

    def get_relation_embedding(self, token: LongTensor):
        raise NotImplemented()

    def get_entity_embedding(self, token: LongTensor):
        raise NotImplemented()

    def forward(self, lhs: LongTensor, rel: LongTensor, rhs: LongTensor):
        raise NotImplemented()


class KEDecoder(Module):

    def __init__(self):
        super(KEDecoder, self).__init__()

    def reset_parameters(self):
        raise NotImplemented()

    def forward(self, lhs: Tensor, rel: Tensor, rhs: Tensor):
        raise NotImplemented()
