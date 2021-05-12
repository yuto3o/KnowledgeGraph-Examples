# -*- coding: utf-8 -*-
from package.dataset import benchmark
from package.dataset.utils import add_reciprocal_relation
from package.tokenizer import Tokenizer
from package.application import ComplExModel
from package.losses import LabelSmoothingCELoss
from package.data import Dataset, AlternateDataLoader
from package.utils import single_hop
from package.trainer import Trainer

from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from pprint import pprint

import torch

benchmark_name = 'FB15k-237'
train_batch_size = 256
test_batch_size = 256
num_epoch = 50

num_workers = 4
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

embedding_dim = 200
lr = 1e-3
weight_decay = 5e-2
label_smoothing = 0.

time_stamp = datetime.now().strftime('%Y%m%d%H%M%S')
tag = f"{benchmark_name}-{time_stamp}"
tb = SummaryWriter(tag)

if __name__ == '__main__':
    entity2id, relation2id, train, valid, test = benchmark(benchmark_name)
    relation2id, (train, valid, test) = add_reciprocal_relation(relation2id, (train, valid, test))

    tokenizer = Tokenizer(entity2id, relation2id)
    train = tokenizer.tokenize(train)
    valid = tokenizer.tokenize(valid)
    test = tokenizer.tokenize(test)

    fwd = single_hop(train + valid + test)

    train_dataset = AlternateDataLoader(
        [Dataset(train, tokenizer.num_entities,
                 sampling_mode=Dataset.SAMPLING_MODE_1VN, corrupt_mode=Dataset.CORRUPT_MODE_TAIL)],
        batch_size=train_batch_size, num_workers=num_workers, shuffle=True)
    test_dataset = AlternateDataLoader(
        [Dataset(test, tokenizer.num_entities,
                 sampling_mode=Dataset.SAMPLING_MODE_1VN, corrupt_mode=Dataset.CORRUPT_MODE_TAIL)],
        batch_size=test_batch_size, num_workers=num_workers)

    model = ComplExModel(num_entities=tokenizer.num_entities, num_relations=tokenizer.num_relations,
                         embedding_dim=embedding_dim)
    model = model.to(device)
    model.reset_parameters()
    criterion = LabelSmoothingCELoss(epsilon=label_smoothing)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    tokenizer.save(tag)

    trainer = Trainer(model, optimizer, weight_decay, criterion, train_dataset, test_dataset)

    for i in range(1, 1 + num_epoch):
        trainer.train_reciprocal_epoch(i, tb)

        metrics = trainer.test_reciprocal(fwd)
        pprint(metrics)

        tb.add_scalar('Test/MRR', metrics['mrr'], i)
        tb.add_scalar('Test/MR', metrics['mr'], i)
        tb.add_scalar('Test/Hits@1', metrics['hits_at_1'], i)
        tb.add_scalar('Test/Hits@3', metrics['hits_at_3'], i)
        tb.add_scalar('Test/Hits@10', metrics['hits_at_10'], i)

        torch.save(model.state_dict(), tag + f"/model_{i}_{metrics['mrr']}.pth")
