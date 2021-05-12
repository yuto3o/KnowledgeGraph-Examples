# -*- coding: utf-8 -*-
import os


def save_to_disk(path, x_observed, y_observed, x_hidden, y_hidden, rules, supports):
    if not os.path.isdir(path):
        os.mkdir(path)

    with open(os.path.join(path, 'rules.txt'), 'w', encoding='utf-8') as f:
        for rule, support in zip(rules, supports):
            f.write(f"{serialize_rule(rule)}\t{support}\n")

    with open(os.path.join(path, 'observed.txt'), 'w', encoding='utf-8') as f:
        for x, y in zip(x_observed, y_observed):
            f.write(f"{','.join([str(_) for _ in x])}\t{','.join([str(_) for _ in y])}\n")

    with open(os.path.join(path, 'hidden.txt'), 'w', encoding='utf-8') as f:
        for x, y in zip(x_hidden, y_hidden):
            f.write(f"{','.join([str(_) for _ in x])}\t{','.join([str(_) for _ in y])}\n")


def load_from_disk(path):
    rules = []
    precisions = []
    x_observed = []
    y_observed = []
    x_hidden = []
    y_hidden = []

    with open(os.path.join(path, 'rules.txt'), 'r', encoding='utf-8') as f:
        while True:

            line = f.readline().strip('\n')

            if not line: break

            rule, precision = line.split('\t')

            rules.append(deserialize_rule(rule))
            precisions.append(float(precision))

    with open(os.path.join(path, 'observed.txt'), 'r', encoding='utf-8') as f:
        while True:

            line = f.readline().strip('\n')

            if not line: break

            x, y = line.split('\t')
            h, r, t = [int(e) for e in x.split(',')]
            y = set([int(e) for e in y.split(',') if e])

            x_observed.append((h, r, t))
            y_observed.append(y)

    with open(os.path.join(path, 'hidden.txt'), 'r', encoding='utf-8') as f:
        while True:

            line = f.readline().strip('\n')

            if not line: break

            x, y = line.split('\t')
            h, r, t = [int(e) for e in x.split(',')]
            y = set([int(e) for e in y.split(',') if e])

            x_hidden.append((h, r, t))
            y_hidden.append(y)

    return x_observed, y_observed, x_hidden, y_hidden, rules, precisions


def serialize_rule(rule):
    fmt = ''
    for b in rule.body:
        fmt += f"{b} "
    fmt += f"-> {rule.head}: {rule.type}"
    return fmt


def deserialize_rule(x, integer=False):
    body, head = x.split(' -> ')
    head, cls = head.split(': ')

    if integer:
        head, cls = int(head), int(cls)

    body = [int(b) if integer else b for b in body.split(' ')]

    return cls, head, body
