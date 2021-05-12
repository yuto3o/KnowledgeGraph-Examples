# -*- coding: utf-8 -*-
# using a brute force method where we search for all possible composition rules, inverse rules, symmetric rules and
# subrelation rules in the triplets.
from collections import namedtuple, defaultdict, OrderedDict
from multiprocessing import Pool
from functools import reduce
from tqdm import tqdm
from math import ceil

Rule = namedtuple('Rule', ['type', 'head', 'body'])

RULE_COMPOSITION = 0
RULE_SYMMETRIC = 1
RULE_INVERSE = 2
RULE_SUBRELATION = 3


def rule_mining(x, num_workers, chunk_size):
    """
    Brute Force Method
    Parameters
    ----------
    x: array-like, (N, 3)
        [(h, r, t), ...]
    num_workers: int
    chunk_size: int

    Returns
    -------

    """

    hop = defaultdict(list)
    for h, r, t in x:
        hop[h].append((r, t))

    rules = set()
    num_x = len(x)
    num_chunk = ceil(num_x / chunk_size)

    with Pool(num_workers) as p:
        rules = list(
            tqdm(p.imap(rule_mining_thread,
                        [(i * chunk_size, min((i + 1) * chunk_size, num_x), x, hop, rules) for i in range(num_chunk)]),
                 total=num_chunk, desc='Mining Rules ...')
        )

    rules = reduce(lambda x, y: x | y, rules)

    return list(rules)


def rule_mining_thread(args):
    bg, ed, x, hop, rules = args
    rules = set()
    for h, r, t in x[bg:ed]:
        rule_mining_composition(h, r, t, hop, rules)
        rule_mining_symmetric(h, r, t, hop, rules)
        rule_mining_inverse(h, r, t, hop, rules)
        rule_mining_subrelation(h, r, t, hop, rules)
    return rules


def rule_mining_composition(h, r, t, hop, rules):
    for r1, t1 in hop[h]:
        for r2, t2 in hop[t1]:
            if t2 != t:
                continue

            rule = Rule(RULE_COMPOSITION, r, (r1, r2))
            rules.add(rule)


def rule_mining_symmetric(h, r, t, hop, rules):
    for r1, t1 in hop[t]:

        if r1 != r:
            continue
        if t1 != h:
            continue

        rule = Rule(RULE_SYMMETRIC, r, (r1,))
        rules.add(rule)


def rule_mining_inverse(h, r, t, hop, rules):
    for r1, t1 in hop[t]:

        if r1 == r:
            continue
        if t1 != h:
            continue

        rule = Rule(RULE_INVERSE, r, (r1,))
        rules.add(rule)


def rule_mining_subrelation(h, r, t, hop, rules):
    for r1, t1 in hop[h]:

        if t1 != t:
            continue
        if r1 == r:
            continue

        rule = Rule(RULE_SUBRELATION, r, (r1,))
        rules.add(rule)


def rule_precision(x, rules, num_workers, chunk_size):
    """
    Compute the empirical precision of each rule
    Parameters
    ----------
    x: array-like, (N, 3)
       [(h, r, t), ...]
    rules: list or tuple
       [Rule, ...], Rule.type, Rule.head, Rule.body
    num_workers: int
       number of workers
    chunk_size: int

    Returns
    -------

    """

    hop = defaultdict(list)
    observed_x_set = set()

    for idx, (h, r, t) in enumerate(x):
        hop[h].append((r, t))
        observed_x_set.add((h, r, t))

    num_rules = len(rules)
    num_chunk = ceil(num_rules / chunk_size)

    with Pool(num_workers) as p:
        precision = list(
            tqdm(p.imap(rule_precision_thread,
                        [(i * chunk_size, min((i + 1) * chunk_size, num_rules),
                          rules, x, hop, observed_x_set) for i in
                         range(num_chunk)]),
                 total=num_chunk, desc='Supporting Rules ...')
        )

    return sum(precision, [])


def rule_precision_thread(args):
    bg, ed, rules, x, hop, observed_x_set = args

    precision = []
    for idx in range(bg, ed):
        if rules[idx].type == RULE_COMPOSITION:
            p = rule_precision_composition(idx, rules, x, hop, observed_x_set)

        elif rules[idx].type == RULE_SYMMETRIC:
            p = rule_precision_symmetric(idx, rules, x, hop, observed_x_set)

        elif rules[idx].type == RULE_INVERSE:
            p = rule_precision_inverse(idx, rules, x, hop, observed_x_set)

        else:  # Rule.RULE_SUBRELATION
            p = rule_precision_subrelation(idx, rules, x, hop, observed_x_set)

        precision.append(p)

    return precision


def rule_precision_composition(idx, rules, x, hop, observed_x_set):
    p = 0.
    q = 0.

    r1t, r2t = rules[idx].body
    r = rules[idx].head

    for h1, r1, t1 in x:
        if r1 != r1t:
            continue

        for r2, t2 in hop[t1]:

            if r2 != r2t:
                continue

            if (h1, r, t2) in observed_x_set:
                p += 1
            q += 1

    return p / q


def rule_precision_symmetric(idx, rules, x, hop, observed_x_set):
    p = 0.
    q = 0.

    r1t = rules[idx].body[0]
    r = rules[idx].head

    for h1, r1, t1 in x:
        if r1 != r1t:
            continue

        if (t1, r, h1) in observed_x_set:
            p += 1
        q += 1

    return p / q


def rule_precision_inverse(idx, rules, x, hop, observed_x_set):
    p = 0.
    q = 0.

    r1t = rules[idx].body[0]
    r = rules[idx].head

    for h1, r1, t1 in x:
        if r1 != r1t:
            continue

        if (t1, r, h1) in observed_x_set:
            p += 1
        q += 1

    return p / q


def rule_precision_subrelation(idx, rules, x, hop, observed_x_set):
    p = 0.
    q = 0.

    r1t = rules[idx].body[0]
    r = rules[idx].head

    for h1, r1, t1 in x:
        if r1 != r1t:
            continue

        if (h1, r, t1) in observed_x_set:
            p += 1
        q += 1

    return p / q


def rule_grounding(x, rules, num_workers, chunk_size):
    """
    Simple Rule Grounding
    Parameters
    ----------
    x: array-like, (N, 3)
       [(h, r, t), ...]
    rules: list or tuple
       [Rule, ...], Rule.type, Rule.head, Rule.body
    num_workers: int
       number of workers
    chunk_size: int

    Returns
    -------

    """

    hop = defaultdict(list)

    observed_x_set = set()
    observed_x = OrderedDict()
    hidden_x = defaultdict(set)

    for idx, (h, r, t) in enumerate(x):
        hop[h].append((r, t))
        observed_x_set.add((h, r, t))

    num_rules = len(rules)
    num_chunk = ceil(num_rules / chunk_size)

    with Pool(num_workers) as p:
        res = list(
            tqdm(p.imap(rule_grounding_thread,
                        [(i * chunk_size, min((i + 1) * chunk_size, num_rules),
                          rules, x, hop, observed_x_set) for i in
                         range(num_chunk)]),
                 total=num_chunk, desc='Grounding Rules ...')
        )

    # keep the order
    for idx, (h, r, t) in enumerate(x):
        observed_x[h, r, t] = set()

    for observed_x_, hidden_x_ in res:

        for k, v in observed_x_.items():
            observed_x[k] |= v

        for k, v in hidden_x_.items():
            hidden_x[k] |= v

    x = x
    y = list(observed_x.values())
    x_ = list(hidden_x.keys())
    y_ = list(hidden_x.values())

    return x, y, x_, y_


def rule_grounding_thread(args):
    bg, ed, rules, x, hop, observed_x_set = args

    observed_x = defaultdict(set)
    hidden_x = defaultdict(set)
    for idx in range(bg, ed):
        if rules[idx].type == RULE_COMPOSITION:
            rule_grounding_composition(idx, rules, x, hop, observed_x_set, observed_x, hidden_x)

        elif rules[idx].type == RULE_SYMMETRIC:
            rule_grounding_symmetric(idx, rules, x, hop, observed_x_set, observed_x, hidden_x)

        elif rules[idx].type == RULE_INVERSE:
            rule_grounding_inverse(idx, rules, x, hop, observed_x_set, observed_x, hidden_x)

        else:  # Rule.RULE_SUBRELATION
            rule_grounding_subrelation(idx, rules, x, hop, observed_x_set, observed_x, hidden_x)

    return observed_x, hidden_x


def rule_grounding_composition(idx, rules, x, hop, observed_x_set, observed_x, hidden_x):
    r1t, r2t = rules[idx].body
    r = rules[idx].head

    for h1, r1, t1 in x:
        if r1 != r1t:
            continue

        for r2, t2 in hop[t1]:

            if r2 != r2t:
                continue

            if (h1, r, t2) in observed_x_set:
                observed_x[(h1, r, t2)].add(idx)
            else:
                hidden_x[(h1, r, t2)].add(idx)


def rule_grounding_symmetric(idx, rules, x, hop, observed_x_set, observed_x, hidden_x):
    r1t = rules[idx].body[0]
    r = rules[idx].head

    for h1, r1, t1 in x:
        if r1 != r1t:
            continue

        if (t1, r, h1) in observed_x_set:
            observed_x[(t1, r, h1)].add(idx)
        else:
            hidden_x[(t1, r, h1)].add(idx)


def rule_grounding_inverse(idx, rules, x, hop, observed_x_set, observed_x, hidden_x):
    r1t = rules[idx].body[0]
    r = rules[idx].head

    for h1, r1, t1 in x:
        if r1 != r1t:
            continue

        if (t1, r, h1) in observed_x_set:
            observed_x[(t1, r, h1)].add(idx)
        else:
            hidden_x[(t1, r, h1)].add(idx)


def rule_grounding_subrelation(idx, rules, x, hop, observed_x_set, observed_x, hidden_x):
    r1t = rules[idx].body[0]
    r = rules[idx].head

    for h1, r1, t1 in x:
        if r1 != r1t:
            continue

        if (h1, r, t1) in observed_x_set:
            observed_x[(h1, r, t1)].add(idx)
        else:
            hidden_x[(h1, r, t1)].add(idx)
