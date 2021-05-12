# -*- coding: utf-8 -*-
from zipfile import ZipFile

import requests


def add_reciprocal_relation(relation2id, triplets, prefix='inv@'):
    num_relation = len(relation2id)
    relation2id_all = {prefix + k: v + num_relation for k, v in relation2id.items()}
    relation2id_all.update(relation2id)

    triplets_all = []
    for data in triplets:
        reverse_data = []
        for _1, _2, _3 in data:
            reverse_data.append((_3, prefix + _2, _1))

        data += reverse_data
        triplets_all.append(data)

    print(f"Adding reciprocal relations\n"
          f"  number of relations: {len(relation2id_all)}\n")

    return relation2id_all, triplets_all


def read_zip(path: str, mode='hrt'):
    data = {'train.tsv': [], 'valid.tsv': [], 'test.tsv': []}

    entity2id = {}
    relation2id = {}
    with ZipFile(path, 'r') as zp:
        for file in data:
            with zp.open(file) as fp:
                _, _, triplets = read_tsv(fp, entity2id, relation2id, mode)
                data[file].extend(triplets)

    return entity2id, relation2id, data['train.tsv'], data['valid.tsv'], data['test.tsv']


def read_tsv(fp, entity2id: dict = None, relation2id: dict = None, mode='hrt'):
    assert mode in ('hrt', 'rht', 'htr')
    entity2id = {} if entity2id is None else entity2id
    relation2id = {} if relation2id is None else relation2id
    triplets = []

    while True:

        line = fp.readline()

        if isinstance(line, bytes):
            line = line.decode()

        line = line.strip()

        if not line:
            break

        _1, _2, _3 = line.split('\t')

        # default 'hrt'
        if mode == 'rht':
            _1, _2, _3 = _2, _1, _3
        elif mode == 'htr':
            _1, _2, _3 = _1, _3, _2

        if _1 not in entity2id:
            entity2id[_1] = len(entity2id)

        if _2 not in relation2id:
            relation2id[_2] = len(relation2id)

        if _3 not in entity2id:
            entity2id[_3] = len(entity2id)

        triplets.append((_1, _2, _3))

    return entity2id, relation2id, triplets


URL = "https://docs.google.com/uc?export=download"


def download_from_google_drive(id, destination):
    print("Downloading from google drive. Trying to fetch {}".format(destination))

    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value

        return None

    def save_response_content(response, destination):
        CHUNK_SIZE = 32768

        with open(destination, "wb") as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)

    session = requests.Session()

    response = session.get(URL, params={'id': id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)
