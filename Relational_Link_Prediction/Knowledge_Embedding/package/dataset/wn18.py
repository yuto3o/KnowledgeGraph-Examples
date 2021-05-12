# -*- coding: utf-8 -*-
from package.dataset.utils import read_zip, download_from_google_drive
from os.path import isfile, join


def WN18(path_to_home):
    path_to_zip = join(path_to_home, 'WN18.zip')
    if not isfile(path_to_zip):
        google_device_id = '1dwfVH00W8u8IQ6M91R2Tur8MOXrocAGz'
        download_from_google_drive(google_device_id, path_to_zip)

    entity2id, relation2id, train, valid, test = read_zip(path_to_zip, 'hrt')
    msg = f"Loading from WN18\n" \
          f"  number of entities: {len(entity2id)}\n" \
          f"  number of relations: {len(relation2id)}\n" \
          f"  number of train: {len(train)}\n" \
          f"  number of valid: {len(valid)}\n" \
          f"  number of test: {len(test)}\n"
    print(msg)

    return entity2id, relation2id, train, valid, test
