import os
from glob import glob
from os.path import basename, join, splitext

import numpy as np


class SCPDataset:
    def __init__(self, scp_path):
        with open(scp_path, 'r') as f:
            scp_content = [item.strip().split() for item in f]
        self.utt_ids = [item[0] for item in scp_content]
        self.filepaths = {item[0]: item[1] for item in scp_content}

    def __getitem__(self, idx):
        utt_id = self.utt_ids[idx]
        filepath = self.filepaths[utt_id]
        return utt_id, filepath

    def __len__(self):
        return len(self.utt_ids)


class NpyDataset:
    def __init__(self, in_dir):
        filepaths = sorted(glob(join(in_dir, '*.npy')))
        self.utt_ids = [splitext(basename(item))[0] for item in filepaths]
        self.filepaths = {key: value for key, value in zip(self.utt_ids, filepaths)}

    def __getitem__(self, idx):
        utt_id = self.utt_ids[idx]
        filepath = self.filepaths[utt_id]
        return utt_id, np.load(filepath)

    def __len__(self):
        return len(self.utt_ids)


class MultipleNpyDataset:
    def __init__(self, *in_dirs):
        filepaths = [sorted(glob(join(in_dir, '*.npy'))) for in_dir in in_dirs]
        for item in filepaths:
            if len(item) != len(filepaths[0]):
                raise ValueError('the numbers of utterances must be the same:'
                                 f'{[len(item) for item in filepaths]}')
        self.utt_ids = [splitext(basename(item))[0] for item in filepaths[0]]
        self.filepaths = dict()
        for i, utt_id in enumerate(self.utt_ids):
            self.filepaths[utt_id] = [item[i] for item in filepaths]

    def __getitem__(self, idx):
        utt_id = self.utt_ids[idx]
        filepaths = self.filepaths[utt_id]
        return utt_id, [np.load(item) for item in filepaths]

    def __len__(self):
        return len(self.utt_ids)
