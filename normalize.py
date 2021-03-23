import argparse
import os
import pickle

import numpy as np
from tqdm import tqdm

from datasets import NpyDataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in-dir', type=str, required=True,
                        help='Directory containing data.')
    parser.add_argument('--scaler', type=str, required=True,
                        help='Dumped scaler file.')
    parser.add_argument('--dumpdir', type=str, required=True,
                        help='Directory to dump features.')
    args = parser.parse_args()

    os.makedirs(args.dumpdir, exist_ok=True)

    with open(args.scaler, 'rb') as f:
        scaler = pickle.load(f)

    dataset = NpyDataset(args.in_dir)

    for utt_id, data in tqdm(dataset):
        data = scaler.transform(data)
        np.save(os.path.join(args.dumpdir, f'{utt_id}.npy'), data,
                allow_pickle=False)
