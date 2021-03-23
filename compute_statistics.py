import argparse
import pickle

from tqdm import tqdm

from datasets import NpyDataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in-dir', type=str, required=True,
                        help='Directory containing data.')
    parser.add_argument('--out-path', type=str, required=True,
                        help='Path to save scaler file.')
    parser.add_argument('--type', type=str, default='meanvar',
                        help='minmax or meanvar.')
    args = parser.parse_args()

    if args.type == 'meanvar':
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
    elif args.type == 'minmax':
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
    else:
        raise ValueError(f'--type argument must be meanvar or minmax, not {args.type}.')

    dataset = NpyDataset(args.in_dir)

    for _, data in tqdm(dataset):
        scaler.partial_fit(data)

    with open(args.out_path, 'wb') as f:
        pickle.dump(scaler, f)
