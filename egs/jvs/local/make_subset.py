'''Corpus-dependent code for train/dev/eval splitting.'''
import argparse
from glob import glob
import os
from os.path import basename, exists, join, splitext
from tqdm import tqdm
import yaml


def get_utt_id(speaker, filepath):
    return f'{speaker}-{splitext(basename(filepath))[0]}'


def to_hts_format(lab_path):
    label_new = []
    with open(lab_path, 'r') as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        st, ed, label = line.split()
        if i == 0 and float(ed) > 100:
            return  # it is already written in hts format
        st = f'{int(1e+7 * float(st))}'
        ed = f'{int(1e+7 * float(ed))}'
        label_new.append(' '.join([st, ed, label]))

    with open(lab_path, 'w') as f:
        f.write('\n'.join(label_new) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--db-root', type=str, required=True,
                        help='Directory including target corpus.')
    parser.add_argument('--outdir', type=str, required=True,
                        help='Directory to output scp files.')
    parser.add_argument('--config', type=str, required=True,
                        help='YAML format configuration file.')
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    dev_size = config['dev_size']
    eval_utt_nums = config['eval_utt_nums']
    print(f'validation set size per speaker: {dev_size}')
    print(f'utterances used for evaluation set: {eval_utt_nums}')

    speakers_all = [f'jvs{i+1:03d}' for i in range(100)]
    subsets = ('train_nodev', 'dev', 'eval')
    lab_scp_contents = {subset: '' for subset in subsets}
    wav_scp_contents = {subset: '' for subset in subsets}

    for speaker in tqdm(speakers_all):
        # nonpara30
        nonpara_wavs = sorted(glob(join(
            args.db_root, speaker, 'nonpara30', 'wav24kHz16bit', '*')))
        for i, wav_path in enumerate(nonpara_wavs):
            lab_path = wav_path.replace('wav24kHz16bit', 'lab/ful').replace('wav', 'lab')
            if not exists(lab_path):
                continue
            to_hts_format(lab_path)
            utt_id = get_utt_id(speaker, wav_path)
            subset = 'dev' if i < dev_size else 'train_nodev'
            lab_scp_contents[subset] += f'{utt_id} {lab_path}\n'
            wav_scp_contents[subset] += f'{utt_id} {wav_path}\n'

        # parallel100
        parallel_wavs = sorted(glob(join(
            args.db_root, speaker, 'parallel100', 'wav24kHz16bit', '*')))
        for i, wav_path in enumerate(parallel_wavs):
            lab_path = wav_path.replace('wav24kHz16bit', 'lab/ful').replace('wav', 'lab')
            if not exists(lab_path):
                continue
            to_hts_format(lab_path)
            utt_id = get_utt_id(speaker, wav_path)
            subset = 'eval' if utt_id[-3:] in eval_utt_nums else 'train_nodev'
            lab_scp_contents[subset] += f'{utt_id} {lab_path}\n'
            wav_scp_contents[subset] += f'{utt_id} {wav_path}\n'

    os.makedirs(args.outdir, exist_ok=True)
    for subset in subsets:
        with open(join(args.outdir, f'lab_{subset}.scp'), 'w') as f:
            f.write(lab_scp_contents[subset])
        with open(join(args.outdir, f'wav_{subset}.scp'), 'w') as f:
            f.write(wav_scp_contents[subset])
