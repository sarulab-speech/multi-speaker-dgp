import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import os
import yaml

import librosa
import numpy as np
import pysptk
import pyworld

from nnmnkwii.frontend import merlin as fe
from nnmnkwii.io import hts
from nnmnkwii.preprocessing.f0 import interp1d
from nnmnkwii.util import apply_delta_windows
from scipy.io import wavfile
from tqdm import tqdm

from datasets import SCPDataset


def get_linguistic_feature(lab_path, question_path, level='phone'):
    if level == 'phone':
        add_frame_features = False
        subphone_features = None
    elif level == 'frame':
        add_frame_features = True
        subphone_features = 'coarse_coding'
    else:
        raise ValueError(f'phone and frame are supported, but level={level} is given.')

    binary_dict, continuous_dict = hts.load_question_set(question_path)
    labels = hts.load(lab_path)
    feature = fe.linguistic_features(
        labels, binary_dict, continuous_dict,
        add_frame_features=add_frame_features,
        subphone_features=subphone_features)

    if add_frame_features:
        indices = labels.silence_frame_indices().astype(int)
    else:
        indices = labels.silence_phone_indices()
    feature = np.delete(feature, indices, axis=0)

    return feature.astype(np.float32)


def get_duration(lab_path):
    labels = hts.load(lab_path)
    feature = fe.duration_features(labels)
    indices = labels.silence_phone_indices()
    feature = np.delete(feature, indices, axis=0)
    return feature.astype(np.float32)


def get_acoustic_feature(lab_path, wav_path, sampling_rate, hop_size_in_ms, mcep_order, windows):
    fs, audio = wavfile.read(wav_path)
    audio = audio.astype(np.float64) / 2 ** 15
    if fs != sampling_rate:
        audio = audio.astype(np.float32)
        audio = librosa.resample(audio, fs, sampling_rate)
        audio = (audio * 2 ** 15).astype(np.float64)
    # extract f0
    f0, timeaxis = pyworld.dio(audio, sampling_rate, frame_period=hop_size_in_ms)
    # modify f0
    f0 = pyworld.stonemask(audio, f0, timeaxis, sampling_rate)
    # voiced/unvoiced flag
    vuv = (f0 > 0)[:, None].astype(np.float32)
    # calculate log f0
    lf0 = f0.copy()
    nonzero_indices = np.nonzero(f0)
    lf0[nonzero_indices] = np.log(f0[nonzero_indices])
    # interpolate f0 in log-domain
    lf0 = interp1d(lf0, kind='slinear')[:, None]

    # calculate mel-cepstrum
    spectrogram = pyworld.cheaptrick(audio, f0, timeaxis, sampling_rate)
    mgc = pysptk.sp2mc(spectrogram, order=mcep_order,
                       alpha=pysptk.util.mcepalpha(sampling_rate))
    # calculate aperiodicity parameter
    aperiodicity = pyworld.d4c(audio, f0, timeaxis, sampling_rate)
    bap = pyworld.code_aperiodicity(aperiodicity, sampling_rate)

    # calculate dynamic features
    mgc = apply_delta_windows(mgc, windows)
    lf0 = apply_delta_windows(lf0, windows)
    bap = apply_delta_windows(bap, windows)

    feature = np.hstack((mgc, lf0, vuv, bap))

    # cut silence frames by HTS alignment
    labels = hts.load(lab_path)
    feature = feature[:labels.num_frames()]
    if labels.num_frames() > len(feature):
        return
    indices = labels.silence_frame_indices()
    feature = np.delete(feature, indices, axis=0)

    return feature.astype(np.float32)


def _process(idx, label_dataset, wav_dataset, args, config):
    utt_id_lab, lab_path = label_dataset[idx]
    utt_id_wav, wav_path = wav_dataset[idx]
    assert utt_id_lab == utt_id_wav, f'{utt_id_lab} != {utt_id_wav}'
    utt_id = utt_id_lab

    # feature for phoneme duration model
    linguistic_phone = get_linguistic_feature(lab_path, args.question_path, level='phone')
    duration = get_duration(lab_path)

    # feature for acoustic model
    if config['n_delta'] in (0, 1, 2):
        windows = [
            (0, 0, np.array([1.0])),
            (1, 1, np.array([-0.5, 0.0, 0.5])),
            (1, 1, np.array([1.0, -2.0, 1.0])),
        ]
        windows = windows[:config['n_delta'] + 1]
    else:
        raise ValueError('only n_delta = 0, 1, 2 is supported')

    linguistic_frame = get_linguistic_feature(lab_path, args.question_path, level='frame')
    acoustic = get_acoustic_feature(
        lab_path,
        wav_path,
        config['sampling_rate'],
        config['hop_size_in_ms'],
        config['mcep_order'],
        windows
    )
    if acoustic is None:
        print(f'UserWarning: frame-level linguistic feature is longer than acoustic feature. '
              f'Example with utt_id: {utt_id} will not be used.')
        return
    np.save(os.path.join(args.dumpdir, 'X_duration', f'{utt_id}.npy'),
            linguistic_phone, allow_pickle=False)
    np.save(os.path.join(args.dumpdir, 'Y_duration', f'{utt_id}.npy'),
            duration, allow_pickle=False)
    np.save(os.path.join(args.dumpdir, 'X_acoustic', f'{utt_id}.npy'),
            linguistic_frame, allow_pickle=False)
    np.save(os.path.join(args.dumpdir, 'Y_acoustic', f'{utt_id}.npy'),
            acoustic, allow_pickle=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lab-scp', type=str, required=True,
                        help='Scp file including label paths.')
    parser.add_argument('--wav-scp', type=str, required=True,
                        help='Scp file including wav paths.')
    parser.add_argument('--question-path', type=str, required=True,
                        help='Path to question set file.')
    parser.add_argument('--dumpdir', type=str, required=True,
                        help='Directory to dump features.')
    parser.add_argument('--config', type=str, required=True,
                        help='YAML format configuration file.')
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    os.makedirs(os.path.join(args.dumpdir, 'X_duration'), exist_ok=True)
    os.makedirs(os.path.join(args.dumpdir, 'Y_duration'), exist_ok=True)
    os.makedirs(os.path.join(args.dumpdir, 'X_acoustic'), exist_ok=True)
    os.makedirs(os.path.join(args.dumpdir, 'Y_acoustic'), exist_ok=True)

    label_dataset = SCPDataset(args.lab_scp)
    wav_dataset = SCPDataset(args.wav_scp)
    assert len(label_dataset) == len(wav_dataset), \
        f'len(label_dataset): {len(label_dataset)} != len(wav_dataset): {len(wav_dataset)}'

    futures = []
    with ProcessPoolExecutor(config['n_processes']) as executor:
        for idx in range(len(label_dataset)):
            futures.append(executor.submit(partial(
                _process, idx, label_dataset, wav_dataset, args, config)))
        result = [future.result() for future in tqdm(futures)]
