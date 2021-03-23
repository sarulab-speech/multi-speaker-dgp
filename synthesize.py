import argparse
import logging
import os
import pickle
import sys

import numpy as np
import pysptk
import pyworld
import torch
import yaml

from nnmnkwii.frontend import merlin as fe
from nnmnkwii.io import hts
from nnmnkwii.paramgen import mlpg
from scipy.io import wavfile
from tqdm import tqdm

from datasets import SCPDataset
import models


class Synthesizer:
    def __init__(
        self,
        duration_model,
        acoustic_model,
        scaler,
        config,
        speakers,
        question_path,
        device,
    ):
        self.duration_model = duration_model
        self.acoustic_model = acoustic_model
        self.scaler = scaler
        self.binary_dict, self.continuous_dict = hts.load_question_set(question_path)
        self.device = device
        self.config = config

        # prepare window
        if config['n_delta'] in (0, 1, 2):
            windows = [
                (0, 0, np.array([1.0])),
                (1, 1, np.array([-0.5, 0.0, 0.5])),
                (1, 1, np.array([1.0, -2.0, 1.0])),
            ]
            self.windows = windows[:config['n_delta'] + 1]
        else:
            raise ValueError('only n_delta = 0, 1, 2 is supported')

        self.lf0_start_idx = (config['mcep_order'] + 1) * (config['n_delta'] + 1)
        self.vuv_start_idx = self.lf0_start_idx + (config['n_delta'] + 1)
        self.bap_start_idx = self.vuv_start_idx + 1

        # prepare speaker codes dictionary
        with open(speakers, 'r') as f:
            speakers = [item.strip() for item in f]
        speaker_codes = np.eye(len(speakers), dtype=np.float32)
        self.speaker_codes = {speakers[i]: speaker_codes[i] for i in range(len(speakers))}

    def add_speaker_code(self, utt_id, feature):
        speaker_id = utt_id.split('-')[0]
        speaker_code = np.tile(self.speaker_codes[speaker_id], (len(feature), 1))
        return np.hstack([feature, speaker_code])

    def gen_duration(self, utt_id, label_path):
        # prepare phoneme-level linguistic feature
        labels = hts.load(lab_path)

        feature = fe.linguistic_features(
            labels, self.binary_dict, self.continuous_dict,
            add_frame_features=False,
            subphone_features=None).astype(np.float32)

        # normalize
        feature = self.scaler['X']['duration'].transform(feature)

        # add speaker information
        feature = self.add_speaker_code(utt_id, feature)

        # predict phoneme durations
        feature = torch.from_numpy(feature).to(device)
        duration = self.duration_model.predict(feature)['mean'].data.cpu().numpy()

        # denormalize
        duration = self.scaler['Y']['duration'].inverse_transform(duration)
        duration = np.round(duration)

        # set minimum duration to 1
        duration[duration <= 0] = 1
        labels.set_durations(duration)

        return labels

    def gen_parameters(self, utt_id, labels):
        feature = fe.linguistic_features(
            labels, self.binary_dict, self.continuous_dict,
            add_frame_features=True,
            subphone_features='coarse_coding').astype(np.float32)

        # normalize
        feature = scaler['X']['acoustic'].transform(feature)

        # add speaker information
        feature = self.add_speaker_code(utt_id, feature)

        # predict acoustic features
        feature = torch.from_numpy(feature).to(device)
        pred = self.acoustic_model.predict(feature)
        pred_mean = pred['mean'].data.cpu().numpy()
        pred_var = pred['var'].data.cpu().numpy()

        # denormalize
        scale = self.scaler['Y']['acoustic'].scale_
        pred_mean = self.scaler['Y']['acoustic'].inverse_transform(pred_mean)
        pred_var *= scale ** 2

        # split acoustic features
        mgc = pred_mean[:, :self.lf0_start_idx]
        lf0 = pred_mean[:, self.lf0_start_idx:self.vuv_start_idx]
        vuv = pred_mean[:, self.vuv_start_idx]
        bap = pred_mean[:, self.bap_start_idx:]

        # make variances for Maximum Likelihood Parameter Generation (MLPG)
        mgc_variances = pred_var[:, :self.lf0_start_idx]
        lf0_variances = pred_var[:, self.lf0_start_idx:self.vuv_start_idx]
        bap_variances = pred_var[:, self.bap_start_idx:]

        # perform MLPG to calculate static features
        mgc = mlpg(mgc, mgc_variances, self.windows)
        lf0 = mlpg(lf0, lf0_variances, self.windows)
        bap = mlpg(bap, bap_variances, self.windows)

        feature = np.hstack([mgc, lf0, vuv.reshape(-1, 1), bap])

        return feature

    def gen_waveform(self, feature):
        mcep_dim = self.config['mcep_order'] + 1
        mgc = feature[:, :mcep_dim]
        lf0 = feature[:, mcep_dim:mcep_dim + 1]
        vuv = feature[:, mcep_dim + 1: mcep_dim + 2]
        bap = feature[:, mcep_dim + 2:]

        spectrogram = pysptk.mc2sp(
            mgc,
            fftlen=self.config['fft_size'],
            alpha=pysptk.util.mcepalpha(self.config['sampling_rate']),
        )
        aperiodicity = pyworld.decode_aperiodicity(
            bap.astype(np.float64),
            self.config['sampling_rate'],
            self.config['fft_size'],
        )
        f0 = lf0.copy()
        f0[vuv < 0.5] = 0
        f0[np.nonzero(f0)] = np.exp(f0[np.nonzero(f0)])

        waveform = pyworld.synthesize(
            f0.flatten().astype(np.float64),
            spectrogram.astype(np.float64),
            aperiodicity.astype(np.float64),
            self.config['sampling_rate'],
            self.config['hop_size_in_ms'],
        )
        return waveform

    def synthesize(self, utt_id, lab_path):
        duration_modified_label = self.gen_duration(utt_id, lab_path)
        params = self.gen_parameters(utt_id, duration_modified_label)
        waveform = self.gen_waveform(params)
        return waveform


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--labdir', type=str, required=True,
                        help='Kaldi-style scp file of full-context labels.')
    parser.add_argument('--outdir', type=str, required=True,
                        help='Directory to save synthesized waveform.')
    parser.add_argument('--duration-checkpoint', type=str, required=True,
                        help='Checkpoint file of phoneme duration model.')
    parser.add_argument('--acoustic-checkpoint', type=str, required=True,
                        help='Checkpoint file of acoustic model.')
    parser.add_argument('--speakers', type=str, required=True,
                        help='Text file including names of all speakers')
    parser.add_argument('--question-path', type=str, required=True,
                        help='Path to question set file.')
    parser.add_argument('--config', type=str, default=None,
                        help='YAML format configuration file.')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format='%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s')

    # load config
    if args.config is None:
        dirname = os.path.dirname(args.duration_checkpoint)
        args.config = os.path.join(dirname, 'config.yaml')
    with open(args.config) as f:
        config = yaml.safe_load(f)
    config.update(vars(args))

    # get dataset
    dataset = SCPDataset(args.labdir)

    # get scaler
    scaler = {'X': dict(), 'Y': dict()}
    dirname = os.path.dirname(args.duration_checkpoint)
    with open(os.path.join(dirname, 'X_duration.pkl'), 'rb') as f:
        scaler['X']['duration'] = pickle.load(f)
    with open(os.path.join(dirname, 'Y_duration.pkl'), 'rb') as f:
        scaler['Y']['duration'] = pickle.load(f)
    dirname = os.path.dirname(args.acoustic_checkpoint)
    with open(os.path.join(dirname, 'X_acoustic.pkl'), 'rb') as f:
        scaler['X']['acoustic'] = pickle.load(f)
    with open(os.path.join(dirname, 'Y_acoustic.pkl'), 'rb') as f:
        scaler['Y']['acoustic'] = pickle.load(f)

    # setup model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    duration_model_class = getattr(models, config['duration_config']['model_class'])
    duration_model = duration_model_class(
        **config['duration_config']['model_params']).eval().to(device)
    logging.info(f'Loading checkpoint from {args.duration_checkpoint}.')
    duration_checkpoint = torch.load(args.duration_checkpoint, map_location='cpu')
    duration_model.load_state_dict(duration_checkpoint['model_state_dict'])
    logging.info('phoneme duration model:')
    logging.info(duration_model)

    acoustic_model_class = getattr(models, config['acoustic_config']['model_class'])
    acoustic_model = acoustic_model_class(
        **config['acoustic_config']['model_params']).eval().to(device)
    logging.info(f'Loading checkpoint from {args.acoustic_checkpoint}.')
    acoustic_checkpoint = torch.load(args.acoustic_checkpoint, map_location='cpu')
    acoustic_model.load_state_dict(acoustic_checkpoint['model_state_dict'])
    logging.info('acoustic model:')
    logging.info(acoustic_model)

    # setup synthesizer
    synthesizer = Synthesizer(
        duration_model,
        acoustic_model,
        scaler,
        config,
        args.speakers,
        args.question_path,
        device,
    )

    # start generation
    with torch.no_grad(), tqdm(dataset) as pbar:
        for idx, (utt_id, lab_path) in enumerate(pbar, 1):
            waveform = synthesizer.synthesize(utt_id, lab_path)
            wavfile.write(os.path.join(args.outdir, f'{utt_id}_gen.wav'),
                          config['sampling_rate'],
                          waveform.astype(np.int16))

    logging.info(f'Finished generation of {idx} utterances.')
