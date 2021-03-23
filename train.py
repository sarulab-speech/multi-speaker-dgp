import argparse
import logging
import os
import sys
from collections import defaultdict
from copy import deepcopy

import numpy as np
import torch
import yaml
from tensorboardX import SummaryWriter
from tqdm import tqdm

import models
from datasets import MultipleNpyDataset


class SqueezeCollater:
    '''
    This class loads `buffer_size` utterances and concatenate them along axis 0.
    By randomly sampling from the large array, this class makes frame-level batch.
    '''
    def __init__(self, speakers):
        '''
        Args:
            speakers (str): path to text file containing all speakers' ID.
            shuffle (bool): whether to shuffle frames included in a batch.
        '''
        with open(speakers, 'r') as f:
            speakers = [item.strip() for item in f]
        speaker_codes = np.eye(len(speakers), dtype=np.float32)
        self.speaker_codes = {speakers[i]: speaker_codes[i] for i in range(len(speakers))}

    def __call__(self, batch):
        '''
        Args:
            batch (tuple of numpy.ndarray): pair of input/output arrays.

        Returns:
            torch.Tensor: mini-batch frame-level input tensor, shape (T_1 + ... + T_B, D_in + N_speakers).
            torch.Tensor: mini-batch frame-level output tensor, shape (T_1 + ... + T_B, D_out).
        '''
        # parse batch
        utt_ids = [utt[0] for utt in batch]
        inputs = [utt[1][0] for utt in batch]
        outputs = [utt[1][1] for utt in batch]

        # add speaker information to inputs
        for i, (utt_id, in_data) in enumerate(zip(utt_ids, inputs)):
            speaker_id = utt_id.split('-')[0]
            speaker_code = np.tile(self.speaker_codes[speaker_id], (len(in_data), 1))
            inputs[i] = np.hstack([in_data, speaker_code])

        # stack inputs and outputs
        inputs = np.vstack(inputs)
        outputs = np.vstack(outputs)
        assert inputs.shape[0] == outputs.shape[0], f'{inputs.shape}, {outputs.shape}'

        shuffled_indices = np.random.permutation(inputs.shape[0])
        inputs = inputs[shuffled_indices]
        outputs = outputs[shuffled_indices]

        return torch.from_numpy(inputs), torch.from_numpy(outputs)


class BatchWiseIterator():
    '''
    This class gets 'batch_size' samples from large tensor in each iteration.

    Examples:
        >>> inputs = torch.Tensor([1, 2, 3, 4, 5, 6, 7, 8, 9])
        >>> outputs = torch.Tensor([0, 0, 1, 0, 1, 0, 1, 1, 0])
        >>> b = BatchWiseIterator(batch_size=3)
        >>> b.set_tensors(inputs, outputs)
        >>> for (x, y) in b:
        ...     print(x, y)
        ...
        tensor([1., 2., 3.]) tensor([0., 0., 1.])
        tensor([4., 5., 6.]) tensor([0., 1., 0.])
        tensor([7., 8., 9.]) tensor([1., 1., 0.])
    '''
    def __init__(self, batch_size):
        self.x, self.y = None, None
        self.batch_size = batch_size

    def set_tensors(self, x, y):
        if self.x is None:
            self.x, self.y = x, y
        else:
            self.x = torch.cat([self.x, x], dim=0)
            self.y = torch.cat([self.y, y], dim=0)

    def __iter__(self):
        return self

    def __next__(self):
        x, y = self.x[:self.batch_size], self.y[:self.batch_size]
        if len(x) < self.batch_size:
            raise StopIteration
        self.x, self.y = self.x[self.batch_size:], self.y[self.batch_size:]
        return x, y


class Trainer:
    def __init__(self, steps, epochs, dataloader, model, optimizer, criterion, config, device):
        self.steps = steps
        self.epochs = epochs
        self.dataloader = dataloader
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.config = config
        self.device = device
        self.writer = SummaryWriter(os.path.join(config['outdir'], 'tensorboard', config['mode']))
        self.step_loss = defaultdict(float)
        self.epoch_loss = {'train': defaultdict(float),
                           'dev': defaultdict(float)}

    def run(self):
        iterator = BatchWiseIterator(batch_size=self.config['batch_size'])
        estimated_total_steps = self.model.data_size * self.config['epochs'] // self.config['batch_size']
        self.tqdm = tqdm(initial=self.steps,
                         total=estimated_total_steps,
                         desc='[train]')

        while self.epochs < self.config['epochs']:
            for phase in ('train', 'dev'):
                if phase == 'train':
                    model.train()
                else:
                    model.eval()

                steps_tmp = 0
                for (X, Y) in dataloader[phase]:
                    iterator.set_tensors(X, Y)
                    for (x, y) in iterator:
                        if phase == 'train':
                            self._step(x, y)
                            self._step_log()
                        else:
                            with torch.no_grad():
                                self._step(x, y)
                        steps_tmp += 1

                self._epoch_log(phase, steps_tmp)

            # save checkpoint
            self.epochs += 1
            if self.epochs % config['save_interval_epochs'] == 0:
                self.save_checkpoint(os.path.join(self.config['outdir'],
                                     f"checkpoint-{self.config['mode']}-{self.epochs}epochs.pkl"))
            logging.info(f'Saved checkpoint @ epoch {self.epochs}.')

        self.tqdm.close()

    def _step(self, x, y):
        '''A single step of training/validation.'''
        x, y = x.to(device), y.to(device)
        self.optimizer.zero_grad()

        f_pred = self.model(x)

        mse = self.criterion(f_pred['mean'], y).item()

        if self.model.training:
            bound, bound_info = self.model.calc_bound(f_pred, y)
            loss = -bound
            loss.backward()
            self.optimizer.step()

            self.step_loss['train/predictive_expectation'] += bound_info['predictive_expectation']
            for key in bound_info['kld']:
                self.step_loss[f'train/kld/{key}'] += bound_info['kld'][key]
            self.step_loss['train/mse'] += mse

            self.epoch_loss['train']['train/predictive_expectation'] += bound_info['predictive_expectation']
            for key in bound_info['kld']:
                self.epoch_loss['train'][f'train/kld/{key}'] += bound_info['kld'][key]
            self.epoch_loss['train']['train/mse'] += mse

            self.steps += 1
            self.tqdm.update(1)

        else:
            self.epoch_loss['dev']['dev/mse'] += mse

    def _step_log(self):
        if self.steps % self.config['log_interval_steps'] == 0:
            for key in self.step_loss:
                self.step_loss[key] /= self.config['log_interval_steps']
                logging.info(f'(Steps: {self.steps}) {key} = {self.step_loss[key]:.4f}.')
            for key, value in self.step_loss.items():
                self.writer.add_scalar(f'step/{key}', value, self.steps)
            self.step_loss = defaultdict(float)

    def _epoch_log(self, phase, interval):
        for key in self.epoch_loss[phase]:
            self.epoch_loss[phase][key] /= interval
            logging.info(f'(Epoch {self.epochs + 1}) {key} = {self.epoch_loss[phase][key]:.4f}.')
        for key, value in self.epoch_loss[phase].items():
            self.writer.add_scalar(f'epoch/{key}', value, self.epochs + 1)
        self.epoch_loss[phase] = defaultdict(float)

    def load_checkpoint(self, checkpoint_path):
        logging.info(f'Loading checkpoint from {checkpoint_path}.')
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.steps = checkpoint['steps']
        self.epochs = checkpoint['epochs']

    def save_checkpoint(self, checkpoint_path):
        print(f'Saving model and optimizer state at epoch {self.epochs} to {checkpoint_path}.')
        torch.save({'steps': self.steps,
                    'epochs': self.epochs,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict()},
                   checkpoint_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-in-dir', type=str, required=True,
                        help='Directory including input data for training.')
    parser.add_argument('--train-out-dir', type=str, required=True,
                        help='Directory including output data for training.')
    parser.add_argument('--dev-in-dir', type=str, required=True,
                        help='Directory including input data for validation.')
    parser.add_argument('--dev-out-dir', type=str, required=True,
                        help='Directory including output data for validation.')
    parser.add_argument('--outdir', type=str, required=True,
                        help='Directory to save checkpoints.')
    parser.add_argument('--config', type=str, required=True,
                        help='YAML format configuration file.')
    parser.add_argument('--speakers', type=str, required=True,
                        help='Text file including names of all speakers')
    parser.add_argument('--mode', type=str, default='acoustic',
                        help='duration or acoustic')
    parser.add_argument('--resume', type=str, default=None, nargs='?',
                        help='Checkpoint file path to resume training')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format='%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s')

    # load and save config
    with open(args.config) as f:
        config = yaml.safe_load(f)
    with open(os.path.join(args.outdir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f, Dumper=yaml.Dumper)
    config = config[f'{args.mode}_config']
    config.update(vars(args))
    for key, value in config.items():
        logging.info(f'{key} = {value}')

    # set random seed
    np.random.seed(config['random_state'])
    torch.manual_seed(config['random_state'])
    torch.cuda.manual_seed(config['random_state'])

    # prepare dataloaders
    train_dataset = MultipleNpyDataset(args.train_in_dir, args.train_out_dir)
    dev_dataset = MultipleNpyDataset(args.dev_in_dir, args.dev_out_dir)
    collater = SqueezeCollater(args.speakers)
    dataloader = {
        'train': torch.utils.data.DataLoader(
            train_dataset, batch_size=config['buffer_size'], shuffle=True,
            num_workers=config['num_workers'], collate_fn=collater,
            pin_memory=config['pin_memory'], drop_last=False,
        ),
        'dev': torch.utils.data.DataLoader(
            dev_dataset, batch_size=config['buffer_size'], shuffle=False,
            num_workers=config['num_workers'], collate_fn=collater,
            pin_memory=config['pin_memory'], drop_last=False,
        )}

    logging.info('Count the number of total frames for training')
    data_size = 0
    for utt in tqdm(train_dataset):
        data_size += len(utt[1][0])
    logging.info(f'The number of total frames for training: {data_size}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_class = getattr(models, config['model_class'])
    model = model_class(**config['model_params'], data_size=data_size).to(device)
    logging.info(model)

    optimizer = torch.optim.Adam(model.parameters(), **config['optimizer_params'])

    criterion = torch.nn.MSELoss().to(device)

    # define trainer
    trainer = Trainer(
        steps=0,
        epochs=0,
        dataloader=dataloader,
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        config=config,
        device=device)

    # resume from checkpoint
    if args.resume is not None:
        trainer.load_checkpoint(args.resume)

    # run training loop
    trainer.run()
