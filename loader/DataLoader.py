'''
This module stores all of the data loading tools needed for GeneratorNet training and testing.
The tools available include a midi file information parser and a sound file parser.
'''

# dependencies
import random
import numpy as np
import h5py
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch

# user defined modules
from DataParameters import PARAMETERS as params


class RoseEtudes(Dataset):
    '''Data loader class for reading the Philharmonia data from the .h5 file stored in path.

    Args: path
        path (string): The location of the Philharmonia .h5 file.
        name (string): Name of the Philharmonia .h5 file.
    '''

    def __init__(self, path, name):
        self.rose_frame = h5py.File(path + name, 'r')
        self.rose_keys = list(self.rose_frame.keys())
        # the number of frames to include from the file
        self.num_frames = int(params['sound_duration'] * 44100)

    def __len__(self):
        return len(self.rose_keys)

    def __getitem__(self, idx):
        rose_data = torch.from_numpy(
            self.rose_frame[self.rose_keys[idx]][:self.num_frames])
        return rose_data


class Philharmonia(Dataset):
    '''Data loader class for reading the Rose Etude data from the .h5 file stored in path.

    Args: path
        path (string): The location of the Rose Etudes .h5 file.
        name (string): Name of the Rose Etudes .h5 file.
    '''

    def __init__(self, path, name):
        self.phil_frame = h5py.File(path + name, 'r')
        phil_keys = list(self.phil_frame.keys())
        # shuffle the keys so as to not bias the input data
        random.Random(4).shuffle(phil_keys)
        '''
        Information from the key names separated by the '_' delimiter:.
        Index 0: instrument (banjo, bass-clarinet, bassoon, ..., violin).
        Index 1: midi note (22, 23,24, ..., 108).
        Index 2: duration (025, 05, 1, ..., very long).
        Index 3: dynamics (pianissimo, piano, mezzo-piano, ... fortissimo).
        Index 4: style (normal, fluttertonguing, nonlegato, ..., glissando).
        '''
        information = np.array([key.split('_') for key in phil_keys])
        # only include samples with monophonic, dynamically stable sounds
        # played normally on the clarinet,
        useful_samples = [(inst == 'clarinet' and 'phrase' not in dur
                           and 'long' not in dur and 'cresc' not in dyn
                           and 'normal' in style)
                          for inst, dur, dyn, style in zip(information[:, 0],
                                                           information[:, 2],
                                                           information[:, 3],
                                                           information[:, 4])]
        self.phil_keys = phil_keys[useful_samples]
        self.information = information[useful_samples]
        # the labels are the note names
        self.labels = torch.from_numpy(
            self.information[:, 1].astype(int)).long()

    def __len__(self):
        return len(self.phil_keys)

    def __getitem__(self, idx):
        phil_data = torch.from_numpy(
            self.phil_frame[self.phil_keys[idx]][:]).float()

        return phil_data

DATASETS = {'Rose Etudes': RoseEtudes('../data/audio_data/', 'Rose.h5'),
            'Philharmonia': Philharmonia('../data/audio_data', 'Phil.h5')}


def get_loader(dataset, batch_size=1, shuffle=False,
               sampler=None, batch_sampler=None, num_workers=0,
               pin_memory=False, drop_last=False, timeout=0, worker_init_fn=None):
    '''Method for loading datasets with batching, samplers, and collate functions'''

    loader = DataLoader(dataset=DATASETS[dataset],
                        batch_size=batch_size,
                        shuffle=shuffle,
                        sampler=sampler,
                        batch_sampler=batch_sampler,
                        num_workers=num_workers,
                        pin_memory=pin_memory,
                        drop_last=drop_last,
                        timeout=timeout,
                        worker_init_fn=worker_init_fn)
    return loader
