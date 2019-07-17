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
    '''Data loader class for reading the Rose Etude data from the .h5 file stored in path.

    Args: path
        path (string): The location of the Rose Etudes .h5 files.
        data_name (string): Name of the Rose Etudes data .h5 file.
        labels_name (string): Name of the Rose Etudes label .h5 file.
    '''
    def __init__(self, path, data_name, labels_name):
        self.rose_data_frame = h5py.File(path + data_name, 'r')
        self.rose_data_keys = list(self.rose_data_frame.keys())
        self.rose_labels_frame = h5py.File(path + labels_name, 'r')
        self.rose_labels_keys = list(self.rose_labels_frame.keys())
        # the number of frames to include from the file
        self.num_frames = int(params['sound_duration'] * 44100)

    def __len__(self):
        return len(self.rose_data_keys)

    def __getitem__(self, idx):
        rose_data = torch.from_numpy(
            self.rose_data_frame[self.rose_data_keys[idx]][:self.num_frames])
        rose_labels = self.rose_labels_frame[self.rose_labels_keys[idx]][:, 3:5]
        rose_labels = torch.tensor([self.name_to_midi(note, octave) for note, octave in
                                    zip(rose_labels[:, 0], rose_labels[:, 1])])
        return rose_data, rose_labels
    def name_to_midi(self, note, octave):
        '''Method for converting between note names and midi labels

        Input: note, octave
            note (string): The name of the note to be converted to midi.
            octave (int): The octave of the note to be converted to midi.

        Output: midi
            midi (int): The midi note corresponding to the input.
        '''
        name = {b'rest': 0,
                b'C-': -1, b'C': 0, b'C#': 1, b'C##': 2,
                b'D-': 1, b'D': 2, b'D#': 3,
                b'E-':3, b'E': 4, b'E#': 5,
                b'F-': 4, b'F': 5, b'F#': 6, b'F##': 7,
                b'G-': 6, b'G': 7, b'G#': 8, b'G##': 9,
                b'A-': 8, b'A': 9, b'A#': 10,
                b'B--': 9, b'B-': 10, b'B': 11, b'B#': 12}
        midi = name[note] + (int(octave) + 1) * 12
        return midi


class Philharmonia(Dataset):
    '''Data loader class for reading the Philharmonia data from the .h5 file stored in path.

    Args: path
        path (string): The location of the Philharmonia .h5 file.
        name (string): Name of the Philharmonia .h5 file.
    '''
    def __init__(self, path, name):
        self.phil_frame = h5py.File(path + name, 'r')
        phil_keys = np.array(list(self.phil_frame.keys()))
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
        self.labels = torch.tensor([
            self.name_to_midi(info) for info in self.information[:, 1]]).long()

    def __len__(self):
        return len(self.phil_keys)

    def __getitem__(self, idx):
        phil_data = torch.from_numpy(
            self.phil_frame[self.phil_keys[idx]][:]).float()
        phil_labels = self.labels[idx].long()
        return phil_data, phil_labels
    def name_to_midi(self, note):
        '''Method for converting note name labels to midi labels
        Input: note
            note (string): Note name to convert to midi

        Output: midi
            midi (int): output midi note
        '''
        note_names = 'C Cs D Ds E F Fs G Gs A As B'.split(' ')
        midi = (note_names.index(note[:-1]))+(int(note[-1])+1)*12
        return midi

DATASETS = {'Rose Etudes': RoseEtudes('../data/audio_data/', 'Rose_Data.h5', 'Rose_Labels.h5'),
            'Philharmonia': Philharmonia('../data/audio_data/', 'Phil.h5')}


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
