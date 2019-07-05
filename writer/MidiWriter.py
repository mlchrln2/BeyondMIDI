'''
This module is responsible for writing midi metadata to disk so that it is easier and faster
to load during trainig and inference. The file type saved is a .h5 file.
'''


import argparse
from fractions import Fraction
import pandas as pd
import numpy as np
from mido import MidiFile
from mido import tempo2bpm


def midi2name(midi, offset=0):
    '''Method for converting midi note values to note names

    Args: midi
        midi (int): The midi note value.
        offset (int): Value to transpose note by.

    Output: name
        name (string): The name of the note corresponding to the given midi note value.
                       If the midi value is a zero it is interpreted as a rest.
    '''
    if midi == 0:
        return 'R'
    note_val = midi + offset
    note_names = 'C C# D D# E F F# G G# A A# B'.split(' ')
    note = note_names[note_val % 12]
    octave = note_val // 12 - 1
    name = '{}{}'.format(note, octave)
    return name


def vel2dyn(velocity):
    '''Method for labeling the dynamic of a message

    Args: velocity
        velocity (float): The velocity written in the midi file message.

    Output: word
        word (string): Dynamic marking for a given velocity.
                       If it is not in the list return an empty string.
    '''
    dyn_dict = {20: 'ppp', 31: 'pp', 42: 'p',
                53: 'mp', 64: 'mf', 80: 'f',
                96: 'ff', 112: 'fff', 127: 'ffff'}
    try:
        word = dyn_dict[velocity]
    except:
        word = velocity
    return word


def midi_writer(etude_number, in_path,out_path='.', out_file='out.h5'):
    '''Method for storing midi file information into a .h5 file. This method will parse through
    a midi file and extract important metadata for data labeling. This method assumes that the data
    is stored in the midi files is stored in the standard .mid format of type 0 (single track),

    Args: path, out_file
      in_path (string): The location of the midi file directory.
      out_path (string): The location to store the output .h5 file.
                                         Default is the current directory.
      out_file (string): The name of the output .h5 file to be saved. Default is 'out.h5'.
    '''
    rose_midi = MidiFile(in_path)
    etude = {1: {'start': 1, 'end': 43, 'down_beat': 1},
             2: {'start': 43, 'end': 50, 'down_beat': 1}}[etude_number]
    played = {}
    sign = 0
    ticks = 0
    upper_signature = 0
    lower_signature = 0
    measure = 1
    rhythm = etude['down_beat'] - 1
    tempo = 0
    itr = 0
    rhythm_list = []
    beats_list = []
    note_list = []
    dynamics_list = []
    tempo_list = []
    measure_list = []
    for msg in rose_midi.tracks[0]:
        if msg.type == 'time_signature':
            upper_signature = msg.numerator
            lower_signature = msg.denominator
        if msg.type == 'set_tempo':
            '''
            if etude_number != 0:
                data_frame = pd.DataFrame({'measure': measure_list, 'rhythm': rhythm_list,
                                           'beat value': beats_list, 'note': note_list,
                                           'dynamics': dynamics_list, 'tempo': tempo_list})
                out_file_arr = out_file.split('.')
                out = '{}{}_{}.{}'.format(out_path, out_file_arr[0], etude_number, out_file_arr[1])
                data_frame.to_csv(out, index=False)
                measure = 1
            etude_number += 1
            tempo = int(tempo2bpm(msg.tempo))
            '''
        if msg.type == 'note_on':
            note = msg.note
            velocity = msg.velocity
            if velocity != 0:
                played_vals = np.array(list(played.values()))
                sign = np.any(played_vals != 0)
                played[note] = velocity
                ticks = msg.time - 1
            else:
                sign = 1
                dynamics = vel2dyn(played[note])
                played[note] = velocity
                ticks = msg.time + 1
            if ticks > 0:
                beats = Fraction(ticks * lower_signature,
                                 rose_midi.ticks_per_beat * 4)
                note *= sign
                measure_list.append(measure)
                rhythm_list.append(float(rhythm))
                beats_list.append(float(beats))
                note_list.append(midi2name(note, 2))
                if measure == etude['start'] and itr != etude_number:
                    rhythm = etude['down_beat'] - 1
                    itr += 1
                if measure >= etude['start'] and measure < etude['end']:
                    print('measure: {}, start: {}, end: {}, duration: {} note name: {}'.format(measure - etude['start'] + 1, rhythm % upper_signature + 1, (rhythm + beats) % upper_signature + 1, beats, midi2name(note, 2)))
                if note != 0:
                    dynamics_list.append(dynamics)
                else:
                    dynamics_list.append('NA')
                tempo_list.append(tempo)
                rhythm += beats
                measure = rhythm // upper_signature + 1

def main():
    '''Main method for midi writing'''
    parser = argparse.ArgumentParser(
        description='Dataset Organizer and Writer')
    parser.add_argument('in_path', metavar='input directory',
                        help='path to midi file')
    parser.add_argument('out_path', metavar='output directory',
                        help='output path for the midi writer')
    parser.add_argument('out_file', metavar='output file',
                        help='output fle for the midi writer')
    parser.add_argument('etude_number', metavar='etude number', type=int,
                        help='the etude number of the rose etudes')
    args = parser.parse_args()
    midi_writer(etude_number=args.etude_number,
                in_path=args.in_path,
                out_path=args.out_path,
                out_file=args.out_file)

main()
