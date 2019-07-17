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
from mido import tick2second


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


def midi_writer(in_path, out_path='out.h5'):
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
    time = 0
    key = ''
    tempo = {}
    time_numer, time_denom = 0, 0
    for msg in rose_midi.tracks[0]:
        time += msg.time
        if msg.type == 'key_signature':
            key = msg.key
        if msg.type == 'time_signature':
            time_numer = msg.numerator
            time_denom = msg.denominator
        if msg.type == 'set_tempo':
            tempo[time] = msg.tempo
    time = 0
    note = []
    on, off = [], []
    for msg in rose_midi.tracks[1]:
        time += msg.time
        if msg.type == 'note_on':
            note_on = msg.note
            velocity_on = msg.velocity
            time_on = msg.time
            note.append(midi2name(note_on, 2))
            on.append(time_on)
        if msg.type == 'note_off':
            note_off = msg.note
            time_off = msg.time
            velocity_off = msg.velocity
            off.append(time_off)
        print(msg)
    for a, b in zip(on, off):
        print(a / rose_midi.ticks_per_beat, b / rose_midi.ticks_per_beat)
    print(time / rose_midi.ticks_per_beat)
    #data_frame = pd.DataFrame({'note': note})
    #data_frame.to_csv(out_path, index=False)


def main():
    '''Main method for midi writing'''
    parser = argparse.ArgumentParser(
        description='Dataset Organizer and Writer')
    parser.add_argument('in_path', metavar='input directory',
                        help='path to midi file')
    parser.add_argument('out_path', metavar='output directory',
                        help='output path for the midi writer')
    args = parser.parse_args()
    midi_writer(in_path=args.in_path,
                out_path=args.out_path)

main()
