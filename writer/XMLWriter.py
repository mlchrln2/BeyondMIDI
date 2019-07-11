'''Module for converting XML meta data to audio labels'''

import argparse
from fractions import Fraction
import numpy as np
import pandas as pd
from music21 import *


def XMLWriter(in_path, out_path):
    '''Data writer that converts XML file metadata to data labels

    Args: in_path
            in_path (string): Location of the XML file.
    '''

    song = converter.parse(in_path)
    instrument = ''
    measure = 0
    time_num, time_denom = 0, 0
    key = ''
    tie = ''
    note = ''
    octave = 0
    duration = []
    dynamic = ''
    data = []
    metadata = song.recurse()
    for msg in metadata:
        if msg.classes[0] == 'Tempo':
            print('here')
        if msg.classes[0] == 'Instrument':
            instrument = msg
        if msg.classes[0] == 'Measure':
            measure += 1
        if msg.classes[0] == 'KeySignature':
            key = msg.key
        if msg.classes[0] == 'TimeSignature':
            time_num = msg.numerator
            time_denom = msg.denominator
        if msg.classes[0] == 'Dynamic':
            dynamic = msg
        if msg.classes[0] == 'Note':
            note = msg.name
            octave = msg.octave
            time = Fraction(metadata.currentHierarchyOffset())
            if msg.tie:
                if msg.tie.type == 'start':
                    duration.append(time)
                    data.append([measure, note, octave])
            else:
                duration.append(time)
                data.append([measure, note, octave])
        if msg.classes[0] == 'Rest':
            note = msg.name
            time = Fraction(metadata.currentHierarchyOffset())
            duration.append(time)
            data.append([measure, note, 0])
    duration.append(time + time_num - time % time_num)
    duration = np.array(duration)
    duration = duration[1:] - duration[0:-1]
    data = np.hstack((np.asarray(data), duration.reshape(-1, 1)))
    data_frame = pd.DataFrame(
        {'measure': data[:, 0], 'note': data[:, 1], 'octave': data[:, 2], 'beat': data[:, 3]})
    data_frame.to_csv(out_path, index=False)


def main():
    '''Main method for XML writer'''
    parser = argparse.ArgumentParser(
        description='XML File parser and data writer')
    parser.add_argument('in_path', metavar='input directory',
                        help='path to midi file')
    parser.add_argument('out_path', metavar='output directory',
                        help='path to output csv file')
    args = parser.parse_args()
    XMLWriter(args.in_path, args.out_path)

main()
