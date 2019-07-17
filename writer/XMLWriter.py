'''Module for storing all the audio data labels in one large .h5 file. The input directory
must only contain MusicXML files to store the data.'''

import argparse
import os
from fractions import Fraction
import numpy as np
import h5py
from music21 import *


def XMLWriter(in_path, out_path='', out_file='out.h5'):
    '''Data writer that converts XML file metadata to data labels

    Args: in_path
            in_path (string): location of the XML file.
            out_path (string): location of the output file's directory.
            out_file (string): location of the output file
    '''
    # collect the files in the input directory
    xml_dirs = [in_path + directory for directory in os.listdir(in_path)]
    xml_files = []
    while xml_dirs:
        curr_path = xml_dirs.pop()
        if os.path.isdir(curr_path):
            for directory in os.listdir(curr_path):
                xml_dirs.append(curr_path + '/' + directory)
        else:
            xml_files.append(curr_path)
    if not os.path.isdir(out_path):
        os.makedirs(out_path)
    label_frame = h5py.File(out_path + out_file, 'w')
    total_files = len(xml_files)
    # loop through all of the files and store the labels in a .h5 file.
    for num_file, file in enumerate(xml_files):
        # initialize some important values
        measure = 0
        time_num, time_denom = 0, 0
        note, octave = '', 0
        dynamic = 'none'
        duration = [0]
        still_rest = True
        # append the start token and start time to the labels
        data = [['start', 'rest', 0, 'none']]
        # retrieve the metadata from the xml objects
        song = converter.parse(file)
        metadata = song.recurse()
        for msg in metadata:
            if msg.classes[0] == 'Note':
                note = msg.name
                octave = msg.octave
                time = Fraction(metadata.currentHierarchyOffset())
                # only store the first note from the tie if it is tied
                if msg.tie:
                    if msg.tie.type == 'start':
                        duration.append(time)
                        data.append([str(measure), note, octave, dynamic])
                # store the note if it is not tied
                else:
                    duration.append(time)
                    data.append([str(measure), note, octave, dynamic])
                # reset the rest flag in case another rest shows up
                still_rest = False
            # current rest
            elif msg.classes[0] == 'Rest':
                # only store the first rest if there are multiple rest chains
                if not still_rest:
                    # if the next note is a rest then the next pass will skip the
                    # if statement
                    still_rest = True
                    note = msg.name
                    time = Fraction(metadata.currentHierarchyOffset())
                    duration.append(time)
                    data.append([str(measure), note, 0, 'none'])
            # current measure
            elif msg.classes[0] == 'Measure':
                measure += 1
            # num: beats in a measure / denom: what constitutes one beat
            elif msg.classes[0] == 'TimeSignature':
                time_num = msg.numerator
                time_denom = msg.denominator
            # current dynamic
            elif msg.classes[0] == 'Dynamic':
                dynamic = msg.value
            # current played note
        # append the end time of the last note
        duration.append(time + time_num - time % time_num)
        # if the last data value appended was a rest then remove it before adding
        # the end token
        if still_rest:
            data.pop()
        else:
            duration.append(time + time_num + time_num - time % time_num)
        # append end token
        data.append(['end', 'rest', 0, 'none'])
        # cast to numpy array and concatenate labels with time
        data = np.asarray(data)
        duration = np.array(duration) * Fraction(time_denom / 4)
        start_beat = (duration[0:-1] % time_num) + 1
        duration = duration[1:] - duration[0:-1]
        # gather data and save the data
        data = np.hstack((data[:, 0].reshape(-1, 1), start_beat.reshape(-1, 1),
                          duration.reshape(-1, 1), data[:, 1:]))
        data = np.array(data, dtype='<S10')
        file_name = file.split('/')[-1].split('.xml')[0]
        label_frame.create_dataset(file_name, data=data, dtype='<S10')
        print('file {} of {} written'.format(
            num_file + 1, total_files), end='\r')
    label_frame.close()


def main():
    '''Main method for XML writer'''
    parser = argparse.ArgumentParser(
        description='XML File parser and data writer')
    parser.add_argument('in_path', metavar='input directory',
                        help='path to midi file')
    parser.add_argument('out_path', metavar='output directory',
                        help='path to output csv file')
    parser.add_argument('out_file', metavar='output file',
                        help='csv file name')
    args = parser.parse_args()
    XMLWriter(args.in_path, args.out_path, args.out_file)

main()
