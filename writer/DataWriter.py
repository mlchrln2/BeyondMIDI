'''
This module is responsible for writing data to disk so that it is easier to load different types of
data for training and inference in the future. Some translation tools are defined in this module in
order to facilitate this process. Those tools include a .mp3 to .wav translator, an a .wav to .h5
file translator. The files saved include .h5 files filled with instrument samples such as exerpts
and single note recordings.
'''

import argparse
import os
import h5py


def audio_writer(in_path, out_path='.', out_file='out.h5'):
    '''Method for storing sound file information into a .h5 file. This method will traverse through
    every file in the in_path directory and will store the information in the out_file at the
    out_path location. This method assumes that the data stored in the audio files is stored
    channel_first and is stored in .wav format. The audio files will be normalized between 0 and 1
    and will be saved according to their names in the in_path directory without their extensions.

    Args: path, out_file
      in_path (string): The location of the sound file directory.
      out_path (string): The location to store the output .h5 file.
                         Default is the current directory.
      out_file (string): The name of the output .h5 file to be saved. Default is 'out.h5'.
    '''
    import torchaudio
    mp3_dirs = [in_path + '/' + directory for directory in os.listdir(in_path)]
    mp3_files = []
    while mp3_dirs:
        curr_path = mp3_dirs.pop()
        if os.path.isdir(curr_path):
            for directory in os.listdir(curr_path):
                mp3_dirs.append(curr_path + '/' + directory)
        else:
            mp3_files.append(curr_path)
    if not os.path.isdir(out_path):
        os.makedirs(out_path)
    audio_frame = h5py.File(out_path + out_file, 'w')
    total_files = len(mp3_files)
    for num_file, file in enumerate(mp3_files):
        clip = torchaudio.load(filepath=file,
                               out=None,
                               normalization=True,
                               channels_first=True,
                               num_frames=0,
                               offset=0,
                               signalinfo=None,
                               encodinginfo=None,
                               filetype=None)[0]
        file_name = file.split('/')[-1].split('.wav')[0]
        audio_frame.create_dataset(file_name, data=clip.numpy())
        print('file {} of {} written'.format(
            num_file + 1, total_files), end='\r')
    audio_frame.close()

def main():
    '''Main method for data writing'''
    parser = argparse.ArgumentParser(
        description='Dataset Organizer and Writer')
    parser.add_argument('in_path', metavar='DIR', help='path to dataset')
    parser.add_argument('out_path', metavar='DIR',
                        help='output path for the data writer')
    parser.add_argument('out_file', metavar='DIR',
                        help='output fle for the data writer')
    args = parser.parse_args()
    audio_writer(in_path=args.in_path,
                 out_path=args.out_path,
                 out_file=args.out_file)

main()
