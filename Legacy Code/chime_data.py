import json
import os
import pickle

import numpy as np
import tqdm
import scipy.io as io
from fgnt.mask_estimation import get_IRM
from fgnt.signal_processing import audioread
from fgnt.signal_processing import stft
from fgnt.utils import mkdir_p


def gen_flist_simu(chime_data_dir, stage, ext=False):
    print('gen_flist_simu')
    with open(os.path.join(
            chime_data_dir, 'annotations',
            '{}05_{}.json'.format(stage, 'simu'))) as fid:
        print(fid)
        annotations = json.load(fid)
    if ext:
        isolated_dir = 'isolated_ext'
    else:
        isolated_dir = 'isolated'
    flist = [os.path.join(
            chime_data_dir, 'audio', '16kHz', isolated_dir,
            '{}05_{}_{}'.format(stage, a['environment'].lower(), 'simu'),
            '{}_{}_{}'.format(a['speaker'], a['wsj_name'], a['environment']))
             for a in annotations]
    return flist

def gen_flist(chime_data_dir):
    flist=[]
    for line in  open(os.path.join(chime_data_dir, 'flist.txt'),'r'):
        flist.append(line.split('\n')[0])
    return flist

def gen_flist_real(chime_data_dir, stage):
    print('gen_flist_real')
    with open(os.path.join(
            chime_data_dir, 'annotations',
            '{}05_{}.json'.format(stage, 'real'))) as fid:
        annotations = json.load(fid)
    flist_tuples = [(os.path.join(
            chime_data_dir, 'audio', '16kHz', 'embedded', a['wavfile']),
                     a['start'], a['end'], a['wsj_name']) for a in annotations]
    return flist_tuples


def get_audio_data(file_template, postfix='', ch_range=range(1, 4)):
    audio_data = list()
    #ch_range=[2,5]
    for ch in ch_range:
        audio_data.append(audioread(
                file_template + '.ch{}{}.wav'.format(ch, postfix))[None, :])
    audio_data = np.concatenate(audio_data, axis=0)
    audio_data = audio_data.astype(np.float32)
    return audio_data

def get_audio_data_new(file_template, postfix='', ch_range=range(1, 4)):
    audio_data = list()
    for ch in ch_range:
        audio_data.append(audioread(
                file_template + '.ch{}{}.wav'.format(ch, postfix))[None, :])
    audio_data = np.concatenate(audio_data, axis=0)
    audio_data = audio_data.astype(np.float32)
    return audio_data

def get_audio_data_with_context(embedded_template, t_start, t_end,
                                ch_range=range(1, 7)):
    start_context = max((t_start - 5), 0)
    context_samples = int((t_start - start_context) * 16000)
    audio_data = list()
    for ch in ch_range:
        audio_data.append(audioread(
                embedded_template + '.CH{}.wav'.format(ch),
                offset=start_context, duration=t_end - start_context)[None, :])
    audio_data = np.concatenate(audio_data, axis=0)
    audio_data = audio_data.astype(np.float32)
    return audio_data, context_samples


def prepare_training_data(chime_data_dir, dest_dir):
    for stage in ['tr', 'dt']:
        flist = gen_flist_simu(chime_data_dir, stage, ext=True)
        export_flist = list()
        mkdir_p(os.path.join(dest_dir, stage))
        for f in tqdm.tqdm(flist, desc='Generating data for {}'.format(stage)):
            clean_audio = get_audio_data(f, '.Clean')
            noise_audio = get_audio_data(f, '.Noise')
            X = stft(clean_audio, time_dim=1, size=512, shift=128).transpose((1, 0, 2))
            N = stft(noise_audio, time_dim=1, size=512, shift=128).transpose((1, 0, 2))
            IBM_X, IBM_N = get_IRM(X, N, 1)
            Y_abs = np.abs(X + N)
            export_dict = {
                'IBM_X': IBM_X.astype(np.float32),
                'IBM_N': IBM_N.astype(np.float32),
                'Y_abs': Y_abs.astype(np.float32)
            }
            export_name = os.path.join(dest_dir, stage, f.split('/')[-1])
            with open(export_name, 'wb') as fid:
                pickle.dump(export_dict, fid)
            export_flist.append(os.path.join(stage, f.split('/')[-1]))
        with open(os.path.join(dest_dir, 'flist_{}.json'.format(stage)),
                  'w') as fid:
            json.dump(export_flist, fid, indent=4)
