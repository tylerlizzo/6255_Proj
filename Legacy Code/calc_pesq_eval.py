# coding:utf-8
'''
by Jiangwenbin, 0171115
'''
import os
import argparse
import json
from multiprocessing import Pool 
import numpy as np
from tqdm import tqdm

def gen_flist(chime_data_dir, enhance_dir, stage='dt', mode='real'):
    #stage:'dt', 'et'; mode:'simu', 'real'
    #print(mode+stage+enhance_dir+chime_data_dir)
    with open(os.path.join(
            chime_data_dir, 'annotations',
            '{}05_{}.json'.format(stage, mode))) as fid:
        annotations = json.load(fid)
    flist = [os.path.join(enhance_dir, 
            '{}05_{}_{}'.format(stage, a['environment'].lower(), mode),
            '{}_{}_{}'.format(a['speaker'], a['wsj_name'], a['environment']))
             for a in annotations]
    return flist

def gen_flist_1ch(chime_data_dir, stage='dt', mode='real'):
    #stage:'dt', 'et'; mode:'simu', 'real'    
    with open(os.path.join(
            chime_data_dir, 'annotations',
            '{}05_{}_1ch_track.list'.format(stage, mode))) as fid:
        lines = fid.readlines()    
    flist = [os.path.join(chime_data_dir, 'audio', '16kHz', 'isolated_1ch_track',
            '{}'.format(l.split('.')[0]))
            for l in lines]
    return flist

def gen_flist_bth(chime_data_dir, stage='dt'):
    with open(os.path.join(
            chime_data_dir, 'annotations',
            '{}05_{}.json'.format(stage, 'bth'))) as fid:
        annotations = json.load(fid)
    flist = [os.path.join(
            chime_data_dir, 'audio', '16kHz', 'isolated',
            '{}05_{}'.format(stage, a['environment'].lower()),
            '{}_{}_{}'.format(a['speaker'], a['wsj_name'], a['environment']))
             for a in annotations]
    return flist

def pesq(deg_file):
    PESQ_bin = '/home/jiangwenbin/workspace/ITU-T_pesq/bin/pesq'
    ref_path = '/home/zlc/evaluation/'
    #file_name = deg_file.split('/')[-1][:-4]
    #file_name_env = deg_file.split('/')[-1]
    #env_str = deg_file.split('/')[-2]
    #stage_str = env_str.split('_')[0]#et05, dt05
    #mode = env_str.split('_')[-1]#simu, real
    #if mode == 'simu':
        #ref_file = '{}/{}_bth/{}_BTH.CH0.wav'.format(ref_path, stage_str, file_name)
    #else:
        #ref_file = '{}/{}/{}.CH0.wav'.format(ref_path, env_str, file_name_env)
    #deg_file = '{}.wav'.format(deg_file)
    ref_file=ref_path+deg_file.split('/')[-1]+'.ref.wav'
    deg_file='{}.wav'.format(deg_file)
    cmd = '{} +16000 {} {} > /dev/null'.format(PESQ_bin, ref_file, deg_file)
    # print(cmd)
    os.system(cmd)

import shutil
def calc_pesq(enhance='none', stage='et', mode='real'):
    #enhance:method; stage:dt,et; mode: simu,real
    #chime_data_dir = '/workspace/CHiME3/data'
    '''
    if enhance in ('BLSTM32', ):
        #enhance_path = '/workspace/wenfei/nn-gev/beam_out'
        enhance_path = '/workspace/wenfei/nn-gev-32/beam_out'
    elif enhance in ('BLSTM32_2', ):
        enhance_path = '/workspace/wenfei/nn-gev-32/beam_out2'
    elif enhance in ('BLSTM32_3', ):
        enhance_path = '/workspace/wenfei/nn-gev-32/beam_out3' 
    elif enhance in ('LSTM32_3', ):
        enhance_path = '/workspace/wenfei/nn-gev-32/beam_out_lstm'
    else:
        enhance_path = ''
    '''

    # = '/workspace/wenfei/nn-gev-32/beam_out_lstm_mvdr_48_6'
    #if enhance == 'none':
    #    flist = gen_flist_1ch(chime_data_dir, stage, mode)
    #else:
    #    enhance_dir = '{}/{}'.format(enhance_path, enhance)
    #    flist = gen_flist(chime_data_dir, enhance_dir, stage, mode)
    #flist = gen_flist(chime_data_dir, enhance_path, stage, mode)
    flist=[]
    for line in open('/home/zlc/tf_test/flist/flist_HS-5.txt','r'):
        #flist.append(line.split('\n')[0])
        #flist.append('/home/zlc/tf_test/beam_out_lstm_gev_sam_16k/'+line.split('\n')[0].split('/')[-1])
        flist.append('/home/djy/nn-gev-3ch/enhan_data/'+line.split('\n')[0])

    print('total: {}'.format(len(flist)))
    flist_0 = flist[0]+'.wav'
    if os.path.isfile(flist_0):
        print('test file OK!')
    else:
        print('test file:' + flist_0 + ' not exist!')
        return
    #if os.path.isfile('pesq_results.txt'):
    #    os.rename('pesq_results.txt', 'pesq_results.bak.txt')
    #print(flist)
    pool = Pool(16)
    pool.map(pesq, flist)
    pool.close()
    pool.join()
    #os.mknod('./results/{}05_{}_{}_pesq.txt'.format(stage, mode, enhance))
    if os.path.isfile('pesq_results.txt'):
        print('write ok')
        shutil.move('pesq_results.txt', './results/{}05_{}_{}_pesq.txt'.format(stage, mode, enhance))
        print('move ok')
        #print('pesq_results.txt not exist')

def analysis_pesq(enhance, stage='dt', mode='simu'):
    score_list = []
    pesq_score_file = './results/{}05_{}_{}_pesq.txt'.format(stage, mode, enhance)
    with open(pesq_score_file, 'r') as fid:
        fid.readline() #first line
        for line in fid:
            score = float(line.split('\t')[2])            
            score_list.append(score)
    return np.mean(score_list)

if __name__ == '__main__':
    #stage_list = ('dt', 'et')
    #mode_list = ('simu', 'real')
    #enhance_list = ('CGMM','none')
    # parser = argparse.ArgumentParser(description='PESQ')
    # parser.add_argument('stage',
    #                 help='Base directory of the CHiME challenge.')
    # args = parser.parse_args()
    # stage = args.stage
    stage_list = ('MV128new_HF-5_',)
    mode_list = ('real',)
    enhance_list = ('LSTM',)
    for enhance in enhance_list: 
        for stage in stage_list:
            for mode in mode_list:
                print(enhance)
                calc_pesq(enhance, stage, mode)
                print('{}05_{}_{}_pesq: {:.3f}'.format(stage, mode, enhance, analysis_pesq(enhance, stage, mode)))
        print('')
'''
dt05_simu_none_pesq: 2.011
dt05_real_none_pesq: 2.167
et05_simu_none_pesq: 1.978
et05_real_none_pesq: 2.318

dt05_simu_beamformit_5mics_pesq: 2.30
dt05_real_beamformit_5mics_pesq: 2.403
et05_simu_beamformit_5mics_pesq: 2.197
et05_real_beamformit_5mics_pesq: 2.493

dt05_simu_beamform_cgmm_pesq: 2.577
dt05_real_beamform_cgmm_pesq: 2.581
et05_simu_beamform_cgmm_pesq: 2.577
et05_real_beamform_cgmm_pesq: 2.594

dt05_simu_nn_gev_FW_pesq: 2.389
dt05_real_nn_gev_FW_pesq: 2.722
et05_simu_nn_gev_FW_pesq: 2.404
et05_real_nn_gev_FW_pesq: 2.718

dt05_simu_NN_IRM_pesq: 2.60
dt05_real_NN_IRM_pesq: 2.870
et05_simu_NN_IRM_pesq: 2.641
et05_real_NN_IRM_pesq: 2.765

'''
