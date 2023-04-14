import numpy as np
import soundfile as sf
from utils import util as ut
from utils import beamform as bf

direct = 0
Fs = 16000
fft_len = 512
ang_vec = np.array([0, 60, 120, 180, 270, 330])
mic_dia = 0.1

mc_data = ut.multichannel_load('./data/20G_20GO010I_STR.CH[].wav',6)

mc_spec = ut.multichannel_spectrum(mc_data,512)

steer_vec = ut.steering_vector(direct,Fs,fft_len,ang_vec,mic_dia)

corr_mat = bf.corr_matrix(mc_data,ang_vec,Fs,fft_len)

mvdr = bf.MVDR(corr_mat,steer_vec,ang_vec,Fs,fft_len)

signal = bf.beamform(mvdr,mc_spec,Fs,fft_len)

sf.write('enhanced.wav',signal/np.max(np.abs(signal))*.7,Fs)
