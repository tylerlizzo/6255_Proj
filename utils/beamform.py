import numpy as np
from scipy.fftpack import fft, ifft
from scipy import signal as sg

def corr_matrix(mc_data,ang_vec,Fs,fft_len):
    mc_data = mc_data.T
    freq_vec = np.linspace(0,int(Fs/2),fft_len//2)
    start_idx = 0
    end_idx = fft_len

    num_samps, num_channels = np.shape(mc_data)
    mat = np.zeros((num_channels,num_channels,fft_len//2),dtype=np.complex64)

    num_frames = 0
    
    for i in range(0,10):
        mc_data_samp = mc_data[start_idx:end_idx,:]
        samp_comp = fft(mc_data_samp, n=fft_len,axis=0)
        for idx in range(0,len(freq_vec)):
            mat[:,:,idx] += np.multiply.outer(samp_comp[idx,:],np.conj(samp_comp[idx,:]).T)
        num_frames += 1
        start_idx += fft_len//2
        end_idx += fft_len//2
        if num_samps <= end_idx:
            num_frames -= 1
            break

    end_idx = num_samps
    start_idx = end_idx - fft_len

    for i in range(0,10):
        mc_data_samp = mc_data[start_idx:end_idx,:]
        samp_comp = fft(mc_data_samp, n=fft_len,axis=0)
        for idx in range(0,len(freq_vec)):
            mat[:,:,idx] += np.multiply.outer(samp_comp[idx,:],np.conj(samp_comp[idx,:]).T)
        num_frames += 1
        start_idx -= fft_len//2
        end_idx -= fft_len//2
        if start_idx < 1:
            num_frames -= 1
            break
        
    return mat/num_frames

def MVDR(mat,steer_vec,ang_vec,Fs,fft_len):
    num_channels = len(ang_vec)
    freq_vec = np.linspace(0,int(Fs/2),fft_len//2)
    mvdr = np.ones((num_channels,fft_len//2),dtype=np.complex64)
    for idx in range(0,fft_len//2):
        samp_mat = np.reshape(mat[:,:,idx],[num_channels,num_channels])
        inv_mat = np.linalg.pinv(samp_mat)
        temp = np.matmul(np.conjugate(steer_vec[:,idx]).T,inv_mat)
        temp = np.matmul(temp,steer_vec[:,idx])
        temp = np.reshape(temp,[1,1])
        mvdr[:,idx] = np.matmul(inv_mat, steer_vec[:,idx]) / temp
    return mvdr
    
def beamform(beamform,mc_spectrums,Fs,fft_len):
    num_channels, num_frames, num_bins = np.shape(mc_spectrums)
    enhanced_spec = np.zeros((num_frames,num_bins),dtype=np.complex64)
    for idx in range(0,num_bins-1):
        enhanced_spec[:,idx] = np.matmul(np.conjugate(beamform[:,idx]).T, mc_spectrums[:,:,idx])

    hanning = sg.windows.hann(fft_len+1,'periodic')[: -1]
    cut_data = np.zeros(fft_len,dtype=np.complex64)
    result = np.zeros(Fs * 300, dtype =np.float32)
    start_idx = 0
    end_idx = fft_len
    for i in range(0,num_frames):
        half_spec = enhanced_spec[i,:]
        cut_data[0:fft_len//2+1] = half_spec.T
        cut_data[fft_len//2+1:] = np.flip(np.conjugate(half_spec[1:fft_len//2]),axis=0)
        temp = np.real(ifft(cut_data, n=fft_len))
        result[start_idx:end_idx] += np.real(temp * hanning.T)
        start_idx += fft_len//2
        end_idx += fft_len//2
    return result[0:end_idx - fft_len//2]
