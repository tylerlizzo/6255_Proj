import numpy as np
import soundfile as sf
from scipy.fftpack import fft
import scipy.constants as sci_c

def spectrogram(signal,Fs):
    f,t,sxx = signal.spectrogram(x,Fs,nfft=8192)
    plt.pcolormesh(t, f, sxx)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.ylim([0,1000])
    plt.show()
    return

def multichannel_load(directory,num_channels):
    L = len(sf.read(directory.replace('[]','1'))[0])
    wavs = np.zeros((num_channels,L))
    for i in range(0,num_channels):
        wavs[i,:] = sf.read(directory.replace('[]',str(i+1)))[0]
    return wavs

def multichannel_spectrum(mc_data,fft_len):
    shift = fft_len//2

    num_channels, L = np.shape(mc_data)

    mc_data = mc_data/np.max(np.abs(mc_data)) * 0.7

    num_frames = (L - fft_len)//shift

    mc_spectrums = np.zeros((num_channels,num_frames,(fft_len//2)+1),dtype=np.complex64)

    for i in range(0,num_frames):
        temp = fft(mc_data[:, 0+i*shift:fft_len+i*shift],n=fft_len,axis=1)[:,0:fft_len//2+1]
        mc_spectrums[:,i,:] = temp
    return mc_spectrums

def steering_vector(direct,Fs,fft_len,ang_vec,mic_dia):
    num_channels = len(ang_vec)
    freq_vec = np.linspace(0,Fs,fft_len)
    steer_vec = np.ones((len(freq_vec),num_channels), dtype=np.complex64)
    direct = direct * (-1)
    for f, freq in enumerate(freq_vec):
        for a,ang in enumerate(ang_vec):
            steer_vec[f,a] = complex(np.exp((-1j)*((2*np.pi*freq) / sci_c.mach) \
                                               * (mic_dia / 2) \
                                               * np.cos( (direct-ang) * np.pi/180)))
    steer_vec = np.conjugate(steer_vec).T
    for idx in range(0,fft_len):
        weight = np.matmul(np.conjugate(steer_vec[:,idx]).T,steer_vec[:,idx])
        steer_vec[:,idx] /= weight
    return steer_vec

