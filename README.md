
# Group 16: Multiple-Channel Speech Enhancement

## Abstract

## Code Organization
### Legacy Code
This directory contains some work by Joy on a previous project. We used some of this code as a starting point that we built upon. Her original code came from her undergraduate thesis, and builds upon this Github: https://github.com/fgnt/nn-gev.

### Data Directory
This directory contains one sample multiple channel utterance. It also contains the demo generation audio and output for our synthetic multiple channel data generation script. It does not contain any of the CHiME-4 dataset that we ran on.

### Utils Directory
This directory contains utility function relating to multiple channel data and beamforming

#### beamform.py
This file has three main functions for implementing a beamformer:
* `corr_matrix(mc_data,ang_vec,Fs,fft_len)` generates a correlation matrix between the multiple channel data and the ideal direction signal.
* `MVDR(mat,steer_vec,ang_vec,Fs,fft_len)` generates an MVDR beamformer based off the inputs. This is based off the following paper: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=1614.
* `beamform(beamform,mc_spectrums,Fs,fft_len)` applies the provided beamformer to the spectrums of the multi-channel data.

#### generate_mc_data.py
This file takes in sound and noise files (and direction angle) and generates synthetic multi-channel data.
`python generate_mc_data.py -sf /path/to/sound/file -nf /path/to/noise/file -dir 30 -out /path/to/output/directory`

#### utils.py
* `multichannel_load(directory,num_channels)` loads the multi-channel data from disc to np arrays.
* `multichannel_spectrum(mc_data,fft_len)` takes FFTs for all of the multiple channel data.
* `steering_vector(direct,Fs,fft_len,ang_vec,mic_data)` calculates the steering vector for the beamformer.

### Wiener Filtering Directory
This directory contains MATLAB files related to our wiener filtering and spectral subtraction.
* `denoise2(y,fs,noiseLengthSec,nfft,noverlap)` implements spectral subtraction on a passed-in signal.
* `wienerFilter(y,n,nfft,noverlap,fs)` implements Wiener filtering on a passed-in signal.
