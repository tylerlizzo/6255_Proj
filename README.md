
# Group 16: Multiple-Channel Speech Enhancement

## Abstract

## Code Organization
### Data Directory
This directory contains one sample multiple channel utterance. It also contains the demo generation audio and output for our synthetic multiple channel data generation script. It does not contain any of the CHiME-4 dataset that we ran on.

### Utils Directory
#### beamform.py
This file has three main functions for implementing a beamformer:
* `corr_matrix(mc_data,ang_vec,Fs,fft_len)` generates a correlation matrix between the multiple channel data and the ideal direction signal.
* `MVDR(mat,steer_vec,ang_vec,Fs,fft_len)` generates an MVDR beamformer based off the inputs.
* `beamform(beamform,mc_spectrums,Fs,fft_len)` applies the provided beamformer to the spectrums of the multi-channel data.

#### generate_mc_data.py
This file takes in sound and noise files (and direction angle) and generates synthetic multi-channel data.
`python generate_mc_data.py -sf /path/to/sound/file -nf /path/to/noise/file -dir 30 -out /path/to/output/directory`

#### utils.py
* `multichannel_load(directory,num_channels)` loads the multi-channel data from disc to np arrays.
* `multichannel_spectrum(mc_data,fft_len)` takes FFTs for all of the multiple channel data.
* `steering_vector(direct,Fs,fft_len,ang_vec,mic_data)` calculates the steering vector for the beamformer.
