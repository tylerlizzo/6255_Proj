
# Group 16: Multiple-Channel Speech Enhancement

## Abstract

For our final project, we will look at time-varying information in online decision making. This semester, we have focused on decision classes that have consistent selections during the period of evaluation; these classes use consistent algorithms or a probabilistic distribution for the reward/loss. Unfortunately, real data is always changing.

We will look at time-varying inforamation in two main settings: perfection posterior information and a multi-armed bandit setting where we only ahve the information from the decision calsses we choose. These situations are both incredibly common in real-world situations.

For this project, we will perform a literature review of what has already been done for time-varying information in these two settings, and we will perform an evaluation of different algorithms for these two settings. We will look at both the algorithms that were covered in class as well as a few novel algorithms from our literature review.



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
