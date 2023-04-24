import sys
import numpy as np
import soundfile as sf
import argparse as ap

from scipy.signal import resample

##python .\generate_mc_data.py -sf ..\data\Demo_Raw\clean_audio.mp3 -dir 30 -out ..\data\Demo_Output\

def snr(sound, noise):

    sound_rss = np.sqrt(np.sum(np.square(sound)))
    noise_rss = np.sqrt(np.sum(np.square(noise)))
    ratio = sound_rss/noise_rss
    db = 20*np.log10(ratio)

    return db

parser = ap.ArgumentParser(
    prog='Generate Multichannel Data',
    description = 'This program takes single channel data and generates multi-channel data')

parser.add_argument('-sf', '--soundfile',required=True)
parser.add_argument('-nf', '--noisefile')
parser.add_argument('-dir', '--direction',required=True)
parser.add_argument('-out', '--output_directory',required=True)

args = parser.parse_args()

sound = sf.read(args.soundfile)
fs = sound[1]
sound = sound[0]
num_samps = len(sound)
sound = sound/np.max(sound)


if args.noisefile is None:
    noise = np.random.normal(size=num_samps)
else:
    noise = sf.read(args.noisefile)
    noise = resample(noise[0],round(len(noise[0])*noise[1]/fs))
    noise = np.array(noise)
    noise = noise[124000:124000+num_samps,0]
    noise = noise/np.max(noise)

db = snr(sound,noise)

desired_db = 2

A = (0.1)**((desired_db-db)/10)


angles = np.array([0, 60, 120, 180, 270, 330])
for i in range(0,6):
    temp = A*noise + sound * np.cos(np.pi*(angles[i]-int(args.direction))/180)
    file = args.output_directory + "data_file.CH" + str(i+1) + ".wav"
    sf.write(file,temp,samplerate=fs)
