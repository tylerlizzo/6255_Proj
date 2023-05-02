import argparse
import os, pdb

import numpy as np
import tensorflow as tf
from tqdm import tqdm
import scipy.io as io
from chime_data import gen_flist_simu, \
    gen_flist_real, get_audio_data, get_audio_data_with_context, gen_flist, get_audio_data_new
from fgnt.beamforming import gev_wrapper_on_masks
from fgnt.signal_processing import audiowrite, stft, istft
from fgnt.utils import Timer
from fgnt.utils import mkdir_p
from nn_models3_tf import LSTMMaskEstimator

parser = argparse.ArgumentParser(description='NN GEV beamforming')
#parser.add_argument('flist',
                    #help='Name of the flist to process (e.g. tr05_simu)')
parser.add_argument('chime_dir',
                    help='Base directory of the CHiME challenge.')
parser.add_argument('output_dir',
                    help='The directory where the enhanced wav files will '
                         'be stored.')
parser.add_argument('model',
                    help='Trained model file')
parser.add_argument('--gpu', '-g', default=0, type=int,
                    help='GPU ID (negative value indicates CPU)')
args = parser.parse_args()

# Prepare model
model = LSTMMaskEstimator()
with tf.device('/gpu:'+str(args.gpu)):
    tf_Y, tf_IBM_X, tf_IBM_N = model.model_inputs()
    tf_dropout = tf.placeholder(tf.float32, shape=())
    loss, x_mask, n_mask = model._propagate(tf_Y, tf_IBM_X, tf_IBM_N, dropout=tf_dropout)
    optimizer = tf.train.AdamOptimizer(0.001)
    train_op = optimizer.minimize(loss)
    saver = tf.train.Saver()

tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
tf_config.allow_soft_placement = True
tf_config.log_device_placement = False


flist=gen_flist(args.chime_dir)
t_io = 0
t_net = 0
t_beamform = 0

# Beamform loop
with tf.Session(config=tf_config) as sess:
    saver.restore(sess, args.model)

    for cur_line in tqdm(flist):
        
        with Timer() as t:
        	audio_data = get_audio_data_new(cur_line)
        	context_samples = 0
        t_io += t.msecs
        Y = stft(audio_data, time_dim=1).transpose((1, 0, 2))
        Y_var = np.abs(Y).astype(np.float32)
        feed_dict = {
            tf_Y: Y_var,
            tf_dropout: 1.0
        }
        with Timer() as t:
            X_masks, N_masks = sess.run([x_mask, n_mask], feed_dict=feed_dict)
        t_net += t.msecs

        with Timer() as t:
            Ch123_N_mask = N_masks.data
            Ch123_X_mask = X_masks.data            
            
            N_mask = np.squeeze(N_masks[:,0,:])
            X_mask = np.squeeze(X_masks[:,0,:])

            print(N_mask.shape, X_mask.shape)

            Y_hat = gev_wrapper_on_masks(Y, N_mask, X_mask)


        t_beamform += t.msecs

        env = cur_line.split('/')[-1]
        filename = os.path.join(
                args.output_dir,
                '{}.wav'.format(env)
        )
        matname = os.path.join(
                args.output_dir,
                '{}.mat'.format(env)
        )
        matname_ch123_X = os.path.join(
                args.output_dir,'X_mask',
                '{}.mat'.format(env)
        )
        matname_ch123_N = os.path.join(
                args.output_dir,'N_mask',
                '{}.mat'.format(env)
        )
        with Timer() as t:
            io.savemat(matname,{'lambda_noise':N_mask})

            audiowrite(istft(Y_hat)[context_samples:], filename, 16000, True, True)
        t_io += t.msecs

print('Finished')
print('Timings: I/O: {:.2f}s | Net: {:.2f}s | Beamformer: {:.2f}s'.format(
        t_io / 1000, t_net / 1000, t_beamform / 1000
))