import tensorflow as tf
from fgnt import NormLSTMCell
import pdb

class LSTMMaskEstimator(object):

    def model_inputs(self):
        Y = tf.placeholder(tf.float32, shape=(None,2,257))
        IBM_X = tf.placeholder(tf.float32, shape=(None,2,257))
        IBM_N = tf.placeholder(tf.float32, shape=(None,2,257))
        return Y, IBM_X, IBM_N

    def _propagate(self, Y, IBM_X=0., IBM_N=0., dropout=1.):

        input = tf.nn.dropout(Y, keep_prob=dropout)
        input = tf.reshape(input,[-1,257])
        cell_in = tf.contrib.layers.fully_connected(input, 4*128, activation_fn=None)
        cell_in = tf.contrib.layers.batch_norm(cell_in, scale=True)
        cell_in = tf.reshape(cell_in,[-1,2,4*128])

        lstm = NormLSTMCell.LSTMCell(128, dropout=dropout)
        output, state = tf.nn.dynamic_rnn(lstm, cell_in, time_major=True, dtype=tf.float32)
        output = tf.reshape(output,[-1,128])

        drop0 = tf.nn.dropout(output, keep_prob=dropout)
        fc1 = tf.contrib.layers.fully_connected(drop0, 128, activation_fn=None)
        norm1 = tf.contrib.layers.batch_norm(fc1, scale=True)
        relu1 = tf.nn.relu6(norm1)

        drop1 = tf.nn.dropout(relu1, keep_prob=dropout)
        fc2 = tf.contrib.layers.fully_connected(drop1, 128, activation_fn=None)
        norm2 = tf.contrib.layers.batch_norm(fc2, scale=True)
        relu2 = tf.nn.relu6(norm2)

        n_fc = tf.contrib.layers.fully_connected(relu2, 257, activation_fn=None)
        n_norm = tf.contrib.layers.batch_norm(n_fc, scale=True)
        n_mask = tf.nn.sigmoid(tf.reshape(n_norm,[-1,2,257]))

        x_fc = tf.contrib.layers.fully_connected(relu2, 257, activation_fn=None)
        x_norm = tf.contrib.layers.batch_norm(x_fc, scale=True)
        x_mask = tf.nn.sigmoid(tf.reshape(x_norm,[-1,2,257]))

        loss_X = tf.reduce_mean(tf.pow((IBM_X - x_mask),2))
        loss_N = tf.reduce_mean(tf.pow((IBM_N - n_mask),2))
        loss = (loss_X + loss_N)/2

        return loss, x_mask, n_mask
