import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.contrib import learn
import pdb
import dataset
import scipy.io as sio

class Net(object):
    def __init__(self, config):
        self.batch_size = 50
        self.sent_dim = 20
        self.output_dim = 100
        self.learning_rate = 0.001
        self.maxiter = 20000
        self.decay_step = 2000
        self.channels = 2
        self.is_train = True
        
        self.signal = tf.placeholder(tf.float32, [None, self.sent_dim, self.channels])
        self.y = tf.placeholder(tf.float32, [None, self.sent_dim, self.channels])

        self.savedir = 'models/lr_'+str(self.learning_rate)+'_batch_'+str(self.batch_size)+'_sent_'+str(self.sent_dim)+'_output_'+str(self.output_dim)+'_iter_'+str(self.maxiter)+'_channels_'+str(self.channels)+'.ckpt'
        
        lstm_cells = []
        lstm1 = tf.nn.rnn_cell.DropoutWrapper(
            tf.nn.rnn_cell.BasicLSTMCell(100, state_is_tuple=False), 0.5)
        lstm_cells.append(lstm1)
        
        lstm2 = tf.nn.rnn_cell.DropoutWrapper(
            tf.nn.rnn_cell.BasicLSTMCell(20, state_is_tuple=False), 0.5)
        lstm2 = tf.contrib.rnn.AttentionCellWrapper(lstm2, 20, state_is_tuple=False)
        lstm_cells.append(lstm2)
        
        stacked_lstm = tf.nn.rnn_cell.MultiRNNCell(lstm_cells, state_is_tuple=False)
        X_ = tf.unpack(self.signal, axis=1)
        self.cell = stacked_lstm
        self.initial_state = self.cell.zero_state(batch_size=self.batch_size, dtype=tf.float32)
        output, self.final_state = tf.nn.rnn(stacked_lstm, X_, initial_state=self.initial_state, dtype=tf.float32)

        out_w = tf.Variable(tf.random_normal([20, self.channels]))
        out_b = tf.Variable(tf.zeros([self.channels]))
        self.predict = tf.pack([tf.add(tf.matmul(out, out_w), out_b) for out in tf.unpack(output, axis=1)], axis=0)
        self.loss = tf.reduce_mean(tf.square(tf.sub(self.predict, self.y)))
        self.global_step = tf.Variable(0, trainable=False)
        self.lr = tf.train.exponential_decay(self.learning_rate, self.global_step, self.decay_step, 0.5, staircase=True)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

        configProto = tf.ConfigProto()
        configProto.gpu_options.allow_growth = True
        configProto.allow_soft_placement = True
        self.sess = tf.Session(config=configProto)
        self.saver = tf.train.Saver()
        return

    def load_model(self):
        if self.is_train:
            self.sess.run(tf.initialize_all_variables())
        else:
            self.saver.restore(self.sess, self.savedir)

    def save_model(self):
        if not self.is_train:
            pass
        else:
            self.saver.save(self.sess, self.savedir)

    def train(self, train_set):
        self.is_train = True
        self.load_model()
        state = self.sess.run(self.initial_state)
        for _ in xrange(self.maxiter):
            X, y = train_set.next_batch(self.batch_size)
            __, loss, state = self.sess.run([self.optimizer, self.loss, self.final_state], feed_dict={
                self.signal: X,
                self.y: y,
                self.initial_state: state,
            })
            print("loss = %.4f"%loss)
        self.save_model()

    def generate(self, length):
        self.is_train = False
        self.load_model()
        hist = np.zeros([1, 1, self.channels])
        wave = np.zeros([length, self.channels])
        savedir = 'models/lr_'+str(self.learning_rate)+'_batch_'+str(self.batch_size)+'_sent_'+str(self.sent_dim)+'_output_'+str(self.output_dim)+'_iter_'+str(self.maxiter)+'_channels_'+str(self.channels)+'_length_'+str(length)+'.npy'
        pdb.set_trace()
        prev_state = self.sess.run(self.cell.zero_state(1, tf.float32))
        for i in xrange(length):
            output, prev_state = self.sess.run([self.predict, self.final_state], feed_dict={
                self.signal: hist,
                self.initial_state: prev_state,
            })
            wave[i, :] = output
            hist = np.expand_dims(output, 0)
        np.save(savedir, wave)
        return wave

net = Net({})
#net.train(dataset.train())
net.generate(44100*10)
