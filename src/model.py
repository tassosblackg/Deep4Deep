# ASV --anti spoofing project 2018--based on papers
# that's where you implement your model's graph and execution functionality
# for feature maps 17x64 we can get up to 10 layers CNN(V.DCNN)
# we should try first the baseline CNN ?? is that implemented in the given matlab code??
# So we only design our V.D.CNN

# Bonus: ->batch normalization


from __future__ import division, print_function

import sys
import time
from datetime import datetime
import math
import tensorflow as tf
import numpy as np
from lib.model_io import save_variables
from lib.precision import _FLOATX

# ------define architecture functions--------------------------------------------------------------
# define weights


def weight_dict(shape):
    init = tf.truncated_normal(shape, stddev=0.1)
    return(tf.Variable(init))
# define bias--optional


def bias_dict(shape):
    init = tf.constant(0.1, shape=shape)
    return(tf.Variable(init))

# return convolution result --optional add bias


def conv2d(x, W, stride=1):
    return(tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='SAME')+bias_dict(shape))
# define convolution layer


def conv_layer(inp, shape):
    W = weight_dict(shape)
    b = bias_dict([shape[3]])
    return(tf.nn.relu(conv2d(inp, W)+b))
# define max pooling function


def max_pool(x, stride, k):
    return (tf.nn.max_pool(x, strides=[1, 2, stride, 1], ksize=[1, 2, k, 1], padding='VALID'))

# define dense layers--last block of layers


def dense_layer(inp, size, loss):
    in_s = int(inp.getshape()[1])  # flatten layer
    W = weight_dict([in_s, size])
    b = bias_dict(shape)
    # loss+=
    return (tf.matmul(inp, W), loss)
# batch normalization


def batch_n(convl):
    return (tf.nn.relu(tf.contrib.layers.batch_norm(convl)))
# -------------------------------------------------------------------------------------------------------


class CNN(object):

    def __init__(self, model_id=None):
        self.model_id = model_id
        self.train_list = []
        self.valid_list = []
        self.batch_size = 256  # 64 || 128 || 256
        self.train_size = 1587420  # number of files
        self.dev_size = 1029721  # number of files in dev/
        sef.eval_size = 8522944

    def inference(self, X, reuse=True, is_training=True):
        with tf.variable_scope("inference", reuse=reuse):
            # Implement your network here
            # equation or predefiend fuctions --convolution operation
            # we define set of layers according to the max_pooling ksize
            # each set has more than one convolution and max_poolin layers
            # totaly we have 2 sets and 5 blocks
            # init phase
            shape1 = [3, 3, 1, 4]  # [filter_h,filter_w,in_channel,out_channel]
            w = weight_dict(shape1)
            b = bias_dict([shape1[3]])
            conv1 = conv2d(X, w) + b  # init_convolution

    # -----------1st set--------{2 blocks}---------------------------------------
            # -------1st block
            shape1 = [3, 3, 1, 8]
            conv_l1 = conv_layer(conv1, shape1)
            batch_norm1 = batch_n(conv_l1)  # normalization
            conv_l2 = conv_layer(batch_norm1, shape1)
            batch_norm2 = batch_n(conv_l2)  # normalization batch

            mpool_1 = max_pool(batch_norm2, 1, 1)  # stride =1 , k=1
            # ------2nd block
            shape2 = [3, 3, 1, 8]
            conv_l3 = conv_layer(mpool_1, shape2)
            batch_norm3 = batch_n(conv_l3)  # normalization
            conv_l4 = conv_layer(batch_norm3, shape2)
            batch_norm4 = batch_n(conv_l4)  # normalization batch

            mpool_2 = max_pool(batch_norm4, 1, 1)  # stride =1 , k=1

    # --------2nd set------{3 blocks}--------------------------------------------
            # -------3d block
            shape3 = [3, 3, 1, 16]
            conv_l5 = conv_layer(mpool_2, shape3)
            batch_norm5 = batch_n(conv_l5)  # normalization
            conv_l6 = conv_layer(batch_norm1, shape3)
            batch_norm6 = batch_n(conv_l6)  # normalization batch

            mpool_3 = max_pool(batch_norm6, 1, 1)  # stride =1 , k=1
            # --------4th block
            shape4 = [3, 3, 1, 32]
            conv_l7 = conv_layer(mpool_3, shape4)
            batch_norm7 = batch_n(conv_l7)
            conv_l8 = conv_layer(batch_norm7, shape4)
            batch_norm8 = batch_n(conv_l8)

            mpool_4 = max_pool(batch_norm8, 2, 2)  # stride=2, k=2
            # --------5th blocks
            shape5 = [3, 3, 1, 64]
            conv_l9 = conv_layer(mpool_4, shape5)
            batch_norm9 = batch_n(conv_l9)
            conv_l10 = conv_layer(batch_norm9, shape5)
            batch_norm10 = batch_n(conv_l10)

            mpool_5 = max_pool(batch_norm10, 2, 2)  # stride=2, k=2

    # ------------add dense layers {4 layers}-------------------------------------

        return Y

    def define_train_operations(self):

        # --- Train computations
        self.trainDataReader = trainDataReader
        # shaping variables--
        # __________________________________________________
        height = 64
        width = 17
        chan = 1  # channel of image 1 or 3 if rgb
        n_classes = 2  # genuine or spoof --number of classes
        # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

        X_data_train = tf.placeholder(tf.float32, shape=(
            None, height, width, chan))  # Define this

        Y_data_train = tf.placeholder(
            tf.int32, shape=(None, n_classes))  # Define this

        Y_net_train = self.inference(
            X_data_train, reuse=False)  # Network prediction

        # Loss of train data
        self.train_loss = tf.reduce_mean(tf.nn.sparce_softmax_cross_entropy_with_logits(
            labels=Y_data_train, logits=Y_net_train, name='train_loss'))

        # define learning rate decay method
        global_step = tf.Variable(0, trainable=False, name='global_step')
        learning_rate = 0.01  # Define it--play with this

        # define the optimization algorithm
        # Define it --shall we try different type of optimizers
        optimizer = tf.train.AdamOptimizer(learning_rate)

        trainable = tf.trainable_variables()  # may be the weights??
        self.update_ops = optimizer.minimize(
            self.train_loss, var_list=trainable, global_step=global_step)

        # --- Validation computations
        X_data_valid = tf.placeholder(tf.float32, shape=(
            None, height, width, chan))  # Define this
        Y_data_valid = tf.placeholder(
            tf.int32, shape=(None, n_classes))  # Define this

        Y_net_valid = self.inference(
            X_data_valid, reuse=True)  # Network prediction

        # Loss of validation data
        self.valid_loss = tf.reduce_mean(tf.nn.sparce_softmax_cross_entropy_with_logits(
            labels=Y_data_valid, logits=Y_net_valid, name='valid_loss'))

    def train_epoch(self, sess):
        train_loss = 0
        total_batches = 0
        n_elemnt = self.train_size/self.batch_size  # ??
        while (total_batches <= n_elemnt):  # loop through train batches:
            mean_loss, _ = sess.run([self.train_loss, self.update_ops], feed_dict={X_train: , Y_train: })
            if math.isnan(mean_loss):
                print('train cost is NaN')
                break
            train_loss += mean_loss
            total_batches += 1

        if total_samples > 0:
            train_loss /= total_batches

        return train_loss

    def valid_epoch(self, sess):
        valid_loss = 0
        total_batches = 0
        n_elmnts = self.dev_size/self.batch_size  # number of elements
        while (total_batches < n_elmnts):  # Loop through valid batches:
            mean_loss = sess.run(self.valid_loss, feed_dict={X_val:, Y_val: })
            if math.isnan(mean_loss):
                print('valid cost is NaN')
                break
            valid_loss += mean_loss
            total_batches += 1

        if total_samples > 0:
            valid_loss /= total_batches

        return valid_loss

    def train(self, sess):
        start_time = time.clock()

        n_early_stop_epochs =  # Define it
        n_epochs = 50  # Define it

        saver = tf.train.Saver(
            var_list=tf.trainable_variables(), max_to_keep=4)

        early_stop_counter = 0

        init_op = tf.group(tf.global_variables_initializer())

        sess.run(init_op)

        min_valid_loss = sys.float_info.max
        epoch = 0
        while (epoch < n_epochs):
            epoch += 1
            epoch_start_time = time.clock()

            train_loss = self.train_epoch(sess)
            valid_loss = self.valid_epoch(sess)

            epoch_end_time = time.clock()

            info_str = 'Epoch=' + \
                str(epoch) + ', Train: ' + str(train_loss) + ', Valid: '
            info_str += str(valid_loss) + ', Time=' + \
                str(epoch_end_time - epoch_start_time)
            print(info_str)

            if valid_loss < min_valid_loss:
                print('Best epoch=' + str(epoch))
                save_variables(sess, saver, epoch, self.model_id)
                min_valid_loss = valid_loss
                early_stop_counter = 0
            else:
                early_stop_counter += 1

            if early_stop_counter > n_early_stop_epochs:
                # too many consecutive epochs without surpassing the best model
                print('stopping early')
                break

        end_time = time.clock()
        print('Total time = ' + str(end_time - start_time))

    def define_predict_operations(self):
        self.X_data_test_placeholder = tf.placeholder(....)

        self.Y_net_test = self.inference(
            self.X_data_test_placeholder, reuse=False)

    def predict_utterance(self, sess, testDataReader, dataWriter):
        # Define it
