from __future__ import division, print_function
import model_func as mf

import sys
import time
from datetime import datetime
import math
import tensorflow as tf
import numpy as np
from lib.model_io import save_variables
from lib.model_io import restore_variables
from lib.precision import _FLOATX
import read_img as rim



class CNN(object):

    # initialize class object
    def __init__(self,model_id=None):
        self.model_id = model_id
        # variables to store input data
        self.Xtrain_in = np.empty(0)
        self.Ytrain_in = np.empty(0)
        self.Xvalid_in = np.empty(0)
        self.Yvalid_in = np.empty(0)

        # model variables
        self.height = 64
        self.width = 17
        self.chan = 1           # channel of image 1 or 3 if rgb
        self.n_classes = 2      # genuine or spoof --number of classes
        self.batch_size = 256   # 64 || 128 || 256
        self.train_size = 0
        self.valid_size = 0
        self.eval_size = 0

        # model variables --placeholders
        # self.X_train = None
        # self.Y_train = None
        # self.X_valid = None
        # self.Y_valid = None
        # self.train_loss = 0
        # self.keep_prob = None
        # self.valid_loss = None

    def model_architecture(self, X, keep_prob,reuse=True,is_training=True):
        with tf.variable_scope("model_architecture", reuse=reuse):



            pass

        # return out_layer

    # last layer of network is a softmax output
    def inference(self,X,keep_prob):
        # return mf.outp_layer(X,self.n_classes)
        return tf.nn.softmax(self.model_architecture(X,keep_prob,reuse=True,is_training=False))

    def define_train_operations(self):
        # set placeholders
        self.keep_prob = tf.placeholder(dtype=tf.float32,name='keep_prob')

        self.X_train   = tf.placeholder(dtype=tf.float32, shape=(None,self.height,self.width,self.chan),name='X_train')

        self.Y_train   = tf.placeholder(dtype=tf.int32,shape=(None, ), name='Y_train')

        # network prediction
        Y_train_predict = self.model_architecture(self.X_train,self.keep_prob,reuse=False)
        # calculate training loss between real label and predicted
        self.train_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=Y_train_predict, labels=self.Y_train,name='train_loss'))

        # define learning rate decay method
        global_step = tf.Variable(0, trainable=False, name='global_step')
        # Define it--play with this
        learning_rate = 0.001

        # define the optimization algorithm
        # Define it --shall we try different type of optimizers
        optimizer = tf.train.AdamOptimizer(learning_rate)

        trainable = tf.trainable_variables()  # may be the weights??
        self.update_ops = optimizer.minimize(self.train_loss, var_list=trainable, global_step=global_step)

        # --- Validation computations
        self.X_valid = tf.placeholder(dtype=tf.float32, shape=(None, self.height, self.width, self.chan))  # Define this
        self.Y_valid = tf.placeholder(dtype=tf.int32, shape=(None, ))  # Define this

        Y_valid_predict = self.model_architecture(self.X_valid,self.keep_prob,reuse=True)

        # Loss on validation
        self.valid_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=Y_valid_predict, labels=self.Y_valid,name='valid_loss'))

    # define train actions per epoch
    def train_epoch(self,sess):
        print("Train_epoch")
        train_loss = 0
        total_batches = 0
        # print("TOTAL="+str(self.train_size))
        n_batches = self.train_size / self.batch_size  # ??
        indx=0
        X,Y=mf.shuffling(self.Xtrain_in,self.Ytrain_in)                   # shuffle X ,Y data
        Xbatch,Ybatch,indx=mf.read_nxt_batch(X,Y,self.batch_size,indx)    # take the right batch

        while Xbatch is not None:     # loop through train batches:
            mean_loss, _ = sess.run([self.train_loss, self.update_ops], feed_dict={self.X_train: Xbatch ,self.Y_train: Ybatch,self.keep_prob:0.3})
            Xbatch,Ybatch,indx=mf.read_nxt_batch(X,Y,self.batch_size,indx)
            if math.isnan(mean_loss):
                print('train cost is NaN')
                break
            train_loss += mean_loss
            total_batches += 1

        if total_batches > 0:
            train_loss /= total_batches

        return train_loss

    # validation actions per epoch
    def valid_epoch(self,sess):
        print("Valid_epoch")
        valid_loss = 0
        total_batches = 0
        n_batches = self.dev_size / self.batch_size  # number of elements
        indx=0
        X,Y=mf.shuffling(self.Xvalid_in,self.Yvalid_in)  # shuffle X ,Y data
        Xbatch,Ybatch,indx=mf.read_nxt_batch(X,Y,self.batch_size,indx)    # take the right batch

        # Loop through valid batches:
        while Xbatch is not None  :
            # print("batch_i="+str(total_batches)+"/"+str(n_batches)+"\n")
            mean_loss = sess.run(self.valid_loss, feed_dict={self.X_valid: Xbatch,self.Y_valid: Ybatch,self.keep_prob:1.0})
            Xbatch,Ybatch,indx=mf.read_nxt_batch(X,Y,self.batch_size,indx)
            if math.isnan(mean_loss):
                print('valid cost is NaN')
                break
            valid_loss += mean_loss
            total_batches += 1

        if total_batches > 0:
            valid_loss /= total_batches

        return valid_loss


    def train(self,sess,iter):
        start_time = time.clock()

        n_early_stop_epochs = 10  # Define it
        n_epochs = 30  # Define it

        # restore variables from previous train session
        if(iter>0): restore_variables(sess)

        # create saver object
        saver = tf.train.Saver(var_list = tf.trainable_variables(), max_to_keep = 4)

        early_stop_counter=0

        # initialize train variables
        init_op = tf.group(tf.global_variables_initializer())

        sess.run(init_op)

        # assign a large value to min
        min_valid_loss = sys.float_info.max
        epoch=0

        # loop for a given number of epochs
        while (epoch < n_epochs): # max num epoch iteration
            epoch += 1
            epoch_start_time = time.clock()

            train_loss = self.train_epoch(sess)
            valid_loss = self.valid_epoch(sess)
            # print("valid ends")
            epoch_end_time=time.clock()

            info_str='Epoch='+str(epoch) + ', Train: ' + str(train_loss) + ', Valid: '
            info_str += str(valid_loss) + ', Time=' +str(epoch_end_time - epoch_start_time)
            print(info_str)

            if valid_loss < min_valid_loss:
                print('Best epoch=' + str(epoch))
                save_variables(sess, saver, epoch, self.model_id)
                min_valid_loss=valid_loss
                early_stop_counter=0
            else:
                early_stop_counter += 1

            # stop training when overfiiting conditon is true
            if early_stop_counter > n_early_stop_epochs:
                # too many consecutive epochs without surpassing the best model
                print('stopping early')
                break
        end_time=time.clock()
        print('Total time = ' + str(end_time - start_time))


    def define_predict_operations(self):
        self.keep_prob    = tf.placeholder(dtype=tf.float32,name='keep_prob')       # placeholder for keeping dropout probality

        self.X_eval = tf.placeholder(dtype=tf.float32, shape=(None, self.height,self.width,self.chan),name='X_eval')  # Define this

        self.Y_eval = tf.placeholder(dtype=tf.int32, shape=(None, ),name='Y_eval')  # Define this

        self.Y_eval_predict =tf.nn.softmax(self.model_architecture(self.X_eval,keep_prob,reuse=True,is_training=False)) # make a prediction using inference softmax

        #Return the index with the largest value across axis
        Ypredict = tf.argmax(self.Y_eval_predict, axis=1, output_type=tf.int32) #in [0,1,2]
        #Cast a boolean tensor to float32
        correct = tf.cast(tf.equal(Ypredict, self.Y_eval), tf.float32)
        self.accuracy_graph = tf.reduce_mean(correct)

    def predict_utterance(self,sess,Xeval_in,Yeval_in):
        accuracy=sess.run(self.accuracy_graph, feed_dict={self.X_eval: Xeval_in, self.Y_eval: Yeval_in})
        return accuracy
