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
import read_img as rim
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
    return(tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='SAME'))
# define convolution layer


def conv_layer(inp, shape):
    W = weight_dict(shape)
    b = bias_dict([shape[3]])
    return(tf.nn.relu(conv2d(inp, W) + b))
# define max pooling function
def max_pool(x, stride, k):
    return (tf.nn.max_pool(x, strides=[1, 2, stride, 1], ksize=[1, 2, k, 1], padding='VALID'))



#flatten layer
def flatten_l(inp):
    flatten_size=inp.shape[1]*inp.shape[2]*inp.shape[3]
    return (tf.reshape(inp,[-1,flatten_size]) )

# dense layer
#args:
#@ inp : input from previous layer
#@ n_outp : output dimension-nodes
def dense_layer(inp, n_outp):
    n_features=inp.shape[1].value
    w=weight_dict([n_features,n_outp])
    b=bias_dict([n_outp])
    return(tf.matmul(inp,w)+b)

#fully_connected layer -- a dense layer that apllies relu function
def fully_con(inp,n_outp):
    fc=dense_layer(inp,n_outp)
    return(tf.nn.relu(fc))

#Last Layer -output layer decides a class with a possibility
#args:
#@inp : input tensor from previous layers
#@n_outp : output nodes --number of classes at output
def outp_layer(inp,n_outp):
    outp=dense_layer(inp,n_outp)
    return(tf.nn.softmax(outp))

# batch normalization
def batch_n(convl):
    return (tf.nn.relu(tf.contrib.layers.batch_norm(convl)))
# -------------------------------------------------------------------------------------------------------


class CNN(object):

    def __init__(self, model_id=None):
        self.model_id = model_id
        self.Xtrain_in = np.empty(0)
        self.Ytrain_in = np.empty(0)
        self.Xvalid_in = np.empty(0)
        self.Yvalid_in = np.empty(0)

        # __________________________________________________
        self.height = 64
        self.width = 17
        self.chan = 1           # channel of image 1 or 3 if rgb
        self.n_classes = 2      # genuine or spoof --number of classes
        self.batch_size = 256   # 64 || 128 || 256

        self.train_size = 0  # 1587420  # number of frames (train)
        self.dev_size = 0  # 1029721  # number of frames (valid)
        self.eval_size = 0  # 8522944 #number of frames (eval)

    # normalize input data
    def normalize(self, X):
        col_max = np.max(X, axis=0)
        col_min = np.min(X, axis=0)
        normX = np.divide(X - col_min, col_max - col_min)
        return normX
    # read input data
    def input(self):
        self.Xtrain_in, self.Ytrain_in, self.train_size = rim.read_Data(
            "ASVspoof2017_V2_train_fbank", "train_info.txt")  # Read Train data
        # print(self.Xtrain_in)
        # Normalize input train set data
        self.Xtrain_in = self.normalize(self.Xtrain_in)
        print(self.Ytrain_in)
        print("shape"+str(self.Ytrain_in.shape))
        self.Xvalid_in, self.Yvalid_in, self.dev_size = rim.read_Data(
            "ASVspoof2017_V2_train_dev", "dev_info.txt")  # Read validation data
        # Normalize input validation set data
        self.Xvalid_in = self.normalize(self.Xvalid_in)
        print(self.Yvalid_in)
        print("shape"+str(self.Yvalid_in.shape))
    # shuffle index so to get with random order the data
    def shuffling(self, X,Y):
        indx=np.arange(len(X))      #create a array with indexes for X data
        np.random.shuffle(indx)
        X=X[indx]
        Y=Y[indx]
        # print("shuffle")
        # print(Y.shape)
        return X,Y
    #take X batch and Ybatch
    def read_nxt_batch(self,X,Y,batch_s,indx=0):
        if(indx+batch_s <len(X)):     # check and be sure that you're in range
            print(Y.shape)
            X=X[indx:indx+batch_s]  #take a batch from shuffled data 0..255 is 256!
            Y=Y[indx:indx+batch_s]  #take a batch from shuffled labels
            indx+=batch_s           #increase indx, move to indx the next batch
            return(X,Y,indx)
        else:
            return None,None,None

    def inference(self, X, keep_prob,reuse=True,is_training=True):
        with tf.variable_scope("inference", reuse=reuse):
            # Implement your network here
            # equation or predefiend fuctions --convolution operation
            # we define set of layers according to the max_pooling ksize
            # each set has more than one convolution and max_poolin layers
            # totaly we have 2 sets and 5 blocks
            # init phase
            # [filter_h,filter_w,in_channel,out_channel]
            shape1 = [3, 3, 1, 1]
            w = weight_dict(shape1)
            b = bias_dict([shape1[3]])  # out_channel== n_hidden
            conv1 = conv2d(X, w) + b    # init_convolution

    # -----------1st set--------{2 blocks}---------------------------------------
            # -------1st block
            conv_l1 = conv_layer(conv1, [3, 3, 1, 4])
            batch_norm1 = batch_n(conv_l1)  # batch normalization
            conv_l2 = conv_layer(batch_norm1, [3, 3, 4, 4])
            batch_norm2 = batch_n(conv_l2)           # batch normalization

            mpool_1 = max_pool(batch_norm2, 1, 1)   # stride =1 , k=1
            # ------2nd block
            conv_l3 = conv_layer(mpool_1, [3, 3, 4, 8])
            batch_norm3 = batch_n(conv_l3)  # batch normalization
            conv_l4 = conv_layer(batch_norm3, [3, 3, 8, 8])
            batch_norm4 = batch_n(conv_l4)  # batch normalization

            mpool_2 = max_pool(batch_norm4, 1, 1)   # stride =1 , k=1

    # --------2nd set------{3 blocks}--------------------------------------------
            # -------3d block
            conv_l5 = conv_layer(mpool_2, [3, 3, 8, 16])
            batch_norm5 = batch_n(conv_l5)      # normalization
            conv_l6 = conv_layer(batch_norm5, [3, 3, 16, 16])
            batch_norm6 = batch_n(conv_l6)      # normalization batch

            mpool_3 = max_pool(batch_norm6, 1, 1)   # stride =1 , k=1
            # --------4th block
            conv_l7 = conv_layer(mpool_3, [3, 3, 16, 32])
            batch_norm7 = batch_n(conv_l7)
            conv_l8 = conv_layer(batch_norm7, [3, 3, 32, 32])
            batch_norm8 = batch_n(conv_l8)

            mpool_4 = max_pool(batch_norm8, 2, 2)   # stride=2, k=2
            # --------5th blocks

            conv_l9 = conv_layer(mpool_4, [3, 3, 32, 64])
            batch_norm9 = batch_n(conv_l9)
            conv_l10 = conv_layer(batch_norm9, [3, 3, 64, 64])
            batch_norm10 = batch_n(conv_l10)

            mpool_5 = max_pool(batch_norm10, 2, 2)      # stride=2, k=2
            # print("SHAPE../n")
            # print(mpool_5.shape)
            # print(mpool_5.shape[1].value)

    # ------------add dense layers {4 layers}-------------------------------------

            flatt_out=flatten_l(mpool_5)        #flatten out tensor from 4D to 2D
            l=fully_con(flatt_out,256)           #1st dense-relu layer
            l=fully_con(l,512)                 #2nd
            l=tf.nn.dropout(l,keep_prob=keep_prob)

            logits=outp_layer(l,self.n_classes) #propability decision between two classes (Genuine or Spoofed)
            # print("$$")
            # print(logits.shape)
        return logits

    def define_train_operations(self):

        self.keep_prob=tf.placeholder(tf.float32,name='keep_prob')       #placeholder for keeping dropout probality

        self.X_data_train = tf.placeholder(dtype=tf.float32, shape=(None, self.height,self.width,self.chan),name='X_data_train')  # Define this

        self.Y_data_train = tf.placeholder(dtype=tf.int32, shape=(None, ),name='Y_data_train')  # Define this

        # Network prediction
        Y_net_train = self.inference(
            self.X_data_train,self.keep_prob,reuse=False)

        # Loss of train data tf.nn.softmax_cross_entropy_with_logits
        self.train_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.Y_data_train, logits=Y_net_train, name='train_loss'))

        # define learning rate decay method
        global_step = tf.Variable(0, trainable=False, name='global_step')
        # Define it--play with this
        learning_rate = 0.001

        # define the optimization algorithm
        # Define it --shall we try different type of optimizers
        optimizer = tf.train.AdamOptimizer(learning_rate)

        trainable = tf.trainable_variables()  # may be the weights??
        self.update_ops = optimizer.minimize(
            self.train_loss, var_list=trainable, global_step=global_step)

        # --- Validation computations
        self.X_data_valid = tf.placeholder(dtype=tf.float32, shape=(None, self.height, self.width, self.chan))  # Define this
        self.Y_data_valid = tf.placeholder(dtype=tf.int32, shape=(None, ))  # Define this

        # Network prediction
        Y_net_valid = self.inference(self.X_data_valid,self.keep_prob,reuse=True)
        # print("Ydata_valid.shape="+str(self.Y_data_valid.shape))
        # Loss of validation data
        self.valid_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=self.Y_data_valid, logits=Y_net_valid, name='valid_loss'))

    # def read_nxt_batch(self):

    def train_epoch(self, sess):
        print("Train_epoch")
        train_loss = 0
        total_batches = 0
        # print("TOTAL="+str(self.train_size))
        n_batches = self.train_size / self.batch_size  # ??
        indx=0
        X,Y=self.shuffling(self.Xtrain_in,self.Ytrain_in)  # shuffle X ,Y data
        Xbatch,Ybatch,indx=self.read_nxt_batch(X,Y,self.batch_size,indx)    # take the right batch
        while Xbatch is not None:     # loop through train batches:
            # print("Ybatch=")
            # print(Ybatch.shape)
            mean_loss, _ = sess.run([self.train_loss, self.update_ops], feed_dict={self.X_data_train: Xbatch ,self.Y_data_train: Ybatch,self.keep_prob:0.3})
            Xbatch,Ybatch,indx=self.read_nxt_batch(X,Y,self.batch_size,indx)
            if math.isnan(mean_loss):
                print('train cost is NaN')
                break
            train_loss += mean_loss
            total_batches += 1

        if total_batches > 0:
            train_loss /= total_batches

        return train_loss

    def valid_epoch(self, sess):
        print("Valid_epoch")

        valid_loss = 0
        total_batches = 0
        keep_probability=0.0     #dropout probability
        n_batches = self.dev_size / self.batch_size  # number of elements
        indx=0
        X,Y=self.shuffling(self.Xvalid_in,self.Yvalid_in)  # shuffle X ,Y data
        Xbatch,Ybatch,indx=self.read_nxt_batch(X,Y,self.batch_size,indx)    # take the right batch
        # Loop through valid batches:
        while Xbatch is not None  :
            # print("batch_i="+str(total_batches)+"/"+str(n_batches)+"\n")
            mean_loss = sess.run(self.valid_loss, feed_dict={self.X_data_valid: Xbatch,self.Y_data_valid: Ybatch,self.keep_prob:1.0})
            Xbatch,Ybatch,indx=self.read_nxt_batch(X,Y,self.batch_size,indx)
            if math.isnan(mean_loss):
                print('valid cost is NaN')
                break
            valid_loss += mean_loss
            total_batches += 1

        if total_batches > 0:
            valid_loss /= total_batches

        return valid_loss

    def train(self, sess):
        start_time = time.clock()

        n_early_stop_epochs = 10  # Define it
        n_epochs = 50  # Define it

        saver = tf.train.Saver(var_list = tf.trainable_variables(), max_to_keep = 4)

        early_stop_counter=0

        init_op=tf.group(tf.global_variables_initializer())

        sess.run(init_op)

        min_valid_loss=sys.float_info.max
        epoch=0
        while (epoch < n_epochs): #max num epoch iteration
            epoch += 1
            epoch_start_time=time.clock()

            train_loss=self.train_epoch(sess)
            valid_loss=self.valid_epoch(sess)
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

            if early_stop_counter > n_early_stop_epochs:
                # too many consecutive epochs without surpassing the best model
                print('stopping early')
                break

        end_time=time.clock()
        print('Total time = ' + str(end_time - start_time))

    def define_predict_operations(self):
        self.X_data_test_placeholder=tf.placeholder(dtype=tf.float32, shape = (None, self.height,self.width,self.chan))  # ??
        self.keep_prob_placeholder=tf.placeholder(dtype=tf.float32,name='keep_prob')
        self.Y_data_test_placeholder=tf.placeholder(dtype=tf.int32,shape=(None, ))
        self.Y_net_test=self.inference(
            self.X_data_test_placeholder,self.keep_prob_placeholder ,reuse = False,is_training=False)

    def predict_utterance(self, sess, Xeval, Yeval):
        keep_probability=1.0
        Yhat=self.Y_net_test  # logits

        Ypredict=tf.argmax(Yhat, axis = 1, output_type = tf.int32)
        Ycorrect=tf.argmax(Yeval, axis = 1, output_type = tf.int32)

        # CAst boolean tensor to float
        correct=tf.cast(tf.equal(Ypredict, Ycorrect), tf.float32)
        accuracy_graph=tf.reduce_mean(correct)
        accuracy=sess.run(accuracy_graph, feed_dict = {self.X_data_test_placeholder: Xeval, self.Y_data_test_placeholder: Yeval, self.keep_prob_placeholder:keep_probability})

        return accuracy
