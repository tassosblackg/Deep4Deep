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
        self.n_input = self.height * self.width*self.chan # 64x17x1
        self.n_classes = 2      # genuine or spoof --number of classes
        self.dropout = 0.75     # dropout probality
        self.batch_size = 256   # 64 || 128 || 256
        self.train_size = 0
        self.valid_size = 0
        self.eval_size = 0


    def model_architecture(self, X, keep_prob,reuse=True,is_training=True):
        with tf.variable_scope("model_architecture", reuse=reuse):
            # Reshape input picture
            X = tf.reshape(X, shape=[-1, self.height, self.width, 1])
            #               --{1st BLOCK}--
            # 1st layer
            shape = [3,3,1,4] # first layer
            w1 = mf.weight_dict(shape,'w1')
            b1 = mf.bias_dict([shape[3]],'b1')
            conv_l1 = mf.conv2d(X,w1,b1,'conv_l1')
            conv_l1 = mf.batch_n(conv_l1,'batch_norm_l1')
            # 2nd layer
            shape = [3,3,4,4]
            w2 = mf.weight_dict(shape,'w2')
            b2 = mf.bias_dict([shape[3]],'b2')
            conv_l2 = mf.conv2d(conv_l1,w2,b2,'conv_l2')
            conv_l2 = mf.batch_n(conv_l2,'batch_norm_l2')

            max_pool_1 = mf.max_pool(conv_l2,1,1,'max_pool_bl2')
            #               --{2nd BLOCK}--
            # 3d layer
            shape = [3,3,4,8]
            w3 = mf.weight_dict(shape,'w3')
            b3 = mf.bias_dict([shape[3]],'b3')
            conv_l3 = mf.conv2d(max_pool_1,w3,b3,'conv_l3')
            conv_l3 = mf.batch_n(conv_l3,'batch_norm_l3')
            # 4th layer
            shape = [3,3,8,8]
            w4 = mf.weight_dict(shape,'w4')
            b4 = mf.bias_dict([shape[3]],'b4')
            conv_l4 = mf.conv2d(conv_l3,w4,b4,'conv_l4')
            conv_l4 = mf.batch_n(conv_l4,'batch_norm_l4')

            max_pool_2 = mf.max_pool(conv_l4,1,1,'max_pool_bl2')
            #               --{3d BLOCK}--
            # 5th layer
            shape = [3,3,8,16]
            w5 = mf.weight_dict(shape,'w5')
            b5 = mf.bias_dict([shape[3]],'b5')
            conv_l5 = mf.conv2d(max_pool_2,w5,b5,'conv_l5')
            conv_l5 = mf.batch_n(conv_l5,'batch_norm_l5')
            # 6th layer
            shape = [3,3,16,16]
            w6 = mf.weight_dict(shape,'w6')
            b6 = mf.bias_dict([shape[3]],'b6')
            conv_l6 = mf.conv2d(conv_l5,w6,b6,'conv_l6')
            conv_l6 = mf.batch_n(conv_l6,'batch_norm_l6')

            max_pool_3 = mf.max_pool(conv_l6,1,1,'max_pool_3')
            #               --{4th BLOCK}
            # 7th layer
            shape = [3,3,16,32]
            w7 = mf.weight_dict(shape,'w7')
            b7 = mf.bias_dict([shape[3]],'b7')
            conv_l7 = mf.conv2d(max_pool_3,w7,b7,'conv_l7')
            conv_l7 = mf.batch_n(conv_l7,'batch_norm_l7')
            # 8th layer
            shape = [3,3,32,32]
            w8 = mf.weight_dict(shape,'w8')
            b8 = mf.bias_dict([shape[3]],'b8')
            conv_l8 = mf.conv2d(conv_l7,w8,b8,'conv_l8')
            conv_l8 = mf.batch_n(conv_l8,'batch_norm_l8')

            max_pool_4 = mf.max_pool(conv_l8,2,2,'max_pool_4')
            #               --{5th BLOCK}
            # 9th layer
            shape =[3,3,32,64]
            w9 = mf.weight_dict(shape,'w9')
            b9 = mf.bias_dict([shape[3]],'b9')
            conv_l9 = mf.conv2d(max_pool_4,w9,b9,'conv_l9')
            conv_l9 = mf.batch_n(conv_l9,'batch_norm_l9')
            # 10th layer
            shape = [3,3,64,64]
            w10 = mf.weight_dict(shape,'w10')
            b10 = mf.bias_dict([shape[3]],'b10')
            conv_l10 = mf.conv2d(conv_l9,w10,b10,'conv_l10')
            conv_l10 = mf.batch_n(conv_l10,'batch_norm_l10')

            max_pool_5 = mf.max_pool(conv_l10,2,2,'max_pool_5')

            #           --{FULLY CONNECTED LAYERS}--
            flatt_out = mf.flatten_l(max_pool_5,'flatten_out_layer')
            fc1 = mf.fully_con(flatt_out,256,'fc1')
            # fc1 = tf.nn.dropout(fc1,self.dropouttop)
            fc2 = mf.fully_con(fc1,512,'fc2')

            logits = mf.dense_layer(fc2,self.n_classes,'Last_layer')    # last layer not activation function is used for trainning only
            # logits = mf.outp_layer(fc2,self.n_classes,'Last_layer')
            print('Logits_shape='+str(logits.shape))
        return logits

    # last layer of network is a softmax output
    # apply softmax layer to logits is used for evaluation
    def inference(self,X,keep_prob):
        return tf.nn.softmax(self.model_architecture(X,keep_prob,reuse=True,is_training=False))

    def define_train_operations(self):
        # set placeholders
        self.keep_prob = tf.placeholder(dtype=tf.float32,name='keep_prob')

        self.X_train   = tf.placeholder(dtype=tf.float32, shape=(None,self.n_input),name='X_train')

        self.Y_train   = tf.placeholder(dtype=tf.int32,shape=(None, self.n_classes), name='Y_train')

        # network prediction
        Y_train_predict = self.model_architecture(self.X_train,self.keep_prob,reuse=False)
        # calculate training loss between real label and predicted
        self.train_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=Y_train_predict, labels=self.Y_train,name='train_loss'))


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
        self.X_valid = tf.placeholder(dtype=tf.float32, shape=(None, self.n_input))  # Define this
        self.Y_valid = tf.placeholder(dtype=tf.int32, shape=(None,self.n_classes))  # Define this
        # logits layer without softmax
        Y_valid_predict = self.model_architecture(self.X_valid,self.keep_prob,reuse=True)

        # Loss on validation
        self.valid_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=Y_valid_predict, labels=self.Y_valid,name='valid_loss'))
    # evaluation method
    def evaluate(self,sess,Xtest,Ytest,train=True):
        if(train==True):
            #calculate train accuracy
            y_softmax = self.inference(self.Xtrain_in,self.keep_prob) # apply softmax at the last layer
            y_pred = tf.argmax(y_softmax,axis=1,output_type=tf.int32)
            y_correct = tf.argmax(self.Y_train, axis=1, output_type=tf.int32)
            # Cast a boolean tensor to float32
            correct = tf.cast(tf.equal(y_pred, y_correct), tf.float32)
            accuracy_graph = tf.reduce_mean(correct)

            accuracy = sess.run(accuracy_graph,feed_dict={self.X_train: Xtest,self.Y_train: Ytest,self.keep_prob:1.0})
        else:
            # calculate validation accuracy
            y_softmax = self.inference(self.Xvalid_in,self.keep_prob) # apply softmax at the last layer
            y_pred = tf.argmax(y_softmax,axis=1,output_type=tf.int32)
            y_correct = tf.argmax(self.Y_valid, axis=1, output_type=tf.int32)
            # Cast a boolean tensor to float32
            correct = tf.cast(tf.equal(y_pred, y_correct), tf.float32)
            accuracy_graph = tf.reduce_mean(correct)
            accuracy = sess.run(accuracy_graph,feed_dict={self.X_valid: Xtest,self.Y_valid: Ytest,self.keep_prob:1.0})
        return accuracy

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
            mean_loss, _ = sess.run([self.train_loss, self.update_ops], feed_dict={self.X_train: Xbatch ,self.Y_train: Ybatch,self.keep_prob:self.dropout})
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
            # # evaluate training
            # if (epoch % 10 == 0):
            #     train_acc = self.evaluate(sess,self.Xtrain_in,self.Ytrain_in,train=True)
            #     valid_acc = self.evaluate(sess,self.Xvalid_in,self.Yvalid_in,train=False)
            #     print('[epoch= '+str(epoch) + ', train_acc ={:.3f}' .format(train_acc)+'valid_acc ={:.3f}'.format(valid_acc) +']\n')

            # stop training when overfiiting conditon is true
            if early_stop_counter > n_early_stop_epochs:
                # too many consecutive epochs without surpassing the best model
                print('stopping early')
                break
        end_time=time.clock()
        print('Total time = ' + str(end_time - start_time))

    # like define train operations
    def define_predict_operations(self):
        self.keep_prob    = tf.placeholder(dtype=tf.float32,name='keep_prob')       # placeholder for keeping dropout probality

        self.X_eval = tf.placeholder(dtype=tf.float32, shape=(None, self.n_input),name='X_eval')  # Define this

        self.Y_eval = tf.placeholder(dtype=tf.int32, shape=(None, self.n_classes),name='Y_eval')  # Define this

        self.Y_eval_predict = self.model_architecture(self.X_eval,self.keep_prob,reuse=True,is_training=False) # make a prediction using inference softmax
        #Return the index with the largest value across axis
        Ypredict = tf.argmax(self.Y_eval_predict, axis=1, output_type=tf.int32) #in [0,1,2]

        #Cast a boolean tensor to float32
        correct = tf.cast(tf.equal(Ypredict, self.Y_eval), tf.float32)
        self.accuracy_graph = tf.reduce_mean(correct)
    # like train - train epoch
    def predict_utterance(self,sess,Xeval_in,Yeval_in):
        # initialize variables
        init = tf.group(tf.global_variables_initializer)
        sess.run(init)
        accuracy=sess.run(self.accuracy_graph, feed_dict={self.X_eval: Xeval_in, self.Y_eval: Yeval_in,self.keep_prob:1.0})
        return accuracy
