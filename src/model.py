from __future__ import division, print_function
import tensorflow as tf
import model_func as mf
import traceback
import sys
import time
from datetime import datetime
import math
import numpy as np
from lib.model_io import save_variables
from lib.model_io import restore_variables
from lib.model_io import read_model_id
from lib.precision import _FLOATX
import read_img as rim



class CNN(object):

    # initialize class object
    def __init__(self,model_id=None):
        self.model_id = model_id
        # variables to store input data
        self.Xtrain_in = np.empty(0)
        self.Ytrain_in = np.empty(0)
        # self.Xvalid_in = np.empty(0)
        # self.Yvalid_in = np.empty(0)
        self.Xeval_in  = np.empty(0)
        self.Yeval_in  = np.empty(0)
        self.summ_indx = 0
        # model variables
        self.height = 64
        self.width = 17
        self.chan = 1           # channel of image 1 or 3 if rgb
        self.n_input = self.height * self.width*self.chan # 64x17x1
        self.n_classes = 2      # genuine or spoof --number of classes
        self.dropout = 0.75     # dropout probality
        self.batch_size = 512   # 64 || 128 || 256
        self.train_size = 0
        self.valid_size = 0
        self.eval_size = 0

        # flag to early stopping
        self.kill = False


    def model_architecture(self, X, keep_prob,is_training=True):

        with tf.variable_scope("model_architecture", reuse=tf.AUTO_REUSE):
            # Reshape input picture
            X = tf.reshape(X, shape=[-1, self.height, self.width, 1])
            # tf.summary.image('input',X,1)
            #               --{1st BLOCK}--
            # 1st layer
            shape = [3,3,1,4] # first layer
            with tf.variable_scope("convolution_layer1",reuse=tf.AUTO_REUSE):
                conv_l1 = mf.conv_layer(X,shape,"conv_l1")
            with tf.variable_scope("batch_norm_layer1",reuse=tf.AUTO_REUSE):
                conv_l1 = mf.batch_n(conv_l1,'batch_norm_l1')
            # 2nd layer
            shape = [3,3,4,4]
            with tf.variable_scope("convolution_layer2",reuse=tf.AUTO_REUSE):
                conv_l2 = mf.conv_layer(conv_l1,shape,'conv_l2')
            with tf.variable_scope("batch_norm_layer2",reuse= tf.AUTO_REUSE):
                conv_l2 = mf.batch_n(conv_l2,'batch_norm_l2')

            with tf.variable_scope("max-pooling_layer1",reuse= tf.AUTO_REUSE):
                max_pool_1 = mf.max_pool(conv_l2,1,1,'max_pool_bl2')

            #               --{2nd BLOCK}--
            # 3d layer
            shape = [3,3,4,8]
            with tf.variable_scope("convolution_layer3",reuse= tf.AUTO_REUSE):
                conv_l3 = mf.conv_layer(max_pool_1,shape,'conv_l3')
            with tf.variable_scope("batch_norm_layer3",reuse= tf.AUTO_REUSE):
                conv_l3 = mf.batch_n(conv_l3,'batch_norm_l3')
            # 4th layer
            shape = [3,3,8,8]
            with tf.variable_scope("convolution_layer4",reuse= tf.AUTO_REUSE):
                conv_l4 = mf.conv_layer(conv_l3,shape,'conv_l4')
            with tf.variable_scope("batch_norm_layer4",reuse= tf.AUTO_REUSE):
                conv_l4 = mf.batch_n(conv_l4,'batch_norm_l4')
            with tf.variable_scope("max-pooling_layer2",reuse= tf.AUTO_REUSE):
                max_pool_2 = mf.max_pool(conv_l4,1,1,'max_pool_bl2')

            #               --{3d BLOCK}--
            # 5th layer
            shape = [3,3,8,16]
            with tf.variable_scope("convolution_layer5",reuse= tf.AUTO_REUSE):
                conv_l5 = mf.conv_layer(max_pool_2,shape,'conv_l5')
            with tf.variable_scope("batch_norm_layer5",reuse= tf.AUTO_REUSE):
                conv_l5 = mf.batch_n(conv_l5,'batch_norm_l5')
            # 6th layer
            shape = [3,3,16,16]
            with tf.variable_scope("convolution_layer6",reuse= tf.AUTO_REUSE):
                conv_l6 = mf.conv_layer(conv_l5,shape,'conv_l6')
            with tf.variable_scope("batch_norm_layer6",reuse= tf.AUTO_REUSE):
                conv_l6 = mf.batch_n(conv_l6,'batch_norm_l6')
            with tf.variable_scope("max-pooling_layer3",reuse= tf.AUTO_REUSE):
                max_pool_3 = mf.max_pool(conv_l6,1,1,'max_pool_3')
            #              --{4th BLOCK}
            # 7th layer
            shape = [3,3,16,32]
            with tf.variable_scope("convolution_layer7",reuse= tf.AUTO_REUSE):
                conv_l7 = mf.conv_layer(max_pool_3,shape,'conv_l7')
            with tf.variable_scope("batch_norm_layer7",reuse= tf.AUTO_REUSE):
                conv_l7 = mf.batch_n(conv_l7,'batch_norm_l7')
            # 8th layer
            shape = [3,3,32,32]
            with tf.variable_scope("convolution_layer8",reuse= tf.AUTO_REUSE):
                conv_l8 = mf.conv_layer(conv_l7,shape,'conv_l8')
            with tf.variable_scope("batch_norm_layer8",reuse= tf.AUTO_REUSE):
                conv_l8 = mf.batch_n(conv_l8,'batch_norm_l8')
            with tf.variable_scope("max-pooling_layer4",reuse= tf.AUTO_REUSE):
                max_pool_4 = mf.max_pool(conv_l8,2,2,'max_pool_4')
            #               --{5th BLOCK}
            # 9th layer
            shape =[3,3,32,64]
            with tf.variable_scope("convolution_layer9",reuse= tf.AUTO_REUSE):
                conv_l9 = mf.conv_layer(max_pool_4,shape,'conv_l9')
            with tf.variable_scope("batch_norm_layer9",reuse= tf.AUTO_REUSE):
                conv_l9 = mf.batch_n(conv_l9,'batch_norm_l9')
            # 10th layer
            shape = [3,3,64,64]
            with tf.variable_scope("convolution_layer10",reuse= tf.AUTO_REUSE):
                conv_l10 = mf.conv_layer(conv_l9,shape,'conv_l10')
            with tf.variable_scope("batch_norm_layer10",reuse= tf.AUTO_REUSE):
                conv_l10 = mf.batch_n(conv_l10,'batch_norm_l10')
            with tf.variable_scope("max-poooling_layer5",reuse= tf.AUTO_REUSE):
                max_pool_5 = mf.max_pool(conv_l10,2,2,'max_pool_5')

            #           --{FULLY CONNECTED LAYERS}--
            with tf.variable_scope("flatt-out_layer3",reuse= tf.AUTO_REUSE):
                flatt_out = mf.flatten_l(max_pool_5,'flatten_out_layer')
            with tf.variable_scope("fully_connected_layer1",reuse= tf.AUTO_REUSE):
                fc1 = mf.fully_con(flatt_out,256,'fc1')
                fc1 = tf.nn.dropout(fc1,keep_prob)
            with tf.variable_scope("fully_connected_layer2",reuse= tf.AUTO_REUSE):
                fc2 = mf.fully_con(fc1,512,'fc2')
            with tf.variable_scope("Logits-Layer-end",reuse= tf.AUTO_REUSE):
                logits = mf.dense_layer(fc2,self.n_classes,'Last_layer')    # last layer not activation function is used for trainning only
            # logits = mf.outp_layer(fc1,self.n_classes,'Last_layer')
            print('Logits_shape='+str(logits.shape))
            # self.merged = tf.summary.merge_all()
        return logits
    # set up graph operations
    def define_train_operations(self):
        # set placeholders
        self.keep_prob = tf.placeholder(dtype=tf.float32,name='keep_prob')

        self.X   = tf.placeholder(dtype=tf.float32, shape=(None,self.n_input),name='X')

        self.Y   = tf.placeholder(dtype=tf.int32,shape=(None, self.n_classes), name='Y')

        # network TRAIN set prediction                  --|TRAIN|--
        self.Y_predict = self.model_architecture(self.X,self.keep_prob) #logits for loss
        self.Y_pred_soft = tf.nn.softmax(self.Y_predict) # softmax for accuracy
        # tf.summary.histogram('soft-act',self.Y_train_soft)

        # calculate LOSS between real label and predicted
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.Y_predict, labels=self.Y,name='cost'))
        self.cost2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.Y_predict, labels=self.Y,name='cost'))
        # tf.summary.scalar('train_loss',self.loss)
        # define learning rate decay method
        global_step = tf.Variable(0, trainable=False, name='global_step')
        # Define it--play with this
        learning_rate = 0.0001

        # the optimization algorithm
        optimizer = tf.contrib.optimizer_v2.AdamOptimizer(learning_rate,beta1=0.9, beta2=0.999, epsilon=1e-8,name='training_Adam') #tf.train.AdamOptimizer(learning_rate,beta1=0.9, beta2=0.999, epsilon=1e-8,name='training_Adam')
        self.trainable = tf.trainable_variables()  # may be the weights  ??
        self.update_ops = optimizer.minimize(self.cost, var_list=self.trainable, global_step=global_step)

        # softmax - accuracy
        y_pred = tf.argmax(self.Y_pred_soft,axis=1,output_type=tf.int32) # arg max of the predicted output -softmax
        y_correct = tf.argmax(self.Y, axis=1, output_type=tf.int32) # arg-max of the actual input --placeholder
        # Cast a boolean tensor to float32
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(y_pred, y_correct), tf.float32))
        # tf.summary.scalar('acc',self.accuracy)





    # define train actions per epoch
    def train_epoch(self,sess):
        print("Train_epoch")
        train_loss = 0
        total_batches = 0
        # print("TOTAL="+str(self.train_size))
        n_batches = self.train_size / self.batch_size
        # print(n_batches)
        indx=0
        X,Y=mf.shuffling(self.Xtrain_in,self.Ytrain_in)                   # shuffle X ,Y data
        Xbatch,Ybatch,indx=mf.read_nxt_batch(X,Y,self.batch_size,indx)    # take the right batch

        while Xbatch is not None:     # loop through train batches:
            mean_loss, _ = sess.run([self.cost,self.update_ops], feed_dict={self.X: Xbatch ,self.Y: Ybatch,self.keep_prob:self.dropout})
            Xbatch,Ybatch,indx = mf.read_nxt_batch(X,Y,self.batch_size,indx)
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
        # print('vddd:'+str(self.valid_size)+'\n')
        n_batches = self.valid_size / self.batch_size  # number of elements
        indx=0
        X,Y=mf.shuffling(self.Xvalid_in,self.Yvalid_in)  # shuffle X ,Y data
        Xbatch,Ybatch,indx=mf.read_nxt_batch(X,Y,self.batch_size,indx)    # take the right batch

        # Loop through valid batches:
        while Xbatch is not None  :
            # print("batch_i="+str(total_batches)+"/"+str(n_batches)+"\n")
            mean_loss = sess.run(self.cost2, feed_dict={self.X: Xbatch,self.Y: Ybatch,self.keep_prob:1.0})
            Xbatch,Ybatch,indx = mf.read_nxt_batch(X,Y,self.batch_size,indx)
            if math.isnan(mean_loss):
                print('valid cost is NaN')
                break
            valid_loss += mean_loss
            total_batches += 1

        if total_batches > 0:
            valid_loss /= total_batches

        return valid_loss


    def train(self,sess,writer_train,writer_valid,iter):
        start_time = time.clock()

        n_early_stop_epochs = 15 # Define it
        n_epochs = 25  # Define it

        early_stop_counter = 0

        # assign a large value to min
        min_valid_loss = sys.float_info.max
        epoch=0

        # loop for a given number of epochs
        while (epoch < n_epochs): # max num epoch iteration
            epoch += 1
            epoch_start_time = time.clock()
            self.summ_indx += 1
            train_loss = self.train_epoch(sess) # train_loss on the mini batch
            valid_loss = self.valid_epoch(sess) # valid_loss on the mini batch
            # print("valid ends")
            epoch_end_time=time.clock()
            # if (epoch % 10 == 0):
            info_str ='Epoch='+str(epoch) + ', Train:{:.10f} '  .format(train_loss) + ', Valid:{:.10f} '.format(valid_loss) + ', Time=' +str(epoch_end_time - epoch_start_time)
            print(info_str)

            if valid_loss < min_valid_loss:
                print('Best epoch=' + str(epoch))
                saver = tf.train.Saver()
                save_variables(sess,saver,iter,read_model_id)
                # tf.train.write_graph(sess.graph.as_graph_def(),'../Variables8','tensorflowModel.pbtxt', as_text=True)
                min_valid_loss = valid_loss
                early_stop_counter = 0
            else:
                early_stop_counter += 1

            # # accuracy and summaries
            # train_acc,s1 = sess.run([self.accuracy,merged],feed_dict={self.X: self.Xtrain_in,self.Y: self.Ytrain_in,self.keep_prob:1.0})
            # writer_train.add_summary(s1,self.summ_indx)
            # writer_train.flush()
            #
            # valid_acc,s2 = sess.run([self.accuracy,merged],feed_dict={self.X: self.Xvalid_in,self.Y: self.Yvalid_in,self.keep_prob:1.0})
            # writer_valid.add_summary(s2,self.summ_indx)
            # writer_valid.flush()
            # evaluate training
            # if (epoch % 10 == 0):
            #     print('[**epoch= '+str(epoch) + ', train_acc ={:.3f} ' .format(train_acc)+'valid_acc ={:.3f} '.format(valid_acc) +' **]\n')


            # stop training when overfiiting conditon is true
            if early_stop_counter > n_early_stop_epochs:
                # too many consecutive epochs without surpassing the best model
                print('stopping early')
                self.kill = True
                # break
        end_time=time.clock()
        print('Total time = ' + str(end_time - start_time))
