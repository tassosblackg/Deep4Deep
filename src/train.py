import traceback
import os
import numpy as np
import tensorflow as tf
from cnn_model import CNN
from lib.model_io import get_model_id
from lib.model_io import restore_variables
from lib.model_io import read_model_id
from tensorflow.python import debug as tf_debug
import read_img as rim
import model_func as mf
import shutil as sh

LOGDIR_train = '../graphs-train'
LOGDIR_valid = '../graphs-valid'

flag=0
try:

    # change this according to your path
    path_to_train_set = "/home/tassos/Desktop/DATA_ASR/ASVspoof2017_V2_train_fbank"
    path_to_valid_set = "/home/tassos/Desktop/DATA_ASR/ASVspoof2017_V2_train_dev"

    model_id = get_model_id()
    # model_id = read_model_id
    n_tfiles=80 # how many train files will read per step
    n_vfiles=round(0.567*n_tfiles) # number of  validation files to be read per iter

    # cheat count files number
    total_inp_files = len(os.listdir(path_to_train_set))

    # Create the network
    network = CNN(model_id)
    iter=0
    # mf.input(network,n_tfiles,n_vfiles)
    # print(network.Xtrain_in)


    with tf.device('/gpu:0'):
        tf.reset_default_graph()
        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement = True)) # session with log about gpu exec
        # sess = tf_debug.LocalCLIDebugWrapperSession(sess)

        # Define the train computation graph
        network.define_graph_op()
        init_op = tf.group(tf.global_variables_initializer())
        sess.run(init_op)
        # remove previous tensorboard  files
        if (os.path.exists(LOGDIR_train)):
            sh.rmtree(LOGDIR_train)
        writer_train = tf.summary.FileWriter(LOGDIR_train, sess.graph)
        if (os.path.exists(LOGDIR_valid)):
            sh.rmtree(LOGDIR_valid)
        writer_valid = tf.summary.FileWriter(LOGDIR_valid)

        # # ITerate through input data --\SUBSET\

        for i in range(1,total_inp_files,n_tfiles):
        # loop until all data are read

            mf.input(network,n_tfiles,n_vfiles) # read input data
            if(iter>1):
            #     # initialize train variables
            #     init_op = tf.group(tf.global_variables_initializer())
            #     sess.run(init_op)
            # else:
                # tf.reset_default_graph()
                # new_saver = tf.train.import_meta_graph('saved_models/model.ckpt.meta')
                # new_saver.restore(sess, tf.train.latest_checkpoint('./saved_models'))
                restore_variables(sess)

            print(iter)
            network.train(sess,writer_train,writer_valid,iter)
            iter += 1
            flag = 0

        sess.close()

except KeyboardInterrupt:
    flag=1
except Exception as e:
    flag=1

    print("\n")
    traceback.print_exc()

# train finished -remove files

if (flag==1):
    file1 = os.path.basename(path_to_train_set)+"_status.txt"
    if(os.path.exists(file1)): os.remove(file1)
    if(os.path.exists("dev_info_labels_status.txt")):os.remove("dev_info_labels_status.txt")
    file2 = os.path.basename(path_to_valid_set)+"_status.txt"
    if(os.path.exists(file2)): os.remove(file2)
    if(os.path.exists('labels_status.txt')):os.remove('labels_status.txt')
    if(os.path.exists("train_info_labels_status.txt")):os.remove("train_info_labels_status.txt")
