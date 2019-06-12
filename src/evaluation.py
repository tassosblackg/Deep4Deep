# After train restore model and calculate accuracy of training and validation set

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


flag = 0

try:
    # change this according to your path
    path_to_train_set = "/home/tassos/Desktop/DATA_ASR/ASVspoof2017_V2_train_fbank"
    path_to_valid_set = "/home/tassos/Desktop/DATA_ASR/ASVspoof2017_V2_train_dev"

    model_id = read_model_id('some_file')
    # model_id = read_model_id
    n_tfiles=600 # how many train files will read per step
    n_vfiles=round(0.567*n_tfiles) # number of  validation files to be read per iter

    # cheat count files number
    total_inp_files = len(os.listdir(path_to_train_set))
    network = CNN(model_id)
    iter = 0

    with tf.device('/gpu:0'):
        network.evaluate_operations()
        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement = True))


        # ITerate through input data
        for i in range(1,total_inp_files,n_tfiles):
        # loop until all data are read

            mf.input(network,n_tfiles,n_vfiles) # read input data

            restore_variables(sess)


            train_acc = network.evaluate(sess,train=True) # train true, means train set
            valid_acc = network.evaluate(sess,train=False)
            print('[**Loop= '+str(i) + ', train_acc ={:.3f}' .format(train_acc)+'valid_acc ={:.3f}'.format(valid_acc) +' **]\n')

        print("exiting..")
except KeyboardInterrupt:
    flag = 1
except Exception as e:
    flag = 1
    print("\n")
    traceback.print_exc()

if (flag==1):
    file1 = os.path.basename(path_to_train_set)+"_status.txt"
    if(os.path.exists(file1)): os.remove(file1)
    if(os.path.exists("dev_info_labels_status.txt")):os.remove("dev_info_labels_status.txt")
    file2 = os.path.basename(path_to_valid_set)+"_status.txt"
    if(os.path.exists(file2)): os.remove(file2)
    if(os.path.exists('labels_status.txt')):os.remove('labels_status.txt')
    if(os.path.exists("train_info_labels_status.txt")):os.remove("train_info_labels_status.txt")
