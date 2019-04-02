import traceback
import os
import numpy as np
import tensorflow as tf
from cnn_model import CNN
from lib.model_io import get_model_id
from lib.model_io import restore_variables
from lib.model_io import read_model_id
import read_img as rim
import model_func as mf

flag=0

try:
    # change this according to your path
    path_to_train_set = "/home/tassos/Desktop/DATA_ASR/ASVspoof2017_V2_train_fbank"
    path_to_valid_set = "/home/tassos/Desktop/DATA_ASR/ASVspoof2017_V2_train_dev"

    model_id = get_model_id()
    # model_id = read_model_id
    n_tfiles=200 # how many train files will read per step
    n_vfiles=round(0.25*n_tfiles) # number of  validation files to be read per iter
    # print("a= \n")
    # print(n_vfiles)
    # cheat count files number
    total_inp_files = len(os.listdir(path_to_train_set))


    # Create the network
    network = CNN(model_id)
    iter=0
    # mf.input(network,n_tfiles,n_vfiles)
    # print(network.Xtrain_in)
    for i in range(1,total_inp_files,n_tfiles):
    # loop until all data are read

        mf.input(network,n_tfiles,n_vfiles)

        with tf.device('/gpu:0'):
            # restore()
            if(iter==0):
                # Define the train computation graph
                network.define_train_operations()

            # Train the network
            sess = tf.Session(config=tf.ConfigProto(allow_soft_placement = True)) # session with log about gpu exec
            #sess= tf.Session()
            try:
                print(iter)
                network.train(sess,iter)
                iter += 1
                flag = 0
                if(network.kill):break # if overfitting kill loop
            except KeyboardInterrupt:
                print()
                flag = 1
            finally:
                flag = 1
                sess.close()
except KeyboardInterrupt:
    flag=1
except Exception as e:
    flag=1
    # template = "An exception of type {0} occurred. Arguments:\n{1!r}"
    # message = template.format(type(e).__name__, e.args)
    # print(message)
    print("\n")
    traceback.print_exc()

# train finished -remove files

if (flag==1):
    file1 = os.path.basename(path_to_train_set)+"_status.txt"
    if(os.path.exists(file1)): os.remove(file1)
    file2 = os.path.basename(path_to_valid_set)+"_status.txt"
    if(os.path.exists(file2)): os.remove(file2)
    if(os.path.exists('labels_status.txt')):os.remove('labels_status.txt')
