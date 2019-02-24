import os
import numpy as np
import tensorflow as tf
from model import CNN
from lib.model_io import get_model_id
import read_img as rim

# change this according to your path
path_to_train_set = "/home/tassos/Desktop/DATA_ASR/ASVspoof2017_V2_train_fbank"
path_to_valid_set = "/home/tassos/Desktop/DATA_ASR/ASVspoof2017_V2_train_dev"

model_id = get_model_id()

n_tfiles=350 # how many train files will read
n_vfiles=round(0.567*n_tfiles)
# print("a= \n")
# print(n_vfiles)
# cheat count files number
total_inp_files = len(os.listdir(path_to_train_set))

# Create the network
network = CNN(model_id)
iter=0
for i in range(1,total_inp_files,n_tfiles):
# loop until all data are read
    network.input(n_tfiles,n_vfiles)
    # print(i)
    # rim.read_Data("ASVspoof2017_V2_train_fbank","train_info.txt",n_tfiles)
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
            iter+=1
            # save()
        except KeyboardInterrupt:
            print()

        finally:
            sess.close()

# train finished -remove files

file1 = os.path.basename(path_to_train_set)+"_status.txt"
if(os.path.exists(file1)): os.remove(file1)
file2 = os.path.basename(path_to_valid_set)+"_status.txt"
if(os.path.exists(file2)): os.remove(file2)
