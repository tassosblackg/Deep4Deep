import os
import numpy as np
import tensorflow as tf
from model import CNN
from lib.model_io import get_model_id

# change this according to your path
path_to_train_set = "/home/tassos/Desktop/DATA_ASR/ASVspoof2017_V2_train_fbank"
path_to_valid_set = "/home/tassos/Desktop/DATA_ASR/ASVspoof2017_V2_train_dev"

model_id = get_model_id()

n_tfiles=150 # how many train files will read
n_vfiles=round(0,567*n_tfiles)

# cheat count files number
total_inp_files = len(os.listdir(path_to_train_set))

# Create the network
network = CNN(model_id)

for i in range(total_inp_files):
# loop until all data are read
    network.input(n_files)

    with tf.device("/gpu0:"):
        # restore()

        # Define the train computation graph
        network.define_train_operations()


        # Train the network
        sess = tf.Session(config=tf.ConfigProto(log_device_placement=True)) #session with log about gpu exec
        try:
            network.train(sess,i)
            # save train
            # save()
        except KeyboardInterrupt:
            print()
        finally:
            sess.close()
