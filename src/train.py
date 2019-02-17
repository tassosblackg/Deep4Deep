import os
import numpy as np
import tensorflow as tf
from model import CNN
from lib.model_io import get_model_id

model_id = get_model_id()

n_files=150 # how many files will read
# Create the network
network = CNN(model_id)

for i in range():
# loop until all data are read
    network.input(n_files)

    with tf.device("/gpu0:"):
        # restore()

        # Define the train computation graph
        network.define_train_operations()


        # Train the network
        sess = tf.Session(config=tf.ConfigProto(log_device_placement=True)) #session with log about gpu exec
        try:
            network.train(sess)
            # save train
            # save()
        except KeyboardInterrupt:
            print()
        finally:
            sess.close()
