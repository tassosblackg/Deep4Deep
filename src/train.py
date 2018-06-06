import os
import numpy as np
import tensorflow as tf
from model import CNN
from lib.model_io import get_model_id

model_id = get_model_id()




# Create the network
network = CNN(model_id)
network.input()

# Define the train computation graph
network.define_train_operations()

# Train the network
sess = tf.Session()
try:
    network.train(sess)
except KeyboardInterrupt:
    print()
finally:
    sess.close()
