import os
import tensorflow as tf
from model import CNN
from lib.model_io import get_model_id
from lib.model_io import restore_variables


model_id = get_model_id()


# Create the network
network = CNN(model_id)
network.define_predict_operations()

# Recover the parameters of the model
sess = tf.Session()

restore_variables(sess, cfg)



# Iterate through eval files and calculate the classification scores
#read eval files, iteration, score

sess.close()