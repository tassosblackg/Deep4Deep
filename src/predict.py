import os
import tensorflow as tf
from model import CNN
from lib.model_io import get_model_id
from lib.model_io import restore_variables
import read_img as rim

model_id = get_model_id()


# Create the network
network = CNN(model_id)
#read DATA
Xeval,Yeval,network.eval_size= rim.read_Data("ASVspoof2017_V2_train_eval","eval_info.txt")

Xeval=network.normalize(Xeval)  #Normalize eval data
# print(network.eval_size/network.batch_size)

#define placeholders -predict
network.define_predict_operations()

# Recover the parameters of the model
sess = tf.Session()

restore_variables(sess)



# Iterate through eval files and calculate the classification scores
#read eval files, iteration, score
for i in range(network.eval_size):  #how many images
   network.predict_utterance(sess,Xeval,Yeval)


sess.close()
