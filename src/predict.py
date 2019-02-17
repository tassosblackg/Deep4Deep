import os
import tensorflow as tf
from model import CNN
from lib.model_io import get_model_id
from lib.model_io import restore_variables
import read_img as rim

model_id = get_model_id()

n_files=150

# Create the network
network = CNN(model_id)
#read DATA
Xeval,Yeval,network.eval_size= rim.read_Data("ASVspoof2017_V2_train_eval","eval_info.txt",n_files)

Xeval=network.normalize(Xeval)  #Normalize eval data
# print(network.eval_size/network.batch_size)

#define placeholders -predict
network.define_predict_operations()

# Recover the parameters of the model
sess = tf.Session()

restore_variables(sess)

indx=0
network.batch_size=64
# Iterate through eval files and calculate the classification scores
# --read data and evaluate for batch_size 64 for all images
for i in range(network.eval_size):  #how many images
   Xbatch,Ybatch,indx=network.read_nxt_batch(Xeval,Yeval,network.batch_size,indx)
   network.predict_utterance(sess,Xbatch,Ybatch)


sess.close()
