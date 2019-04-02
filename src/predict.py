import os
import tensorflow as tf
from model import CNN
from lib.model_io import get_model_id
from lib.model_io import restore_variables
import read_img as rim
import traceback
from tensorflow.python.tools import inspect_checkpoint as chkp


flag=0
try:
    path_to_eval_set = "/home/tassos/Desktop/DATA_ASR/ASVspoof2017_V2_train_eval"
    total_eval_files = len(os.listdir(path_to_eval_set))
    model_id = get_model_id()

    n_files=550
# create network
model_id = get_model_id()
network = CNN(model_id)
network.define_predict_operations()

sess = tf.Session()

restore_variables(sess,cfg)

except Exception :
    traceback.print_exc()
    flag=1
if(flag==1):
    file1 = os.path.basename(path_to_eval_set)+"_status.txt"
    if(os.path.exists(file1)): os.remove(file1) # delete file read status when an error occurs so to restart
