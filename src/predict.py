import os
import tensorflow as tf
from cnn_model import CNN
from lib.model_io import get_model_id
from lib.model_io import restore_variables
import read_img as rim
import model_func as mf
import traceback
from tensorflow.python.tools import inspect_checkpoint as chkp


flag=0
try:
    path_to_train_set = "/home/tassos/Desktop/DATA_ASR/ASVspoof2017_V2_train_fbank"
    # path_to_eval_set = "/home/tassos/Desktop/DATA_ASR/ASVspoof2017_V2_train_eval"
    # total_eval_files = len(os.listdir(path_to_eval_set))
    total_inp_files = len(os.listdir(path_to_train_set))
    model_id = get_model_id()

    n_tfiles=550
    n_vfiles=round(0.25*n_tfiles)
    # create network
    model_id = get_model_id()
    network = CNN(model_id)
    network.define_train_operations()
    for i in range(0,total_inp_files,n_tfiles):

        mf.input(network,n_tfiles,n_vfiles)
        with tf.device('/gpu:0'):

            sess = tf.Session(config=tf.ConfigProto(allow_soft_placement = True))

            restore_variables(sess) # load last chkp
            network.evaluate(sess)
            try:

                train_acc = self.evaluate(sess,train=True)
                valid_acc = self.evaluate(sess,train=False)
                print('[**  train_acc ={:.3f}' .format(train_acc)+'valid_acc ={:.3f}'.format(valid_acc) +' **]\n')
            except Exception as e:
                print('-7-\n')
                traceback.print_exc()
except Exception :
    traceback.print_exc()
    flag=1
# if(flag==1):
#     file1 = os.path.basename(path_to_eval_set)+"_status.txt"
#     if(os.path.exists(file1)): os.remove(file1) # delete file read status when an error occurs so to restart
