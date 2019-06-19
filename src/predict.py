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
    # path_to_train_set = "/home/tassos/Desktop/DATA_ASR/ASVspoof2017_V2_train_fbank"
    path_to_eval_set = "/home/tassos/Desktop/DATA_ASR/ASVspoof2017_V2_train_eval"
    total_eval_files = len(os.listdir(path_to_eval_set))
    # total_inp_files = len(os.listdir(path_to_train_set))
    model_id = get_model_id()

    n_tfiles=550

    iter=0



    with tf.device('/gpu:0'):
        tf.reset_default_graph()
        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement = True)) # session with log about gpu exec
        # sess = tf_debug.LocalCLIDebugWrapperSession(sess)

        # Define the train computation graph
        network.define_graph_op()
        init_op = tf.group(tf.global_variables_initializer())
        sess.run(init_op)


        # # ITerate through input data --\SUBSET\

        for i in range(1,total_inp_files,n_tfiles):
        # loop until all data are read

            # mf.input(network,n_tfiles,n_vfiles) # read input data
            mf.eval_input(network,n_tfiles)
            # suppose you've allready run train
            restore_variables(sess)

            print(iter)
            # network.train(sess,writer_train,writer_valid,iter)
            iter += 1
            flag = 0

        sess.close()
except KeyboardInterrupt :
    flag = 1
except Exception :
    traceback.print_exc()
    flag=1
# if(flag==1):
#     file1 = os.path.basename(path_to_eval_set)+"_status.txt"
#     if(os.path.exists(file1)): os.remove(file1) # delete file read status when an error occurs so to restart
