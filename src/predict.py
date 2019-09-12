import os
import tensorflow as tf
from cnn_model import CNN
from lib.model_io import get_model_id
from lib.model_io import restore_variables
import read_img as rim
import model_func as mf
import traceback
from tensorflow.python.tools import inspect_checkpoint as chkp
import shutil as sh


def print_results(network):
    print('_________YOUR SCORE__________    ..\n')
    # f1 = network.f1_score()
    # print(f1)
    numbers = '[true_pos= '+str(network.true_pos)+',true_neg= '+str(network.true_neg)+', false_pos= '+str(network.false_pos)+', false_neg= '+str(network.false_neg) +']'
    print('\n'+numbers+'\n')
    total_n = '>( '+str(network.genuine_counter)+',' +str(network.spoof_counter) +' )<'
    print(total_n)
    input = ' | '+numbers +'// '+total_n+'\n'
    with open('results.txt', 'a') as file:
        file.writelines(input)

# Main code execution
flag=0
try:
    LOGDIR_eval = '../graphs-eval'
    # path_to_train_set = "/home/tassos/Desktop/DATA_ASR/ASVspoof2017_V2_train_fbank"
    path_to_eval_set = "/home/tassos/Desktop/DATA_ASR/ASVspoof2017_V2_train_eval"
    total_eval_files = len(os.listdir(path_to_eval_set))
    # total_inp_files = len(os.listdir(path_to_train_set))
    model_id = get_model_id()

    n_tfiles=1

    iter=0
    network = CNN(model_id)


    with tf.device('/gpu:0'):
        tf.reset_default_graph()
        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement = True)) # session with log about gpu exec
        # sess = tf_debug.LocalCLIDebugWrapperSession(sess)

        # Define the train computation graph
        network.define_predict_operations()
        init_op = tf.group(tf.global_variables_initializer())
        sess.run(init_op)
        if (os.path.exists(LOGDIR_eval)):
            sh.rmtree(LOGDIR_eval)
        writer = tf.summary.FileWriter(LOGDIR_eval, sess.graph)

        # # ITerate through input data --\SUBSET\
        for i in range(1,total_eval_files,n_tfiles):
        # loop until all data are read
            print('i='+str(i)+'\n')
            print(network.std)
            # mf.input(network,n_tfiles,n_vfiles) # read input data
            mf.eval_input(network,n_tfiles)
            # print('@\n')
            # print(network.std)
            # print('@@\n')
            # print(network.Xeval_in)
            # suppose you've allready run train
            if(i==1):restore_variables(sess)

            # print(iter)
            # network.train(sess,writer_train,writer_valid,iter)
            network.predict_utterance(sess,writer,iter)
            # network.make_decision() #take binary decision
            network.calc_f1_sets() # compute counters
            if(i % 700 ==0): print_results(network)
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
