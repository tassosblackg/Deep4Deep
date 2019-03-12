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
    # chkp.print_tensors_in_checkpoint_file("../Variables3/cnn-30",tensor_name='inference/Dense-Layer',all_tensors=False)
    network=CNN(model_id)
    with tf.device('/gpu:0'):
        session = tf.Session(config=tf.ConfigProto(allow_soft_placement = True))
        tf_saver = tf.train.import_meta_graph('../Variables3/cnn-30.meta')
        tf_saver.restore(session, tf.train.latest_checkpoint('../Variables3/','checkpoint'))

        x      = tf.placeholder(tf.float32, shape=[None, network.height, network.width, network.chan], name='x')
        y_true = tf.placeholder(tf.float32, shape=[None, network.n_classes], name='y_true')
        y_true_cls = tf.argmax(y_true, axis=1)

        # y_pred     = tf.nn.softmax(layer_fc2, name='y_pred')
        # y_pred_cls = tf.argmax(y_pred, axis=1)
        #
        # correct_prediction = tf.equal(y_pred_cls, y_true_cls)
        # accuracy           = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        #
        # feed_dict_test  = {x: images, y_true: labels}
        #
        # test_acc = session.run(accuracy, feed_dict=feed_dict_test)
        #
        # msg     = "Test Accuracy: {1:>6.1%}"
        # print(msg.format(test_acc))

except Exception :
    traceback.print_exc()
    flag=1
if(flag==1):
    file1 = os.path.basename(path_to_eval_set)+"_status.txt"
    if(os.path.exists(file1)): os.remove(file1) # delete file read status when an error occurs so to restart
