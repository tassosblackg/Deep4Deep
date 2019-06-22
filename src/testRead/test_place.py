import model_func as mf
import read_img as rim
import tensorflow as tf
n_tfiles = 10
n_input =64*17*1
n_classes = 2
x_in,y_in,size = rim.read_Data("ASVspoof2017_V2_train_fbank", "train_info.txt",n_tfiles)
#rim.read_Data("ASVspoof2017_V2_train_dev", "dev_info.txt",n_vfiles)
X = tf.placeholder(dtype=tf.float32, shape=(None,n_input),name='X_train')
Y = tf.placeholder(dtype=tf.float32,shape=(None, n_classes), name='Y_train')

out_x = Y

out_y = tf.nn.softmax(Y)
with tf.Session() as sess:
    print(sess.run(out_y,feed_dict={Y:y_in}))
    # print()
