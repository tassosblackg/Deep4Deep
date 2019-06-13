import model_func as mf
import read_img as rim
import tensorflow as tf
n_input =64*17*1
n_classes = 2
x_in,y_in,size = rim.read_Data("ASVspoof2017_V2_train_fbank", "train_info.txt",n_tfiles)

X = tf.placeholder(dtype=tf.float32, shape=(None,n_input),name='X_train')
Y = tf.placeholder(dtype=tf.int32,shape=(None, n_classes), name='Y_train'))

out_x = X

With tf.Session() as sess:
    print(sess.run(out_x,feed_dict={X:x_in}))
    # print()
