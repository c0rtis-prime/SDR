import tensorflow as tf
import numpy as np
from load_data import load_data

X_train, Y_train_enc, X_test, Y_test_enc = load_data()
X_test, Y_test_enc = X_test.T, Y_test_enc.T.astype(float)
X_train = None
Y_train_enc = None

X_test = X_test[1:2]
Y_test_enc = Y_test_enc[1:2]

sess = tf.Session()

saver = tf.train.import_meta_graph('results/tf_mnist-5/tf_model.ckpt.meta')
saver.restore(sess, "results/tf_mnist-5/tf_model.ckpt")

graph = tf.get_default_graph()
X = graph.get_tensor_by_name("X:0")
Y = graph.get_tensor_by_name("Y:0")
keep_prob = graph.get_tensor_by_name("keep_prob:0")
feed_dict = {X:X_test, Y:Y_test_enc, keep_prob:1.0 }
prediction = graph.get_tensor_by_name("add_3:0")
#accuracy = graph.get_tensor_by_name("Mean_6:0")

print( sess.run(prediction, feed_dict) )

sess.close()