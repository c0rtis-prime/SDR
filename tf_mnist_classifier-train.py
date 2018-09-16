import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math
from load_data import load_data


######################################## UTILITIES ############################
def weight_var(shape, l_id):
    with tf.variable_scope(l_id):
        return tf.get_variable('weights', shape=shape,
                               initializer=tf.truncated_normal_initializer(stddev=0.05) )

def bias_var(shape, l_id):
     with tf.variable_scope(l_id):
        return tf.get_variable('bias', shape=shape,
                                initializer=tf.truncated_normal_initializer(stddev=0.05))

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides = [1,1,1,1], padding = 'SAME')

def max_pool(x):
  pooled =  tf.nn.max_pool(x, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')
  return pooled

def conv_layer(ip, w_shape, l_id):
  W = weight_var(w_shape, l_id)
  b = bias_var([w_shape[3]], l_id)
  return tf.nn.relu(conv2d(ip,W) + b)

def full_layer(ip, size, l_id):
  in_size = int(ip.get_shape()[1])
  W = weight_var([in_size, size], l_id)
  b = bias_var([size], l_id)
  return tf.matmul(ip,W) + b
###############################################################################

def get_next_idx(batch, batch_size):
    return (batch-1)*batch_size
  
######################################### DEFINITION ##########################

X = tf.placeholder(tf.float32, shape = [None,784], name="X")
y_true = tf.placeholder(tf.float32, shape = [None, 10], name="Y")

X_image = tf.reshape(X, [-1,28,28,1])

conv_1 = conv_layer(X_image, [5,5,1,32], l_id='conv_layer_1-params')
conv_1_pooled = max_pool(conv_1)
conv_2 = conv_layer(conv_1_pooled, [5,5,32,64], l_id='conv_layer_2-params')
conv_2_pooled = max_pool(conv_2)

flattened = tf.reshape(conv_2_pooled, [-1, 7*7*64])
full_conn_layer_1 = tf.nn.relu(full_layer(flattened, 1024, l_id='fc_1-params'))
keep_prob = tf.placeholder(tf.float32, name="keep_prob")
full_1_drop = tf.nn.dropout(full_conn_layer_1, keep_prob=keep_prob) 

y_logits = full_layer(full_1_drop, 10, l_id='fc_2-params')
y_pred = tf.nn.softmax(y_logits)
###############################################################################

cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_logits))
optimizer = tf.train.AdamOptimizer(1e-4)
train_step = optimizer.minimize(cross_entropy_loss)

correct_prediction = tf.equal(tf.argmax(y_pred,1), tf.argmax(y_true,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

######################## TensorBoard particulars ##############################
loss_summary = tf.summary.scalar("loss", cross_entropy_loss)
accuracy_summary = tf.summary.scalar("accuracy", accuracy)

###############################################################################

X_train, Y_train_enc, X_test, Y_test_enc = load_data()
X_train, Y_train_enc = X_train.T, Y_train_enc.T.astype(float)
X_test = None
Y_test_enc = None

################################# Hyper parameters ############################
batch_size = 50
n_batches = math.floor(Y_train_enc.shape[0] / batch_size)
epochs = 1
###############################################################################

sess = tf.Session()
sess.run(tf.global_variables_initializer())

TB_writer = tf.summary.FileWriter("summary_dir", sess.graph)

loss_list = []

for e in range(epochs):    
    for t in range(1,n_batches):
        idx = get_next_idx(t, batch_size)
        X_train_batch, Y_train_batch = X_train[idx : idx + batch_size, :], Y_train_enc[idx : idx + batch_size, :]
        if t%100 == 0:
            train_accuracy, train_accuracy_summary, train_loss_summary = sess.run([accuracy, accuracy_summary, loss_summary], 
										  feed_dict = { X : X_train_batch , y_true : Y_train_batch , keep_prob : 1.0})
            TB_writer.add_summary(train_accuracy_summary,t)
            TB_writer.add_summary(train_loss_summary,t)
            print("Epoch: {}, Batch: {}, Train accuracy: {} ".format(e, t, train_accuracy))
            #print("Epoch: {}, Batch: {}".format(e, t))        
        #loss_list.append(sess.run(cross_entropy_loss, feed_dict = { X : X_train_batch, y_true : Y_train_batch, keep_prob : 0.5}) )
  
        sess.run( train_step, feed_dict = { X : X_train_batch, y_true : Y_train_batch, keep_prob : 0.5} )                                

#test_accuracy = np.mean(sess.run(accuracy, feed_dict = { X : X_train[9000:10000], y_true : Y_train_enc[9000:10000], keep_prob : 1.0}) )
#print(test_accuracy)

'''
##################################   SAVE MODEL ###############################
saver = tf.train.Saver()
save_path = saver.save(sess, "results/tf_mnist-5/tf_model.ckpt")

plt.plot(loss_list)
plt.show()
'''

sess.close()
