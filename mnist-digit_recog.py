import numpy as np
import tensorflow as tf
import cv2
import cortopy.models as models


X_train = np.zeros([784,60000])
Y_train_enc = np.zeros([10,60000])


####################### CORTOPY classifier ####################################
hidden_units = [512,512]
ctpy_classifier = models.dense_model( X_train, Y_train_enc, 
                                      hidden_units, 
                                      act_fn_list=['relu','relu','softmax'], 
                                      cost="softmax_cross_entropy_w_logits" )

ctpy_classifier.load_weights("results/mnist-3/mnist-weights_[optmzr=RMS_prop]_[lr=0.0003]")

X_train = None
Y_train_enc = None

####################### TF classifier #########################################

sess = tf.Session()

saver = tf.train.import_meta_graph('results/tf_mnist-5/tf_model.ckpt.meta')
saver.restore(sess, "results/tf_mnist-5/tf_model.ckpt")

###############################################################################

def predict(classifier,img_array):
    if classifier is "ctpy_classifier":
        return np.argmax(ctpy_classifier.predict(img_array))
    elif classifier is "tf_classifier":
        graph = tf.get_default_graph()
        X = graph.get_tensor_by_name("X:0")
        Y = graph.get_tensor_by_name("Y:0")
        keep_prob = graph.get_tensor_by_name("keep_prob:0")
        Y_dummy = np.zeros([1,10])
        prediction = graph.get_tensor_by_name("add_3:0")
        
        img_array = img_array.T
        return np.argmax( sess.run(prediction, feed_dict= {X:img_array, Y:Y_dummy, keep_prob:1.0 }) )

cam = cv2.VideoCapture(0)

while(cam.isOpened()):
    #x, y, w, h = 0, 0, 300, 300
    x, y, w, h = 424-150, 240-150, 300, 300
    
    ret, img = cam.read()
    
    prediction = ''
    
    if ret:
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        scratch_area = img[y:y+h, x:x+h]
        scratch_area = cv2.cvtColor(scratch_area, cv2.COLOR_BGR2GRAY)
        
        _, scratch_area = cv2.threshold(scratch_area,25,255, cv2.THRESH_BINARY_INV)
        
        input_img = cv2.resize(scratch_area, (28,28))
        input_img = np.array(input_img).reshape(784,1)
        
        ctpy_prediction = predict("ctpy_classifier",input_img)
        tf_prediction = predict("tf_classifier",input_img)
        
        cv2.putText(img, 
                    "DNN prediction: "+ str(ctpy_prediction), 
                    (x,420), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1, 
                    (0, 255, 0), 
                    2)
        cv2.putText(img, 
                    "CNN prediction: " + str(tf_prediction), 
                    (x,460), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1, 
                    (0, 255, 0), 
                    2)
        
        cv2.imshow("Frame", img)
        cv2.imshow("scratch area", scratch_area)
        
        k = cv2.waitKey(1)
        if k == ord('q'):
            break

cam.release()
cv2.destroyAllWindows()
