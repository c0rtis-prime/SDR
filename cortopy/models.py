__author__ = "Chinmay Rao"

import numpy as np
import pickle
import math
import matplotlib.pyplot as plt
from cortopy.model_utils import *
#from model_utils import *
from cortopy import optimizers
#import optimizers

class dense_model():
    def __init__(self, X, Y, hidden_units, act_fn_list, cost):
        self.hidden_units = hidden_units
        self.cost = cost
        self.act_fn_list = act_fn_list
        self.act_fn_list.insert(0,None)
        self.parameters = init_params(X.shape[0], Y.shape[0], hidden_units)    #PARAMETERS = {W1:[], b1:[], ....}
        self.cache = [None]                                                    #cache = [None, (Z1,A1,'relu'),(Z2,A2,'relu'), .....]
        #self.gradients = [None]                                               
        
    def train(self, X_train, Y_train, X_test, Y_test, learning_rate, batch_size, epochs, optimizer, momentum_beta = 0.9, rmsprop_beta = 0.999):
        #print("self.hidden_units",self.hidden_units)##
        #print("Training")
        training_error_list = []
        test_error_list = []
        L = len(self.hidden_units) #+ 1               
        n_batches = math.floor(Y_train.shape[1] / batch_size)
        #print("no. of batches : ",n_batches)## 
        Velocities = [None]  #for momentum_GD                                  #Velocities = [None, (V_dW1,V_db1), (V_dW1,V_db1), .....]              
        rmsprop_S = [None]  #for RMS_prop                                     #rmsprop_S = [None, (S_dW1,S_db1), (S_dW2,S_db2), .....]
        for epoch in range(0,epochs):           
            for t in range(1,n_batches+1):
                print("TRAINING - Epoch: {}, Batch: {}".format(epoch,t))#######
                x_ix = batch_size*(t-1)
                y_ix = batch_size*(t-1)
                X_train_batch, Y_train_batch = X_train[:, x_ix : x_ix + batch_size], Y_train[:, y_ix : y_ix + batch_size]
                ###############################################################          
                # forward propagation
                for l in range(1,L+1):
                    #print("layer: ", l)
                    g = self.act_fn_list[l]
                    if l == 1:
                        Z = np.dot(self.parameters['W1'],X_train_batch) + self.parameters['b1']
                        A = activate(Z, act_fn = g)
                    else:
                        A_prev = self.cache[l-1][1]
                        Z = np.dot(self.parameters['W'+str(l)], A_prev) + self.parameters['b'+str(l)]
                        A = activate(Z, act_fn = g)
                    self.cache.append((Z,A,g))
                #y_pred = self.cache[L][1]
                y_pred = A
                      
                # cost               
                J, dA, dZ = cost_calc(y_pred, Y_train_batch, self.cost)
                
                #print("Cost: {}, batch: {}, epoch: {}".format(J,t,epoch))
                      
                # back propagation
                if optimizer is "minibatch_GD":
                    self.parameters, self.cache = optimizers.minibatch_GD(X_train_batch, self.parameters, self.cache, L, learning_rate, batch_size, dA, dZ)
                elif optimizer is "momentum_GD":
                    self.parameters, self.cache, Velocities = optimizers.momentum_GD(Velocities, t, momentum_beta, 
                                                                                     X_train_batch, self.parameters, self.cache, L, learning_rate, batch_size, dA, dZ)
                elif optimizer is "RMS_prop":
                    self.parameters, self.cache, rmsprop_S = optimizers.RMS_prop(rmsprop_S, rmsprop_beta, t,
                                                                        X_train_batch, self.parameters, self.cache, L, learning_rate, batch_size, dA, dZ)
                elif optimizer is "ADAM":
                    self.parameters, self.cache, Velocities, rmsprop_S = optimizers.ADAM(Velocities, rmsprop_S, momentum_beta, rmsprop_beta, t,
                                                                                          X_train_batch, self.parameters, self.cache, L, learning_rate, batch_size, dA, dZ)
                ###############################################################
            training_error_list.append(J)
            test_error = Test(X_test,Y_test, self.parameters, self.act_fn_list, self.cost)
            test_error_list.append(test_error)        
       
        
        plt.plot(training_error_list)
        plt.plot(test_error_list)
        plt.xlabel("epochs")
        plt.ylabel("loss")
        plt.legend(["Training error","Test error"], loc="upper right")
        plt.title("Learning rate:"+ str(learning_rate)+", Optimizer:"+ str(optimizer))
        plt.savefig("results/Loss_plot_[optmzr={}]_[lr={}].png".format(optimizer,learning_rate))
        plt.show()
        
    def predict(self, X_sample): 
        L = len(self.hidden_units)
        # forward propagation
        local_cache = [None]
        #print("L: ",L)
        for l in range(1,L+1):
           # print("l: {}, local_cache size: {}".format(l,len(local_cache)))
            g = self.act_fn_list[l]
            if l == 1:
                Z = np.dot(self.parameters['W1'],X_sample) + self.parameters['b1']
                A = activate(Z, act_fn = g)
            else:
                A_prev = local_cache[l-1][1]
                Z = np.dot(self.parameters['W'+str(l)], A_prev) + self.parameters['b'+str(l)]
                A = activate(Z, act_fn = g)
            local_cache.append((Z,A,g))
        #y_pred = local_cache[L][1]
        y_pred = A        
        return y_pred
                
    def save_weights(self, file_path):
        with open(file_path,'wb') as f:
            pickle.dump(self.parameters, f)
            
    def load_weights(self, file_path):
        with open(file_path,'rb') as f:
            self.parameters = pickle.load(f,encoding='bytes')
