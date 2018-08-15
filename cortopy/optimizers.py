__author__ = "Chinmay Rao"

import numpy as np
from cortopy.model_utils import *
#from model_utils import *


def minibatch_GD(X_train_batch, parameters, cache, L, learning_rate, batch_size, dA, dZ=None):
    #print("back-prop")
    m = batch_size
    for l in range(L,0,-1):       
        g = cache[l][2]        
        if g is not "softmax":
            dZ = dA * activate(cache[l][0], act_fn = g, derivative = True)                 
        if l > 1:
            dW = (1/m) * np.dot(dZ, cache[l-1][1].T)
        else:
            dW = (1/m) * np.dot(dZ, X_train_batch.T)
        db = (1/m) * np.sum(dZ, axis=1, keepdims=True)
        dA = np.dot(parameters['W'+str(l)].T, dZ)
        
        parameters['W'+str(l)] -= learning_rate*dW
        parameters['b'+str(l)] -= learning_rate*db        
    cache = [None]    
    return parameters, cache


def momentum_GD(Velocities, batch, momentum_beta, X_train_batch, parameters, cache, L, learning_rate, batch_size, dA, dZ=None):
    m = batch_size
    if batch == 1:
        for l in range(1,L+1):
            zeros_dW = np.zeros_like(parameters['W'+str(l)])
            zeros_db = np.zeros_like(parameters['b'+str(l)])
            Velocities.append(  [zeros_dW, zeros_db]  )    
    for l in range(L,0,-1):       
        g = cache[l][2]        
        if g is not "softmax":
            dZ = dA * activate(cache[l][0], act_fn = g, derivative = True)                 
        if l > 1:
            dW = (1/m) * np.dot(dZ, cache[l-1][1].T)
        else:
            dW = (1/m) * np.dot(dZ, X_train_batch.T)
        db = (1/m) * np.sum(dZ, axis=1, keepdims=True)
        dA = np.dot(parameters['W'+str(l)].T, dZ)       
        
        Velocities[l][0] = momentum_beta * Velocities[l][0] + (1-momentum_beta)*dW
        Velocities[l][1] = momentum_beta * Velocities[l][1] + (1-momentum_beta)*db               
        
        parameters['W'+str(l)] -= learning_rate * Velocities[l][0]
        parameters['b'+str(l)] -= learning_rate * Velocities[l][1]        
    cache = [None]
    return parameters, cache, Velocities



def RMS_prop(S, rmsprop_beta, batch, X_train_batch, parameters, cache, L, learning_rate, batch_size, dA, dZ=None):
    m = batch_size
    if batch == 1:
        for l in range(1,L+1):
            zeros_dW = np.zeros_like(parameters['W'+str(l)])
            zeros_db = np.zeros_like(parameters['b'+str(l)])
            S.append(  [zeros_dW, zeros_db]  )            
    for l in range(L,0,-1):       
        g = cache[l][2]        
        if g is not "softmax":
            dZ = dA * activate(cache[l][0], act_fn = g, derivative = True)                 
        if l > 1:
            dW = (1/m) * np.dot(dZ, cache[l-1][1].T)
        else:
            dW = (1/m) * np.dot(dZ, X_train_batch.T)
        db = (1/m) * np.sum(dZ, axis=1, keepdims=True)
        dA = np.dot(parameters['W'+str(l)].T, dZ)       
                      
        S[l][0] = rmsprop_beta * S[l][0] + (1-rmsprop_beta)*(dW**2)
        S[l][1] = rmsprop_beta * S[l][1] + (1-rmsprop_beta)*(db**2)               
        
        parameters['W'+str(l)] -= learning_rate * dW/(np.sqrt(S[l][0]) + 1e-8)
        parameters['b'+str(l)] -= learning_rate * db/(np.sqrt(S[l][1]) + 1e-8)
    cache = [None]
    return parameters, cache, S


def ADAM(Velocities, S, momentum_beta, rmsprop_beta, batch, X_train_batch, parameters, cache, L, learning_rate, batch_size, dA, dZ=None):
    m = batch_size
    if batch == 1:
        for l in range(1,L+1):
            zeros_dW = np.zeros_like(parameters['W'+str(l)])
            zeros_db = np.zeros_like(parameters['b'+str(l)])
            Velocities.append(  [zeros_dW, zeros_db]  )
            S.append(  [zeros_dW, zeros_db]  )            
    for l in range(L,0,-1):       
        g = cache[l][2]        
        if g is not "softmax":
            dZ = dA * activate(cache[l][0], act_fn = g, derivative = True)                 
        if l > 1:
            dW = (1/m) * np.dot(dZ, cache[l-1][1].T)
        else:
            dW = (1/m) * np.dot(dZ, X_train_batch.T)
        db = (1/m) * np.sum(dZ, axis=1, keepdims=True)
        dA = np.dot(parameters['W'+str(l)].T, dZ)       
        
        Velocities[l][0] = momentum_beta * Velocities[l][0] + (1-momentum_beta)*dW
        Velocities[l][1] = momentum_beta * Velocities[l][1] + (1-momentum_beta)*db               
        S[l][0] = rmsprop_beta * S[l][0] + (1-rmsprop_beta)*(dW**2)
        S[l][1] = rmsprop_beta * S[l][1] + (1-rmsprop_beta)*(db**2)               
        '''
        Velocities[l][0] /= (1-momentum_beta**batch)  # bias corrections
        Velocities[l][1] /= (1-momentum_beta**batch)  #
        S[l][0] /= (1-rmsprop_beta**batch)            #
        S[l][1] /= (1-rmsprop_beta**batch)            #
        '''
        parameters['W'+str(l)] -= learning_rate * Velocities[l][0]/(np.sqrt(S[l][0]) + 1e-8)
        parameters['b'+str(l)] -= learning_rate * Velocities[l][1]/(np.sqrt(S[l][1]) + 1e-8)
    cache = [None]
    return parameters, cache, Velocities, S