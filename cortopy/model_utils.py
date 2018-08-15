__author__ = "Chinmay Rao"

import numpy as np
#import matplotlib.pyplot as plt

def init_params(n_x, n_y, h_units):
    units = h_units
    units.append(n_y)
    L = len(units)
    parameters = {}
    for l in range(0, L):
        if l == 0:
            W = np.random.rand(units[l],n_x) * 0.1
            b = np.zeros((units[l],1))
        else:
            W = np.random.rand(units[l],units[l-1]) * 0.1
            b = np.zeros((units[l],1))
            
        parameters["W"+str(l+1)] = W
        parameters["b"+str(l+1)] = b
              
    return parameters

#print(init_params(2,1,[4,2]))

###################################### ACTIVATION FUNCTIONS ###################
def sigmoid(x, derivative = False):
    sgmd = 1/(1+np.exp(-x))
    if not derivative:
        return sgmd
    return sgmd*(1-sgmd)


def tanh(x, derivative = False):
    th = np.tanh(x)
    if not derivative:
        return th
    return 1-th**2


def relu(x, derivative = False):
    rlu = x*(x>0)
    if not derivative:
        return rlu
    return 1*(x>0)


def softmax(x, derivative = False):
    m = x.shape[1]
    softmax_result = []
    for i in range(m):
        z = x[:,i] - np.max(x[:,i])
        col_softmax_result = np.exp(z) / np.sum(np.exp(z))
        softmax_result.append(col_softmax_result)
    softmax_result =  np.array(softmax_result).T
    
    if not derivative:
        return softmax_result
    return None 
        


def activate(z, act_fn, derivative = False):  # activation function selector
    if act_fn is "relu":
        return relu(z, derivative)
    elif act_fn is "tanh":
        return tanh(z, derivative)
    elif act_fn is "sigmoid":
        return sigmoid(z, derivative)
    elif act_fn is "softmax":
        return softmax(z, derivative)
    
########################################## COST ###############################
        
def cost_calc(y_pred, y_true, cost):
    m = y_pred.shape[1]
    dZ = None
    #print("m: ", m)###
    if cost is "mse":
        J = (1/m) * np.sum( (y_pred - y_true)**2 )
        dA = (2/m) * np.sum((y_pred - y_true), keepdims=True)
                    
    elif cost is "binary_cross_entropy":
        J = (-1.0/m) * np.sum(y_true*np.log(y_pred) + (1-y_true)*np.log(1-y_pred))
        dA = (-1.0/m) * np.sum(y_true/y_pred + (1-y_true)/(1-y_pred), keepdims=True)
            
    elif cost is "softmax_cross_entropy_w_logits":
        epsilon = 1e-12
        y_pred = np.clip(y_pred, epsilon, 1.0 - epsilon)
        J = (-1.0/m) * np.sum(y_true*np.log(y_pred+1e-10))
        dA = (-1.0/m) * np.sum(y_true/y_pred, keepdims=True)
        dZ = y_pred - y_true  #softmax derivative
    
    return J, dA, dZ

########################################### TEST ERROR CALCULATION ############
    
def Test(X_test, Y_test, parameters, act_fn_list, cost):
    L = len(act_fn_list) - 1
    local_cache = [None]
    # forward propagation
    for l in range(1,L+1):
        g = act_fn_list[l]
        if l == 1:
            Z = np.dot(parameters['W1'],X_test) + parameters['b1']
            A = activate(Z, act_fn = g)
        else:
            A_prev = local_cache[l-1][1]
            Z = np.dot(parameters['W'+str(l)], A_prev) + parameters['b'+str(l)]
            A = activate(Z, act_fn = g)
        local_cache.append((Z,A,g))    
    y_pred = A
    # cost 
    J,_,_ = cost_calc(y_pred, Y_test, cost)  
    return J
    