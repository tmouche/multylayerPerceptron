
import math
import numpy as np

def sigmoid_prime(y):
    y = np.array(y)
    return y*(1.-y)

def relu_prime(y):
    if isinstance(y, float):
        return 1. if y > 0. else 0.
    return [1. if x > 0. else 0. for x in y]

def leaky_relu_prime(y):
    if isinstance(y, float):
        return 1. if y > 0 else 0.01
    return [1. if x > 0. else 0.01 for x in y]

def tanh_prime(y):
    if isinstance(y, float):
        return 1-math.pow(y, 2)
    return [1-math.pow(x, 2) for x in y]

def softmax_prime(z):
    if isinstance(z, float):
        raise Exception("Error log: Softmax is in/out vector function and doesnot handle scalar")
    R = len(z)
    jacobian = np.zeros((R, R))
    for i in range(R):
        for j in range(R):
            if i == j:
                jacobian[i,j] = z[i]*(1-z[i])
            else:
                jacobian[i,j] = -z[i]*z[j]
    return jacobian
