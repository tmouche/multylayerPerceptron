
import math
import numpy as np

import ml_tools.activations as Activations

def sigmoid_prime(z):
    if isinstance(z, float):
        return Activations.sigmoid(z)*(1.-Activations.sigmoid(z))
    return [Activations.sigmoid(x)*(1.-Activations.sigmoid(x)) for x in z]

def relu_prime(z):
    if isinstance(z, float):
        return 1. if z > 0. else 0.
    return [1. if x > 0. else 0. for x in z]

def leaky_relu_prime(z):
    if isinstance(z, float):
        return 1. if z > 0 else 0.01
    return [1. if x > 0. else 0.01 for x in z]

def tanh_prime(z):
    if isinstance(z, float):
        return 1-math.pow(Activations.tanh(z), 2)
    return [1-math.pow(Activations.tanh(x), 2) for x in z]

def step_prime(z):
    if isinstance(z, float):
        return 0.
    return np.zeros(len(z))

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
