
import numpy as np
import math

def sigmoid(z):
    if isinstance(z, float):
        return 1./(1.+math.exp(-z))
    return [1./(1.+math.exp(-x)) for x in z]

def relu(z):
    if isinstance(z, float):
        return z if z > 0. else 0.
    return [x if x > 0. else 0. for x in z]

def leaky_relu(z):
    if isinstance(z, float):
        return z if z > 0. else 0.01*z
    return [x if x > 0. else 0.01*x for x in z]

def tanh(z):
    if isinstance(z, float):
        return (math.exp(z)-math.exp(-z))/(math.exp(z)+math.exp(-z))
    return [(math.exp(x)-math.exp(-x))/(math.exp(x)+math.exp(-x)) for x in z]

def step(z):
    if isinstance(z, float):
        return 1. if z > 0. else 0.
    return [1. if x > 0. else 0. for x in z]

def softmax(z):
    if isinstance(z, float):
        raise Exception("Error log: Softmax is in/out vector function and doesnot handle scalar")
    sum_exp_z = sum(np.exp(z))
    return [math.exp(x)/sum_exp_z for x in z]
