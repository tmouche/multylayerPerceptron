
import numpy as np
import math

EPS = 1e-8

def mean_square_error(y: np.array, e: np.array):
    if isinstance(y, float):
        return math.pow(e-y, 2)
    return (1/len(y))*sum([pow(e[i]-y[i], 2) for i in range(len(y))])
    
def mean_absolute_error(y: np.array, e: np.array):
    if isinstance(y, float):
        return math.abs(e-y)
    return (1/len(y))*sum([abs(e[i]-y[i]) for i in range(len(y))])
    
def binary_cross_entropy(y:np.array, e:np.array):
    if isinstance(y, float):
        return -(e*math.log(y+EPS)+(1-e)*math.log(1-y+EPS))
    return -1/len(y)*sum([e[i]*math.log(y[i]+EPS)+(1-e[i])*math.log(1-y[i]+EPS) for i in range(len(y))])
    
def spare_cross_entropy(y:np.array, e:np.array):
    if isinstance(y, float):
        raise Exception("Error log: spare cross entropy doesnot handle scalar")
    return -math.log(y[np.argmax(e)])

def categorical_cross_entropy(y: np.array, e: np.array):
    if isinstance(y, float):
        raise Exception("Error log: Softmax is in/out vector function and doesnot handle scalar")
    return -sum([e[i]*math.log(y[i]+EPS) for i in range(len(y))])
    
def kullback_leibler_divergence(y: np.array, e:np.array):
    if isinstance(y, float):
        return e*math.log(e/y+EPS)
    return sum([e[i]*math.log(e[i]/y[i]+EPS) for i in range(len(y))])

