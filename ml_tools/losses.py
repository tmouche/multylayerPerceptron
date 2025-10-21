
import numpy as np
import math

def mean_square_error(y: np.array, e: np.array):
    if isinstance(y, float):
        return math.pow(e-y, 2)
    return (1/len(y))/sum([pow(e[i]-y[i], 2) for i in range(len(y))])
    
def mean_absolute_error(y: np.array, e: np.array):
    if isinstance(y, float):
        return math.abs(e-y)
    return (1/len(y))/sum([abs(e[i]-y[i]) for i in range(len(y))])
    
def binary_cross_entropy(y:np.array, e:np.array):
    if isinstance(y, float):
        return -(e*math.log(y)+(1-e)*math.log(1-y))
    return -1/len(y)*sum([e[i]*math.log(y[i])+(1-e[i])*math.log(1-y[i]) for i in range(len(y))])
    
def hinge_loss(y: np.array, e: np.array):
    if isinstance(y, float):
        return max(0, 1 - e*y)
    return 1/len(y)*sum([max(0, 1-e*y)])
    
def categorical_cross_entropy(y: np.array, e: np.array):
    if isinstance(y, float):
        raise Exception("Error log: Softmax is in/out vector function and doesnot handle scalar")
    return [e[i] - y[i] for i in range(len(y))]
    
def kullback_leibler_divergence(y: np.array, e:np.array):
    if isinstance(y, float):
        return e*math.log(e/y)
    return sum([e[i]*math.log(e[i]/y[i]) for i in range(len(y))])

