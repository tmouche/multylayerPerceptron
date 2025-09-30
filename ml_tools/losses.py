
import numpy as np
import math

def mean_square_error(z: np.array, e: np.array):
    if isinstance(z, float):
        return math.pow(e-z, 2)
    return (1/len(z))/sum([pow(e[i]-z[i], 2) for i in range(len(z))])
    
def mean_absolute_error(z: np.array, e: np.array):
    if isinstance(z, float):
        return math.abs(e-z)
    return (1/len(z))/sum([abs(e[i]-z[i]) for i in range(len(z))])
    
def binary_cross_entropy(z: np.array, e: np.array):
    if isinstance(z, float):
        return -(e*math.log(z)+(1-e)*math.log(1-z))
    return -1/len(z)*sum([e[i]*math.log(z[i])+(1-e[i])*math.log(1-z[i]) for i in range(len(z))])
    
def hinge_loss(z: np.array, e: np.array):
    if isinstance(z, float):
        return max(0, 1 - e*z)
    return 1/len(z)*sum([max(0, 1-e*z)])
    
def categorical_cross_entropy(z: np.array, e: np.array):
    if isinstance(z, float):
        raise Exception("Error log: Softmax is in/out vector function and doesnot handle scalar")
    return [e[i] - z[i] for i in range(len(z))]
    
def kullback_leibler_divergence(z: np.array, e:np.array):
    if isinstance(z, float):
        return e*math.log(e/z)
    return sum([e[i]*math.log(e[i]/z[i]) for i in range(len(z))])
