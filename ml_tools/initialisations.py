
import numpy as np
import math

def random_normal(shape=(1,1), mean=0.0, stddev=0.05) -> np.array:
    return np.random.normal(loc=mean, scale=stddev, size=shape)
    
def random_uniform(shape=(1,1),minval=0., maxval=1.) -> np.array:
    return np.random.uniform(low=minval, high=maxval, size=shape)
        
def zeros(shape=(1,1)) -> np.array:
    return np.zeros(shape)

def ones(shape=(1,1)) -> np.array:
    return np.ones(shape)
    
def xavier_normal(shape=(1,1), fan_in=1, fan_out=1) -> np.array:
    stddev = math.sqrt(2/(fan_in+fan_out))
    return np.random.normal(loc=0, scale=stddev, size=shape)

def xavier_uniform(shape=(1,1), fan_in=1, fan_out=1) -> np.array:
    limit = math.sqrt(6/(fan_in+fan_out))
    return np.random.uniform(low=-limit, high=limit, size=shape)
    
def he_normal(shape=(1,1), fan_in=1) -> np.array:
    stddev = math.sqrt(2/fan_in)
    return np.random.normal(loc=0, scale=stddev, size=shape)

def he_uniform(shape=(1,1), fan_in=1) -> np.array:
    limit = math.sqrt(6/fan_in)
    return np.random.uniform(low=-limit, high=limit, size=shape)
