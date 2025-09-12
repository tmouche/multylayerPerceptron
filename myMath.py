
import numpy as np
import math

class myMath:
    
    @staticmethod
    def default(z):
        return z

    # ACTIVATION FUNCTIONS
    @staticmethod
    def sigmoid(z):
        if isinstance(z, float):
            return 1./(1.+math.exp(-z))
        return [1./(1.+math.exp(-x)) for x in z]
    
    @staticmethod
    def reLu(z):
        if isinstance(z, float):
            return z if z > 0. else 0.
        return [x if x > 0. else 0. for x in z]
    
    @staticmethod
    def leakyReLu(z):
        if isinstance(z, float):
            return z if z > 0. else 0.01*z
        return [x if x > 0. else 0.01*x for x in z]
    
    @staticmethod
    def tanh(z):
        if isinstance(z, float):
            return (math.exp(z)-math.exp(-z))/(math.exp(z)+math.exp(-z))
        return [(math.exp(x)-math.exp(-x))/(math.exp(x)+math.exp(-x)) for x in z]
    
    @staticmethod
    def step(z):
        if isinstance(z, float):
            return 1. if z > 0. else 0.
        return [1. if x > 0. else 0. for x in z]
    
    @staticmethod
    def softmax(z):
        if isinstance(z, float):
            raise Exception("Error log: Softmax is in/out vector function and doesnot handle scalar")
        sum_exp_z = sum(np.exp(z))
        return [math.exp(x)/sum_exp_z for x in z]

    # PRIME FUNCTION
    @staticmethod
    def sigmoidPrime(z):
        if isinstance(z, float):
            return myMath.sigmoid(z)*(1.-myMath.sigmoid(z))
        return [myMath.sigmoid(x)*(1.-myMath.sigmoid(x)) for x in z]

    @staticmethod
    def reLuPrime(z):
        if isinstance(z, float):
            return 1. if z > 0. else 0.
        return [1. if x > 0. else 0. for x in z]

    @staticmethod
    def leakyReluPrime(z):
        if isinstance(z, float):
            return 1. if z > 0 else 0.01
        return [1. if x > 0. else 0.01 for x in z]

    @staticmethod
    def tanhPrime(z):
        if isinstance(z, float):
            return 1-math.pow(myMath.tanh(z), 2)
        return [1-math.pow(myMath.tanh(x), 2) for x in z]
    
    @staticmethod
    def stepPrime(z):
        if isinstance(z, float):
            return 0.
        return np.zeros(len(z))
    
    @staticmethod
    def softmaxPrime(z):
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
    
    # RANDOM INITIALIZER
    @staticmethod
    def randomNormal(shape=(1,1), mean=0.0, stddev=0.05):
        return np.random.normal(loc=mean, scale=stddev, size=shape)
    
    @staticmethod
    def randomUniform(shape=(1,1),minval=0., maxval=1.):
        return np.random.uniform(low=minval, high=maxval, size=shape)
        
    @staticmethod
    def zeros(shape=(1,1)):
        return np.zeros(shape)

    @staticmethod
    def ones(shape=(1,1)):
        return np.ones(shape)
    
    @staticmethod
    def xavierNormal(shape=(1,1), fan_in=1, fan_out=1):
        stddev = math.sqrt(2/(fan_in+fan_out))
        return np.random.normal(loc=0, scale=stddev, size=shape)

    @staticmethod
    def xavierUniform(shape=(1,1), fan_in=1, fan_out=1):
        limit = math.sqrt(6/(fan_in+fan_out))
        return np.random.uniform(low=-limit, high=limit, size=shape)
    
    @staticmethod
    def heNormal(shape=(1,1), fan_in=1):
        stddev = math.sqrt(2/fan_in)
        return np.random.normal(loc=0, scale=stddev, size=shape)

    @staticmethod
    def heUniform(shape=(1,1), fan_in=1):
        limit = math.sqrt(6/fan_in)
        return np.random.uniform(low=-limit, high=limit, size=shape)

    # LOSS FUNCTION
    
    def MSE(z: np.array, e: np.array):
        if isinstance(z, float):
            return math.pow(e-z, 2)
        return (1/len(z))/sum([pow(e[i]-z[i], 2) for i in range(len(z))])
    
    def MAE(z: np.array, e: np.array):
        if isinstance(z, float):
            return math.abs(e-z)
        return (1/len(z))/sum([abs(e[i]-z[i]) for i in range(len(z))])
    
    def BCE(z: np.array, e: np.array):
        if isinstance(z, float):
            return -(e*math.log(z)+(1-e)*math.log(1-z))
        return -1/len(z)*sum([e[i]*math.log(z[i])+(1-e[i])*math.log(1-z[i]) for i in range(len(z))])
    
    def hingeLoss(z: np.array, e: np.array):
        if isinstance(z, float):
            return max(0, 1 - e*z)
        return 1/len(z)*sum([max(0, 1-e*z)])
    
    def CCE(z: np.array, e: np.array):
        if isinstance(z, float):
            raise Exception("Error log: Softmax is in/out vector function and doesnot handle scalar")
        return [e[i] - z[i] for i in range(len(z))]
    
    def KLL(z: np.array, e:np.array):
        if isinstance(z, float):
            return e*math.log(e/z)
        return sum([e[i]*math.log(e[i]/z[i]) for i in range(len(z))])
    
    
