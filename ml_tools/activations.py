
import math

import numpy as np
import ml_tools.losses as Losses

from abc import ABC, abstractmethod


from utils.logger import Logger
logger = Logger()

class __Activation(ABC):

    delta = None
    loss = None

    def __init__(self, activation_function_name:str, loss_function_name:str):
        try:
            self.loss = getattr(Losses, loss_function_name)
            self.delta = getattr(self, ("_" + loss_function_name + "_grad"))
        except AttributeError:
            logger.error(f"{loss_function_name} can not be associated with {activation_function_name}")
            raise Exception()

    @staticmethod
    @abstractmethod
    def activation(z):
        pass

    @staticmethod
    @abstractmethod
    def prime(y):
        pass



class sigmoid(__Activation):
    def __init__(self, loss_function_name:str):
        super().__init__(self.__class__.__qualname__, loss_function_name)

    @staticmethod
    def activation(z):
        if isinstance(z, float):
            return 1./(1.+math.exp(-z))
        z = np.clip(z, -500, 500)
        return [1./(1.+math.exp(-x)) for x in z]

    @staticmethod
    def prime(y):
        y = np.array(y)
        return y*(1.-y)
    
    def _mean_square_error_grad(self, y, e):
        loss = np.array(y) - np.array(e)
        prime = self.prime(y)
        return loss * prime
    
    def _mean_absolute_error_grad(self, y, e):
        loss = abs(np.array(y) - np.array(e))
        prime = self.prime(y)
        return loss * prime
    
    def _binary_cross_entropy_grad(self, y, e):
        loss = np.array(y) - np.array(e)
        return loss
    
    def _categorical_cross_entropy_grad(self, y, e):
        loss = np.array(y) - np.array(e)
        return loss

    def _spare_cross_entropy_grad(self, y, e):
        loss = np.array(y) - np.array(e)
        return loss


class tanh(__Activation):
    def __init__(self, loss_function_name:str):
        super().__init__(self.__class__.__qualname__, loss_function_name)

    @staticmethod
    def activation(z):
        if isinstance(z):
            return (math.exp(z)-math.exp(-z))/(math.exp(z)+math.exp(-z))
        return [(math.exp(x)-math.exp(-x))/(math.exp(x)+math.exp(-x)) for x in z]

    @staticmethod
    def prime(y):
        if isinstance(y):
            return 1-math.pow(y, 2)
        return [1-math.pow(x, 2) for x in y]
    
    def _mean_square_error_grad(self, y, e):
        loss = np.array(y) - np.array(e)
        prime = self.prime(y)
        return loss * prime
    
    def _mean_absolute_error_grad(self, y, e):
        loss = abs(np.array(y) - np.array(e))
        prime = self.prime(y)
        return loss * prime 
    


class relu(__Activation):
    def __init__(self, loss_function_name:str):
        super().__init__(self.__class__.__qualname__, loss_function_name)

    @staticmethod
    def activation(z):
        if isinstance(z):
            return z if z > 0. else 0.
        return [x if x > 0. else 0. for x in z]

    @staticmethod
    def prime(y):
        if isinstance(y):
            return 1. if y > 0. else 0.
        return [1. if x > 0. else 0. for x in y]
    
    def _mean_square_error_grad(self, y, e):
        loss = np.array(y) - np.array(e)
        prime = self.prime(y)
        return loss * prime
    
    def _mean_absolute_error_grad(self, y, e):
        loss = abs(np.array(y) - np.array(e))
        prime = self.prime(y)
        return loss * prime



class leaky_relu(__Activation):
    def __init__(self, loss_function_name:str):
        super().__init__(self.__class__.__qualname__, loss_function_name)

    @staticmethod
    def activation(z):
        if isinstance(z, float):
            return z if z > 0. else 0.01*z
        return [x if x > 0. else 0.01*x for x in z]

    @staticmethod
    def prime(y):
        if isinstance(y, float):
            return 1. if y > 0 else 0.01
        return [1. if x > 0. else 0.01 for x in y]
    
    def _mean_square_error_grad(self, y, e):
        loss = np.array(y) - np.array(e)
        prime = self.prime(y)
        return loss * prime
    
    def _mean_absolute_error_grad(self, y, e):
        loss = abs(np.array(y) - np.array(e))
        prime = self.prime(y)
        return loss * prime
    
    def _spare_cross_entropy_grad(self, y, e):
        loss = np.array(y) - np.array(e)
        return loss


class softmax(__Activation):
    def __init__(self, loss_function_name:str):
        super().__init__(self.__class__.__qualname__, loss_function_name)

    @staticmethod
    def activation(z):
        if isinstance(z, float):
            raise Exception("Error log: Softmax is in/out vector function and doesnot handle scalar")
        sum_exp_z = sum(np.exp(z))
        return [math.exp(x)/sum_exp_z for x in z]

    @staticmethod
    def prime(y):
        if isinstance(y, float):
            return 1. if y > 0 else 0.01
        return [1. if x > 0. else 0.01 for x in y]
    
    def _categorical_cross_entropy_grad(self, y, e):
        loss = np.array(y) - np.array(e)
        return loss
    
    def _kullback_leibler_divergence_grad(self, y, e):
        loss = abs(np.array(y) - np.array(e))
        return loss
