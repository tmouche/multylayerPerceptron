
import numpy as np
import mathplotlib as plt

from myMath import myMath

class Layer:

    activation_fnc = None
    prime_fnc = None
    weight_init_fnc = None

    __weight: np.array = None
    __biaises: np.array = None

    def __init__(self, size : int, previous_size : int, units : str, activation_function='sigmoid', weight_initializer='default'):
        try:
            self.activation_fnc = getattr(myMath, activation_function)
            self.prime_fnc = getattr(myMath, activation_function + "Prime")
            if activation_function == 'softmax' and units != 'output':
                raise Exception("Error log: Softmax can not be the activation function for other than output")
        except:
            raise Exception("Error log: Unrecognized activation function")
        if weight_initializer != 'default':
            try:
                self.__weight = myMath.zeros(shape=(size, previous_size))
                self.__biaises = myMath.zeros(shape=(size, 1))
                self.weight_init_fnc = getattr(myMath, weight_initializer)
            except:
                raise Exception("Error log: Weight initializer unrecognized")
        else:
            self.__weight = myMath.randomNormal(shape=(size, previous_size))
            self.__biaises = myMath.randomNormal(shape=(size, 1))
            print("Console: layer well initialized by default")

    

    