
import numpy as np

import ml_tools.activations as Activations
import ml_tools.initializers as Initializers
import ml_tools.primes as Prime

from typing import List

class Layer:

    unit: str = None
    shape: int = None

    activation_name: str = None
    activation_fnc = None
    prime_name: str = None
    prime_fnc = None
    weight_init_name: str = None

    weight: np.array = None
    biai: np.array = None

    def __init__(self, size : int, prev_size : int, unit : str, act_fct : str, w_init : str):
        a_unit = ["input", "hidden", "output"]
        a_act = ["sigmoid", "relu", "leaky_relu", "tanh", "step", "softmax"]
        a_init = ["random_normal", "random_uniform", "zeros", "ones", "xavier_normal", "xavier_uniform", "he_normal", "he_uniform"]
        
        self.shape = size

        if unit not in a_unit:
            raise Exception(f"Error log: {unit} is not know as unit")
        if act_fct not in a_act:
            raise Exception(f"Error log: {act_fct} is not know as activation function")
        if w_init not in a_init:
            raise Exception(f"Error log: {w_init} is not know as weights initializer")
        self.unit = unit
        try:
            self.activation_name = act_fct
            self.activation_fnc = getattr(Activations, self.activation_name)
            self.prime_name = act_fct + "_prime"
            self.prime_fnc = getattr(Prime, self.prime_name)
            if act_fct == 'softmax' and unit != 'output':
                raise Exception("Error log: Softmax can not be the activation function for other than output")
        except:
            raise Exception("Error log: Unrecognized activation function")
        try:
            self.weight_init_name = w_init
            w_init_fnc = getattr(Initializers, self.weight_init_name)
            self.weight = w_init_fnc(shape=(size, prev_size))
            self.biai = w_init_fnc(shape=(size))
        except:
            raise Exception("Error log: Weight initializer unrecognized")

    def fire(self, input:List):
        res = [sum([self.weight[i][j]*input[j] for j in range(len(input))]) + self.biai[i] for i in range(len(self.weight))]
        return res

    def update_biaises(self, nabla:List):
        pass

    def update_weights(self, nabla:List):
        pass