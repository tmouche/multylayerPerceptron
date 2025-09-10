
import numpy as np
import pandas as pd
import mathplotlib as plt

from myMath import myMath
from layer import Layer

class Network:
    
    __learning_rate: float = None
    __epoch: int = None
    __optimisation: str = None

    def __init__(self, init_file_path : str):
        try:
            f = open(init_file_path, 'w')
        except:
            raise Exception(f"Error log: Can not open the file {init_file_path}")
        


    def add_layer(self, activation, size, weight_initialisation):



        
    