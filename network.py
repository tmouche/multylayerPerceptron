
import json
import numpy as np
import pandas as pd
import mathplotlib as plt

from myMath import myMath
from layer import Layer

class Network:
    
    __learning_rate: float = None
    __epoch: int = None
    __batch_size: int = None
    __optimisation: str = None

    __layers = []

    def __init__(self, init_file_path : str):
        try:
            f = open(init_file_path, 'r')
        except:
            raise Exception(f"Error log: Can not open the file {init_file_path}")
        dataStr = f.read()
        f.close()
        dataJson = json.loads(dataStr)
        try:
            self.__learning_rate = float(dataJson["general"]["learningRate"])
            self.__batch_size = float(dataJson["general"]["batchSize"])
            self.__epoch = float(dataJson["general"]["epochs"])
        except ValueError:
            raise Exception("Error log: Unknow or missing token in the initialisation file")
        try:
            all_layers = dataJson["layer"]
            if len(all_layers) < 3:
                raise Exception("Error log: The network needs at least 3 layers")
            x = 1
            while x < len(all_layers):
                self.__layers.append(Layer(all_layers[1], 0, ))
                x += 1
            




    def add_layer(self, activation, size, weight_initialisation):



        
    