
import numpy as np
import pandas as pd
import mathplotlib as plt

class Network:
    
    __learning_rate: float = None
    __max_epoch: int = None
    __activation: str = None
    __optimisation: str = None
    __weights: np.array = None
    __biaises: np.array = None

    def __init__(self, activation="sigmoid", max_epoch=10, learning_rate=0.005, opti="adam"):
        self.__activation = activation
        self.__max_epoch = max_epoch
        self.__learning_rate = learning_rate
        print(f"log: the network has been initialized with the following options:\
                - activation: {self.__activation}\
                - max epoch: {self.__max_epoch}\
                - learning rate: {self.__learning_rate}\
                - optimisation: {self.__optimisation}")