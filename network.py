
import json
import numpy as np
import pandas as pd
import matplotlib as plt

from myMath import myMath
from layer import Layer

class Network:
    
    option_visu_training: bool = False
    option_visu_loss: bool = False
    option_visu_accuracy: bool = False

    __learning_rate: float = None
    __epoch: int = None
    __batch_size: int = None
    __optimisation: str = None

    __loss_name: str = None
    __loss_fnc = None

    __layers = []

    def __init__(self, init_file_path: str, ds_train: np.array, ds_test: np.array):
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
            self.__loss_name = dataJson["general"]["loss"]
        except KeyError:
            raise Exception("Error log: Unknow or missing token in the initialisation file")
        all_l = dataJson.get("layer", [])
        if len(all_l) < 3:
            raise Exception("Error log: The network needs at least 3 layers")
        if all_l["0"].get("unit", "") != "input":
            raise Exception(f"Error log: layer {0}: unknow name {all_l[0].get('unit')}")
        x = 1
        while x < len(all_l):
            u = all_l[str(x)]
            unit = u.get("unit")
            if x < len(all_l)-1 and unit != 'hidden' or x == len(all_l)-1 and unit != 'output':
                raise Exception(f"Error log: layer {x}: unknow name {unit}")
            s = u.get("size",0)
            p_s = all_l[str(x-1)].get("size",0)
            act = u.get("activation", "sigmoid")
            init = u.get("initializer", "default")
            if s < 1 or p_s < 1:
                raise Exception(f"Error log: layer {x}: too small")
            if x == len(all_l)-1:
                if act == "softmax" and s == 1:
                    raise Exception("Error log: softmax can not be used with less than 2 outputs neurons")
                if self.__loss_name == "binaryCrossEntropy" and s != 1:
                    raise Exception("Error log: The binary cross entropy can not be used with more than 1 output neurons")
            self.__layers.append(Layer(s, p_s, unit, act, init))
            x += 1
        return
    
    


    def checkNetwork(self):
        print("-- NETWORK --")
        print("General options:")
        print(f" -learning rate: {self.__learning_rate}")
        print(f" -epochs: {self.__epoch}")
        print(f" -batch size: {self.__batch_size}")
        print(f" -optimisation: {self.__optimisation}")
        print(f" -loss: {self.__loss_name}")
        print()
        print("--LAYERS--")
        for x in self.__layers:
            print("General options:")
            print(f" -activation name: {x.activation_name}")
            print(f" -prime name: {x.prime_name}")
            print(f" -weight init name: {x.weight_init_name}")
            print("All weights:")
            print(x.weight)
            print("All biai:")
            print(x.biai)
        return
    


    



        
    