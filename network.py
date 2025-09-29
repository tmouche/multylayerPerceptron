
import yaml
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

    __optimisation_name: str = None
    __optimisation_fnc:str = None

    __loss_name: str = None
    __loss_fnc = None

    __layers = []

    def __init_mandatories(self, config: dict):
        try:
            self.__learning_rate = float(config["general"]["learning rate"])
            self.__epoch = int(config["general"]["epochs"])
        except KeyError:
            raise Exception("Error log: Missing key in the config file")

    def __init_optimisation_name(self, config: dict):
        try:
            self.__optimisation_name = '_'.join(str.lower(config["general"]["optimisation"]).split())
        except KeyError:
            self.__optimisation_name = "gradient_descent"
        try:
            self.__optimisation_fnc = getattr(Network, self.__optimisation_name)
        except AttributeError:
            raise Exception(f"Error log: Optimisation function {self.__optimisation_name} is unknow")

    def __init_batch_size(self, config: dict):
        try:
            self.__batch_size = int(config["general"]["batch size"])
        except KeyError:
            if (self.__optimisation_name == "stochastic_gradient_descent"):
                raise Exception(f"Error log: Missing batch size for the SGD")

    def __init_loss_name(self, config: dict):
        try:
            self.__loss_name = '_'.join(str.lower(config["general"]["loss"]).split())
        except KeyError:
            raise Exception(f"Error log: The key loss is missing")
        try:
            self.__loss_fnc = getattr(myMath, self.__loss_name)
        except AttributeError: 
            raise Exception(f"Error log: Loss function {self.__loss_fnc} is unknow")
        
    def __init_layers(self, archi: dict):
        try:
            previous_size = int(archi["input"]["size"])
            output_size = int(archi["output"]["size"])
        except KeyError:
            raise Exception(f"Error log: Missing mandatories keys (architecture.input.size, architecture.output.size)")
        if previous_size < 1 or output_size < 0:
            raise Exception(f"Error log: The input size and the output size cannot be lower than 1")
        if "hidden" in archi:
            try:
                hidden_size = int(archi["hidden"]["size"])
                hidden_count = int(archi["hidden"]["count"])
            except KeyError:
                raise Exception(f"Error log: Missing optinnals keys (architecture.hidden.size, architecture.hidden.count)")
            if hidden_count < 0:
                raise Exception(f"Error log: The Network can not have a negative count of layer")
            if hidden_count != len(hidden_size):
                raise Exception(f"Error log: Miss match between the count and the sizes")
            try:
                hidden_act = '_'.join(archi["hidden"]["activation"].split())
            except KeyError:
                hidden_act = "sigmoid"
            try:
                hidden_init = '_'.join(archi["hidden"]["initializer"].split())
            except KeyError:
                hidden_init = "he_normal"

            for x in range(hidden_count):
                self.__layers.append(Layer(hidden_size[x], previous_size, "hidden", hidden_act, hidden_init))
                previous_size = hidden_size[x]
        try:
            output_act = '_'.join(archi["output"]["activation"].split())
        except KeyError:
            output_act = "sigmoid"
        try:
            output_init = '_'.join(archi["output"]["initializer"].split())
        except KeyError:
            output_init = "he_normal"
        self.__layers.append(Layer(output_size, previous_size, "output", output_act, output_init))

    def __init__(self, init_file_path: str):
        try:
            f = open(init_file_path, 'r')
        except:
            raise Exception(f"Error log: Can not open the file {init_file_path}")
        dataStr = f.read()
        config = yaml.safe_load(dataStr)

        self.__init_mandatories(config)
        self.__init_optimisation_name(config)
        self.__init_batch_size(config)
        self.__init_loss_name(config)

        try:
            archi = config["architecture"]
        except KeyError:
            raise Exception(f"Error log: Architecture needed to initialise the network")
        
        self.__init_layers(archi)
        return
    

# all_l = config.get("layer", [])
# if len(all_l) < 3:
#     raise Exception("Error log: The network needs at least 3 layers")
# if all_l["0"].get("unit", "") != "input":
#     raise Exception(f"Error log: layer {0}: unknow name {all_l[0].get('unit')}")
# x = 1
# while x < len(all_l):
#     u = all_l[str(x)]
#     unit = u.get("unit")
#     if x < len(all_l)-1 and unit != 'hidden' or x == len(all_l)-1 and unit != 'output':
#         raise Exception(f"Error log: layer {x}: unknow name {unit}")
#     s = u.get("size",0)
#     p_s = all_l[str(x-1)].get("size",0)
#     act = u.get("activation", "sigmoid")
#     init = u.get("initializer", "default")
#     if s < 1 or p_s < 1:
#         raise Exception(f"Error log: layer {x}: too small")
#     if x == len(all_l)-1:
#         if act == "softmax" and s == 1:
#             raise Exception("Error log: softmax can not be used with less than 2 outputs neurons")
#         if self.__loss_name == "binaryCrossEntropy" and s != 1:
#             raise Exception("Error log: The binary cross entropy can not be used with more than 1 output neurons")
#     self.__layers.append(Layer(s, p_s, unit, act, init))
#     x += 1
    
    def train(self):
        pass

    def gradient_descent(self):
        pass

    def stochastic_gradient_descent(self):
        pass

    def nesterov_momentum(self):
        pass

    def rms_prop(self):
        pass

    def adam(self):
        pass

    def train(self, ds_train: np.array, ds_test: np.array):

        mean_loss_epochs = []
        accuracy_epochs = []
        if len(ds_train) < self.__batch_size * 2:
            raise Exception(f"Error log: data set to small to be used with this batch size")
        ds_len = int(len(ds_train) / self.__batch_size) * self.__batch_size
        # il faut check la batchsize en amont
        for e in range(self.__epoch):
            np.random.shuffle(ds_train)
            batch = [[ds_train[i] for i in range(j, j+self.__batch_size)] for j in range(0, ds_len, self.__batch_size)]
            for b in range(len(batch)):
                for l in range(self.__batch_size):
                    self.backpropagation(batch[b][l][1], batch[b][l][0])
            
    def backpropagation(self, input: np.array, label: np.array):
        if len(label) != self.__layers[-1].shape:
            raise Exception("Error log: The label need to have the same size than output layer")
        act = []
        zs = []
        zs.append(self.__layers[0].fire(input))
        act.append(self.__layers[0].activation_fnc(zs[-1]))
        for i in range(1, len(self.__layers)):
            zs.append(self.__layers[i].fire(act[-1]))
            act.append(self.__layers[0].activation_fnc(zs[-1]))
        
        print("res:", act[-1])
        



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
    


    



        
    