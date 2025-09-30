
import yaml
import numpy as np
import pandas as pd
import matplotlib as plt
import ml_tools.losses as Losses
import ml_tools.optimizers as Optimizers
import ml_tools.utils as Utils
import ml_tools.evaluators as Evaluators

from typing import List, Dict
from core.layer import Layer

class Network:
    
    option_visu_training: bool = False
    option_visu_loss: bool = False
    option_visu_accuracy: bool = False

    _config = None
    _config_general = None
    _config_archi = None

    __learning_rate: float = None
    __epoch: int = None
    __batch_size: int = None

    __error_threshold:float = 0. #A IMPLEMENTER OPTIONNAL

    __optimisation_name:str = None
    __optimisation_fnc = None

    __evaluator_name:str = None
    __evaluator_fnc = getattr(Evaluators, "classification") #A IMPLEMENTER

    __loss_name:str = None
    __loss_fnc = None

    __layers:List[Layer] = []

    def __init_mandatories(self, config: dict):
        try:
            self.__learning_rate = float(config["learning rate"])
            self.__epoch = int(config["epochs"])
        except KeyError:
            raise Exception("Error log: Missing key in the config file")

    def __init_optimisation_name(self, config: dict):
        try:
            self.__optimisation_name = '_'.join(str.lower(config["optimisation"]).split())
        except KeyError:
            self.__optimisation_name = "gradient_descent"
        try:
            self.__optimisation_fnc = getattr(Optimizers, self.__optimisation_name)
        except KeyError:
            raise Exception(f"Error log: Optimisation function {self.__optimisation_name} is unknow")

    def __init_batch_size(self, config: dict):
        try:
            self.__batch_size = int(config["batch size"])
        except KeyError:
            if (self.__optimisation_name == "stochastic_gradient_descent"):
                raise Exception(f"Error log: Missing batch size for the SGD")

    def __init_loss_name(self, config: dict):
        try:
            self.__loss_name = '_'.join(str.lower(config["loss"]).split())
        except KeyError:
            raise Exception(f"Error log: The key loss is missing")
        try:
            self.__loss_fnc = getattr(Losses, self.__loss_name)
        except KeyError: 
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
        try:
            self._config = yaml.safe_load(dataStr)
            self._config_general = self._config["general"]
            self._config_archi = self._config["archi"]
        except:
            raise Exception(f"Error log: The config file need to be a .yaml with atleast general and archi keys")

        self.__init_mandatories(self._config_general)
        self.__init_optimisation_name(self._config_general)
        self.__init_loss_name(self._config_general)
        self.__init_layers(self._config_archi)

        return
    
    def learn(self, ds_train:np.array, ds_test:np.array):
        accuracies:List = []
        errors:List = []

        for e in range(self.__epoch):
            updater:Dict = self.__optimisation_fnc(ds_train, self._config_general)
            nabla_b = updater["nabla_b"]
            nabla_w = updater["nabla_w"]
            for l in range(self.__layers):
                self.__layers[l].update_biaises(nabla_b[l])
                self.__layers[l].update_weights(nabla_w[l])
            self.__learning_rate = updater.get("learning_rate" , self.__learning_rate)
            evaluation:Dict = self.__evaluator_fnc(self, ds_test)
            if self.option_visu_accuracy:
                accuracies.append(evaluation["accuracy"])
            if self.option_visu_loss:
                errors.append(evaluation["error_mean"])
            if self.option_visu_training:
                print(f"Info log: epoch {e}/{self.__epoch}: {evaluation}")
            if self.__error_threshold > 0 and evaluation["error_mean"] < self.__error_threshold:
                break
        print("Info log: The training is completed")

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
        loss = self.__loss_fnc(act[-1], label)
        prime = self.__layers[-1].prime_fnc(zs[-1]) # ya une opti a faire en envoyant directement l act et pas le zs vu que la derivative se calcule sur le l act et pas sur le z
        delta = np.dot(loss, prime)
        # nabla_b = delta
        # nabla_w = delta * act[-2]
        
        print("res:", act[-1])
        
    def fire(self, input:np.array) -> np.array:
        act_input = input
        for l in self.__layers:
            act_input = l.fire(act_input)
        return act_input


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
    


    



        
    