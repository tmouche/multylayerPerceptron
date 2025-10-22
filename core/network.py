
import yaml
import time
import numpy as np
import pandas as pd
import matplotlib as plt
import ml_tools.utils as Utils
import ml_tools.activations as Activations


from typing import List, Dict
from utils.logger import Logger
logger = Logger()

class Network:
    
    option_visu_training: bool = False
    option_visu_loss: bool = False
    option_visu_accuracy: bool = False

    learning_rate: float = None
    epoch: int = None
    batch_size: int = None

    error_threshold:float = 0. #A IMPLEMENTER OPTIONNAL
    
    _config = None
    _config_general = None
    _config_archi = None

    optimisation_name:str = None
    __optimisation_fnc = None

    evaluation_name:str = None
    __evaluation_fnc = None

    loss_name:str = None
    __loss_fnc = None

    activation_name:str = None
    __activation = None

    output_activation_name:str = None
    __output_activation = None

    initialisation_name:str = None
    __initialisation_fnc = None

    __weights:np.array = None
    __nabla_w:np.array = None
    __biaises:np.array = None
    __nabla_b:np.array = None

    __shape:List = []

    def __init__(self, init_file_path: str):
        try:
            f = open(init_file_path, 'r')
        except:
            logger.error(f"Can not open the file {init_file_path}")
            raise Exception()
        dataStr = f.read()
        try:
            self._config = yaml.safe_load(dataStr)
            self._config_general = self._config["general"]
            self._config_archi = self._config["architecture"]
        except:
            logger.error(f"The config file need to be a .yaml with atleast general and architecture keys")
            raise Exception()

        logger.info("Network's initialisation starting...")
        self.__init_mandatories(self._config_general)
        self.__init_optimisation(self._config_general)
        self.__init_batch_size(self._config_general)
        self.__init_evaluation(self._config_general)
        self.__init_loss(self._config_general)
        self.__init_activation(self._config_general)
        self.__init_initialisation(self._config_general)
        self.__init_layers(self._config_archi)
        logger.info("Network's initialisation done !!")
        return

    def __init_mandatories(self, config: dict):
        try:
            self.learning_rate = float(config["learning rate"])
            self.epoch = int(config["epochs"])
        except KeyError:
            logger.error("Missing key in the config file")
            raise Exception()
        
    def __init_optimisation(self, config: dict):
        try:
            self.optimisation_name = '_'.join(str.lower(config["optimisation"]).split())
        except KeyError:
            self.optimisation_name = "gradient_descent"
        try:
            import ml_tools.optimizers as Optimizers
            self.__optimisation_fnc = getattr(Optimizers, self.optimisation_name)
        except KeyError:
            logger.error(f"Optimisation function {self.optimisation_name} is unknown")
            raise Exception()
        
    def __init_evaluation(self, config:dict):
        try:
            self.evaluation_name = '_'.join(str.lower(config["evaluation"]).split())
        except KeyError:
            logger.error("Missing the evaluation method")
        try:
            import ml_tools.evaluations as Evaluators
            self.__evaluation_fnc = getattr(Evaluators, self.evaluation_name)
        except:
            logger.error(f"Evaluation function {self.evaluation_name} is unknown")
            raise Exception()

    def __init_activation(self, config:dict):
        try:
            self.activation_name = '_'.join(str.lower(config["activation"]).split())
        except KeyError:
            self.activation_name = "sigmoid"
        if self.activation_name == "softmax":
            logger.error("Softmax can not be used as activation for hidden layers")
            raise Exception()
        try:
            self.__activation = getattr(Activations, self.activation_name)(self.loss_name)
        except AttributeError:
            logger.error(f"Activation function {self.activation_name} unknown")
            raise Exception()
        
    def __init_initialisation(self, config:Dict):
        a_init = ["random_normal", "random_uniform", "zeros", "ones", "xavier_normal", "xavier_uniform", "he_normal", "he_uniform"]
        try:
            self.initialisation_name = '_'.join(str.lower(config["initialisation"]).split())
        except KeyError:
            self.initialisation_name = "he_normal"
        try:
            import ml_tools.initialisations as Initialisations
            self.__initialisation_fnc = getattr(Initialisations, self.initialisation_name)
        except AttributeError:
            logger.error(f"Initialisation function {self.initialisation_name} is unknown")

    def __init_batch_size(self, config: dict):
        try:
            self.batch_size = int(config["batch size"])
        except KeyError:
            if (self.optimisation_name == "stochastic_gradient_descent"):
                logger.error("Missing batch size for the SGD")
                raise Exception()

    def __init_loss(self, config: dict):
        try:
            self.loss_name = '_'.join(str.lower(config["loss"]).split())
        except KeyError:
            logger.error("Missing loss function")
            raise Exception()
        try:
            import ml_tools.losses as Losses
            self.__loss_fnc = getattr(Losses, self.loss_name)
        except KeyError:
            logger.error(f"Loss function {self.__loss_fnc} is unknow")
            raise Exception()

    def __init_layers(self, archi: dict):
        self.__weights = []
        self.__biaises = []
        self.__nabla_w = []
        self.__nabla_b = []
        try:
            previous_size = int(archi["input"]["size"])
            output_size = int(archi["output"]["size"])
        except KeyError:
            logger.error(f"Missing mandatories keys (architecture.input.size, architecture.output.size)")
            raise Exception()
        if previous_size < 1 or output_size < 0:
            logger.error(f"The input size and the output size cannot be lower than 1")
            raise Exception()
        self.__shape.append(previous_size)
        if "hidden" in archi:
            try:
                hidden_size = list[int](archi["hidden"]["size"])
                hidden_count = int(archi["hidden"]["count"])
            except KeyError:
                logger.error(f"Missing optinnals keys (architecture.hidden.size, architecture.hidden.count)")
                raise Exception()
            if hidden_count < 0:
                logger.error("The Network can not have a negative count of layer")
                raise Exception()
            if hidden_count != len(hidden_size):
                logger.error("Missmatch between the count and the sizes")
                raise Exception()
            for x in range(hidden_count):
                self.__create_layers(hidden_size[x], previous_size)
                previous_size = hidden_size[x]
                self.__shape.append(previous_size)
        self.__create_layers(output_size, previous_size)
        self.__shape.append(output_size)
        try:
            self.output_activation_name = '_'.join(str.lower(archi["output"]["activation"]).split())
        except KeyError:
            self.output_activation_name = self.activation_name
        if self.output_activation_name == "softmax" and output_size < 2:
            logger.error("Softmax needs atleast two output neurons")
            raise Exception()
        try:
            self.__output_activation = getattr(Activations, self.output_activation_name)(self.loss_name)
        except AttributeError:
            logger.error(f"Activation function {self.output_activation_name} unknown")
            raise Exception()
    
    def __create_layers(self, size:int, prev_size:int):
        self.__weights.append(self.__initialisation_fnc(shape=(size,prev_size)))
        self.__biaises.append(self.__initialisation_fnc(shape=(size)))
        self.__nabla_w.append(np.full((size,prev_size), 0.))
        self.__nabla_b.append(np.full((size), 0.))
    
    def learn(self, ds_train:List, ds_test:List):
        accuracies:List = []
        errors:List = []

        if len(ds_train[0]["data"]) != self.__shape[0]:
            logger.error(f"Missmatch between the number of input ({len(ds_train[0]['data'])}) and the number of expected input ({self.__shape[0]})")
            raise Exception()
        if len(ds_train[0]["label"]) != self.__shape[-1]:
            logger.error(f"Missmatch between the number of output ({len(ds_train[0]['label'])}) and the number of expected output ({self.__shape[-1]})")
            raise Exception()

        logger.info("Starting training...")
        start_time = time.perf_counter()
        for e in range(self.epoch):
            self.__optimisation_fnc(self, ds_train)
            evaluation:Dict = self.__evaluation_fnc(self.__loss_fnc, self, ds_test)
            accuracies.append(evaluation["accuracy"])
            if self.option_visu_accuracy:
                pass
            if self.option_visu_loss:
                errors.append(evaluation["error_mean"])
            if self.option_visu_training and not e % 100 or self.epoch < 100:
                logger.info(f"epoch {e}/{self.epoch}: {evaluation}")
            if self.error_threshold > 0 and abs(evaluation["error_mean"]) < self.error_threshold:
                break
        end_time = time.perf_counter()
        logger.info(f"epoch {e}/{self.epoch}: {evaluation}")
        logger.info(f"The training is completed in {end_time - start_time}sec")
        return accuracies, errors
    
    def backpropagation(self, input: np.array, label: np.array):
        if len(label) != self.__shape[-1]:
            logger.info("The label need to have the same size than output layer")
            raise Exception()
        out = []
        out.append(input)
        net = input
        size = len(self.__shape)
        for i in range(size - 2):
            net = self.fire_layer(self.__weights[i], self.__biaises[i], out[-1])
            out.append(self.__activation.activation(net))
        net = self.fire_layer(self.__weights[-1], self.__biaises[-1], out[-1])
        out.append(self.__output_activation.activation(net))
        delta = self.__activation.delta(out[-1], label)
        self.__nabla_w[-1] = np.array(self.__nabla_w[-1]) + np.outer(np.array(delta), np.array(out[-2]))
        self.__nabla_b[-1] = np.array(self.__nabla_b[-1]) + delta
        idx = len(self.__shape)-3
        while idx >= 0:
            prime = self.__activation.prime(out[idx+1])
            delta = np.dot(np.transpose(self.__weights[idx+1]), delta) * prime
            self.__nabla_w[idx] = np.array(self.__nabla_w[idx]) + np.outer(np.array(delta), np.array(out[idx]))
            self.__nabla_b[idx] = np.array(self.__nabla_b[idx]) + delta
            idx-=1
        return
        
    def fire(self, input:np.array) -> np.array:
        act_input = input
        for l in range(len(self.__shape) - 2):
            act_input = np.array(self.__activation.activation(self.fire_layer(self.__weights[l], self.__biaises[l], act_input)))
        act_input = np.array(self.__output_activation.activation(self.fire_layer(self.__weights[-1], self.__biaises[-1], act_input)))
        return act_input
    
    def fire_layer(self, weight:np.array, biaises:np.array, input:np.array) -> np.array:
        res = [np.dot(w, input) + b for w,b in zip(weight, biaises)]
        return res

    def update_weights(self, batch_size:float, eta:float):
        for i in range(len(self.__shape) - 1):
            self.__weights[i] = np.array(self.__weights[i]) - eta * (np.array(self.__nabla_w[i]) / batch_size)
            self.__nabla_w[i] = np.full(np.shape(self.__nabla_w[i]), 0.0)

    def update_biaises(self, batch_size:float, eta:float):
        for i in range(len(self.__shape) - 1):
            self.__biaises[i] = np.array(self.__biaises[i]) - eta * (np.array(self.__nabla_b[i]) / batch_size) 
            self.__nabla_b[i] = np.full(np.shape(self.__nabla_b[i]), 0.0)


    def checkNetwork(self):
        print("-- NETWORK --")
        print("General options:")
        print(f" -learning rate: {self.learning_rate}")
        print(f" -epochs: {self.epoch}")
        print(f" -batch size: {self.batch_size}")
        print(f" -optimisation: {self.optimisation_name}")
        print(f" -loss: {self.loss_name}")
        print()
        print("--LAYERS--")
        print(" -weights:")
        print(self.__weights)
        print(" -biaises:")
        print(self.__biaises)
        return
    


    



        
    