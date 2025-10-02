
import yaml
import numpy as np
import pandas as pd
import matplotlib as plt
import ml_tools.utils as Utils

from typing import List, Dict
from core.layer import Layer
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

    __layers:List[Layer] = []
    __shape:List = []
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

    def __init_batch_size(self, config: dict):
        try:
            self.__batch_size = int(config["batch size"])
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
            try:
                hidden_act = '_'.join(str.lower(archi["hidden"]["activation"]).split())
            except KeyError:
                hidden_act = "sigmoid"
            try:
                hidden_init = '_'.join(str.lower(archi["hidden"]["initializer"]).split())
            except KeyError:
                hidden_init = "he_normal"

            for x in range(hidden_count):
                self.__layers.append(Layer(hidden_size[x], previous_size, "hidden", hidden_act, hidden_init))
                previous_size = hidden_size[x]
                self.__shape.append(previous_size)
        try:
            output_act = '_'.join(str.lower(archi["output"]["activation"]).split())
        except KeyError:
            output_act = "sigmoid"
        try:
            output_init = '_'.join(str.lower(archi["output"]["initializer"]).split())
        except KeyError:
            output_init = "he_normal"
        self.__layers.append(Layer(output_size, previous_size, "output", output_act, output_init))
        self.__shape.append(output_size)

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
        self.__init_loss(self._config_general)
        self.__init_evaluation(self._config_general)
        self.__init_layers(self._config_archi)
        logger.info("Network's initialisation done !!")

        return
    
    def learn(self, ds_train:List, ds_test:List):
        accuracies:List = []
        errors:List = []

        if len(ds_train[0]["data"]) != self.__shape[0]:
            logger.error(f"Missmatch between the number of input ({len(ds_train[0]['data'])}) and the number of expected input ({self.__shape[0]})")
            raise Exception()
        if len(ds_train[0]["label"]) != self.__shape[-1]:
            logger.error(f"Missmatch between the number of output ({len(ds_train[0]['label'])}) and the number of expected output ({self.__shape[-1]})")
            raise Exception()

        for e in range(self.epoch):
            self.__optimisation_fnc(self, ds_train)
            evaluation:Dict = self.__evaluation_fnc(self, ds_test)
            if self.option_visu_accuracy:
                accuracies.append(evaluation["accuracy"])
            if self.option_visu_loss:
                errors.append(evaluation["error_mean"])
            if self.option_visu_training:
                logger.info(f"Info log: epoch {e}/{self.epoch}: {evaluation}")
            if self.error_threshold > 0 and evaluation["error_mean"] < self.error_threshold:
                break
        logger.info("The training is completed")
    
    def backpropagation(self, input: np.array, label: np.array):
        if len(label) != self.__layers[-1].shape:
            logger.info("The label need to have the same size than output layer")
            raise Exception()
        out = []
        net = self.__layers[0].fire(input)
        out.append(self.__layers[0].activation_fnc(net))
        for i in range(1, len(self.__layers)):
            net = self.__layers[i].fire(out[-1])
            out.append(np.array(self.__layers[0].activation_fnc(net[-1])))
        loss = out[-1] - label
        prime = self.__layers[-1].prime_fnc(out[-1])
        delta = np.dot(loss, prime)
        self.__layers[-1].nabla_b += delta
        self.__layers[-1].nabla_w += (delta * out[-2])
        idx = len(self.__layers)-2
        while idx >= 0:
            prime = self.__layers[idx].prime_fnc(out[idx])
            delta = np.dot(np.transpose(self.__layers[idx+1].weights), delta) * prime
            self.__layers[idx].nabla_b += delta
            self.__layers[idx].nabla_w += np.dot(delta, prime)
            idx-=1
        # dans l idee il faudrait calculer l erreur avec la loss function et la save pour la plot plus tard
        return
        
    def fire(self, input:np.array) -> np.array:
        act_input = input
        for l in self.__layers:
            act_input = l.fire(act_input)
        return act_input

    def update_weights(self, batch_size:int, eta:float):
        for l in self.__layers:
            l.weights -= ((l.nabla_w / batch_size) * eta)
            l.nabla_w = np.zeros(np.shape(l.nabla_w))

    def update_biaises(self, batch_size:int, eta:float):
        for l in self.__layers:
            l.biaises -= ((l.nabla_b / batch_size) * eta)
            l.nabla_b = np.zeros(np.shape(l.nabla_b))

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
    


    



        
    