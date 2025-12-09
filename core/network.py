import sys
import yaml
import time
import math
import numpy
import matplotlib as plt
import ml_tools.initialisations as Initialisations
import ml_tools.activations as Activations
import ml_tools.losses as Losses

from ml_tools.utils import step

from utils.history import save_to_history
from typing import List, Dict, Sequence, Optional
from utils.logger import Logger
logger = Logger()

EPS = 1e-8

class NetworkConfig:
    
    learning_rate: float = 0.001
    epoch: int = 50
    batch_size: Optional[int] = None

    momentum_rate: float = 0.9
    velocity_rate: float = 0.9

    loss_threshold: Optional[float] = None

    shape:List

    evaluation:callable

    activation_name:str
    loss_name:str
    optimisation_name:str
    output_activation_name:str
    initialisation_name:str


# ==========================================================
# NETWORK CLASS
# ==========================================================
class Network:
    
    # ======================================================
    # --- 1. CLASS ATTRIBUTES / CONFIG DEFAULTS ---
    # =====================================================
    option_visu_training: bool = False

    config:NetworkConfig 

    # Components and functions
    __opti_fnc:callable = None
    __eval_fnc:callable = None
    __loss_fnc:callable = None
    __act_fnc:callable = None
    __output_act_fnc:callable = None
    __init_fnc:callable = None

    # Network structure
    __weights:List = None
    __biaises:List = None

    __nabla_w:Sequence = None
    __nabla_b:Sequence = None

    __momentum_w:Sequence = None
    __momentum_b:Sequence = None

    __ahead_w:Sequence = None
    __ahead_b:Sequence = None

    __velocity_w:Sequence = None
    __velocity_b:Sequence = None

    # ======================================================
    # --- 2. INITIALIZATION ---
    # ======================================================

    def __init__(self, config:NetworkConfig):
        self.config = config
        self.__check_config()

    def check_config(self, init_file_path: str):
        """
        Load YAML config and initialize network structure and parameters.
        """

        logger.info("Network initialization starting...")
        logger.info("Configuration starting...")
        self.__check_mandatories()
        self.__check_optimisation()
        self.__check_activation()
        self.__init_initialisation()
        logger.info("Configuration complete...")
        logger.info("Layers initialization starting...")
        self.__init_layers(config_archi)
        logger.info("Layers initialization complete...")
        logger.info("Network initialization complete...")

    # ------------------------------------------------------
    # --- 2.1 CONFIGURATION INITIALISATION ---
    # ------------------------------------------------------

    def __check_mandatories(self):
        if self.config.learning_rate is None or self.config.learning_rate <= 0:
            logger.error("Learning rate cannot be negative or egal to 0")
            raise Exception()
        if self.config.epoch is None or self.config.epoch <= 0:
            logger.error("The number of epoch cannot be negative or egal to 0")
            raise Exception()
        logger.info("Mandatories OK..")
        
    def __check_optimisation(self):
        opti_name = '__' + '_'.join(str.lower(self.config.optimisation_name).split())
        try:
            self.__opti_fnc = getattr(self, opti_name)
        except KeyError:
            logger.error(f"Optimisation function {self.config.optimisation_name} is unknown")
            raise Exception()
        
        if self.config.batch_size is None:
            if "mini" in self.__opti_name:
                logger.error(f"Missing batch size for {self.config.optimisation_name}")
                raise Exception()
        else:
            if self.config.batch_size <= 0:
                logger.error(f"{self.config.batch_size} not a valid value for batch size")
                raise Exception()
        
        if self.config.momentum_rate is None or self.config.momentum_rate <= 0.:
            if "adam" or "nag" in self.config.optimisation_name:
                logger.error(f"{self.config.momentum_rate} is not a valid value for momentum rate")
                raise Exception()
        if self.config.velocity_rate is None or self.config.velocity_rate <= 0.:
            if "adam" or "rms" in self.config.optimisation_name:
                logger.error(f"{self.config.velocity_rate} is not a valid value for velocity rate")
                raise Exception()
        
    def __check_activation(self):
        loss_name = '_'.join(str.lower(self.config.loss_name).split())
        act_name = '_'.join(str.lower(self.config.activation_name).split())
        if act_name == "softmax":
            logger.error("Softmax can not be used as activation for hidden layers")
            raise Exception()
        try:
            self.__loss_fnc = getattr(Losses, loss_name)
        except AttributeError:
            logger.error(f"Activation function {act_name} unknown")
            raise Exception()
        try:
            self.__act_fnc = getattr(Activations, act_name)(loss_name)
        except AttributeError:
            logger.error(f"Activation function {act_name} unknown")
            raise Exception()
        
    def __init_initialisation(self):
        init_name = '_'.join(str.lower(self.config.initialisation_name).split())
        try:
            self.__init_fnc = getattr(Initialisations, init_name)
        except AttributeError:
            logger.error(f"Initialisation function {init_name} is unknown")

    # ------------------------------------------------------
    # --- 2.2 LAYER INITIALIZATION ---
    # ------------------------------------------------------
    def __init_layers(self, archi: dict):
        self.__weights = []
        self.__biaises = []
        
        if len(self.config.shape) < 3:
            logger.error("The Network has to have at least 3 layers")
            raise Exception()

        for i in range(1, len(self.config.shape)):
            if self.config.shape[i] <= 0 or self.config.shape[i-1] <= 0:
                logger.error("A layer size can not be negativ or egal to 0")
                raise Exception()
            self.__create_layers(self.config.shape[i], self.config.shape[i-1])
                
        try:
            previous_size = self.config.shape[0]
            output_size = self.config.shape[-1]
        except KeyError:
            logger.error(f"Missing mandatories keys (architecture.input.size, architecture.output.size)")
            raise Exception()
        if previous_size < 1 or output_size < 0:
            logger.error(f"The input size and the output size cannot be lower than 1")
            raise Exception()
        for i in range(1, len(self.config.shape)):
            # try:
            #     hidden_size = list[int](archi["hidden"]["size"])
            #     hidden_count = int(archi["hidden"]["count"])
            # except KeyError:
            #     logger.error(f"Missing optinnals keys (architecture.hidden.size, architecture.hidden.count)")
            #     raise Exception()
            # if hidden_count < 0:
            #     logger.error("The Network can not have a negative count of layer")
            #     raise Exception()
            # if hidden_count != len(hidden_size):
            #     logger.error("Missmatch between the count and the sizes")
            #     raise Exception()
            for x in range(hidden_count):
                self.__create_layers(hidden_size[x], previous_size)
                previous_size = hidden_size[x]
                self.shape.append(previous_size)
        self.__create_layers(output_size, previous_size)
        self.shape.append(output_size)
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

    # ======================================================
    # --- 3. TRAINING METHODS ---
    # ======================================================
    def train(self, ds_train:List, ds_test:List) -> tuple[Sequence, Sequence]:
        accuracies:Dict = {'testing': [], 'training': []}
        losses:Dict = {'testing': [], 'training': []}

        min_training_loss: float = sys.float_info.max
        min_testing_loss: float = sys.float_info.max

        if len(ds_train[0]["data"]) != self.shape[0]:
            logger.error(f"Missmatch between the number of input ({len(ds_train[0]['data'])}) and the number of expected input ({self.shape[0]})")
            raise Exception()
        if len(ds_train[0]["label"]) != self.shape[-1]:
            logger.error(f"Missmatch between the number of output ({len(ds_train[0]['label'])}) and the number of expected output ({self.shape[-1]})")
            raise Exception()

        logger.info("Starting training...")
        start_time = time.perf_counter()
        for e in range(self.config.epoch):
            training:Dict = self.__opti_fnc(ds_train)
            testing:Dict = self.__eval_fnc(self, self.__loss_fnc, ds_test)
            
            accuracies['testing'].append(testing.get("accuracy"))
            accuracies['testing'].append(testing.get("loss"))
            losses['training'].append(training.get("accuracy"))
            losses['training'].append(training.get("loss"))
            
            if self.option_visu_training:
                self.__print_epoch_state(
                    epoch=e,
                    training_accuracy=training.get("accuracy"),
                    training_loss=training.get("loss"),
                    testing_accuracy= testing.get("accuracy"),
                    testing_loss= testing.get("loss")
                )
            if self.error_threshold and abs(testing["loss"]) < self.error_threshold:
                break
        time_stamp = time.perf_counter() - start_time    
        self.__save_training(
            accuracy=testing["accuracy"],
            precision=testing["precision"],
            recall=testing["recall"], 
            f1=testing["f1"], 
            time=time_stamp, 
            min_training_loss=min_training_loss,
            min_testing_loss=min_testing_loss
        )
        return accuracies, losses

    # ------------------------------------------------------
    # --- 3.1 GRADIENT DESCEND METHODS ---
    # ------------------------------------------------------
    def __full_gd(self, dataset:List) -> Dict:
        accuracy, loss = self.__back_propagation(dataset, self.__weights, self.__biaises)
        self.__update_weights(len(dataset))
        return self.__create_epoch_state(accuracy, loss)
    
    def __mini_gd(self, dataset:List) -> Dict:
        accuracies:List = []
        losses:List = []

        batch = self.__prepare_batch(dataset)
        for b in range(len(batch)):
            accuracy, loss = self.__back_propagation(batch[b], self.__weights, self.__biaises)
            accuracies.append(accuracy)
            losses.append(loss)
            self.__update_weights(self.batch_size)
        return self.__create_epoch_state(
            sum(accuracies)/len(accuracies),
            sum(losses)/len(losses)
        )

    def __stochatic_gd(self, dataset:List) -> Dict:
        accuracies:List = []
        losses:List = []

        for d in dataset:
            accuracy, loss = self.__back_propagation([d], self.__weights, self.__biaises)
            accuracies.append(accuracy)
            losses.append(loss)
            self.__update_weights(1)
        return self.__create_epoch_state(
            sum(accuracies)/len(accuracies),
            sum(losses)/len(losses)
        )

    #
    # --- 3.1.X GRADIENT DESCEND UTILS ---
    #
    def __update_weights(self, batch_size:int):
        for i in range(len(self.shape) - 1):
            self.__weights[i] = numpy.array(self.__weights[i]) - self.config.learning_rate * (numpy.array(self.__nabla_w[i]) / batch_size)
            self.__biaises[i] = numpy.array(self.__biaises[i]) - self.config.learning_rate * (numpy.array(self.__nabla_b[i]) / batch_size)
    
    # ------------------------------------------------------
    # --- 3.2 GRADIENT ACCELERATED METHODS ---
    # ------------------------------------------------------
    def __full_nag(self, dataset:List):
        self.__reset_momentum()
        self.__reset_ahead()
        
        self.__update_ahead()
        accuracy, loss = self.__back_propagation(dataset, self.__ahead_w, self.__ahead_b)
        self.__update_momentum_weights(len(dataset))
        return self.__create_epoch_state(accuracy, loss)
    
    def __mini_nag(self, dataset:List):
        accuracies:List = []
        losses:List = []

        self.__reset_momentum()
        self.__reset_ahead()
        batch = self.__prepare_batch(dataset)
        for b in range(len(batch)):
            self.__update_ahead()
            accuracy, loss = self.__back_propagation(batch[b], self.__ahead_w, self.__ahead_b)
            accuracies.append(accuracy)
            losses.append(loss)
            self.__update_momentum_weights(self.config.batch_size)
        return self.__create_epoch_state(
            sum(accuracies)/len(accuracies),
            sum(losses)/len(losses)
        )

    def __stochatic_nag(self, dataset:List):
        accuracies:List = []
        losses:List = []
        
        self.__reset_momentum()
        self.__reset_ahead()
        for d in dataset:
            self.__update_ahead()
            accuracy, loss = self.__back_propagation([d], self.__ahead_w, self.__ahead_b)
            accuracies.append(accuracy)
            losses.append(loss)
            self.__update_momentum_weights(1)
        return self.__create_epoch_state(
            sum(accuracies)/len(accuracies),
            sum(losses)/len(losses)
        )

    #
    # --- 3.2.X GRADIENT ACCELERATED UTILS ---
    #
    def __update_momentum_weights(self, batch_size:int):
        for i in range(len(self.shape) - 1):
            self.__nabla_w[i] = numpy.array(self.__nabla_w[i], dtype=float)
            self.__nabla_b[i] = numpy.array(self.__nabla_b[i], dtype=float)
            self.__momentum_w[i] = numpy.array(self.__momentum_w[i], dtype=float)
            self.__momentum_b[i] = numpy.array(self.__momentum_b[i], dtype=float)
            self.__weights[i] = numpy.array(self.__weights[i], dtype=float)
            self.__biaises[i] = numpy.array(self.__biaises[i], dtype=float)

            self.__momentum_w[i] = self.config.momentum_rate * self.__momentum_w[i] + (1 - self.config.momentum_rate)*(self.__nabla_w[i] / batch_size) 
            self.__weights[i] = numpy.array(self.__weights[i]) - self.__momentum_w[i] * self.config.learning_rate

            self.__momentum_b[i] = self.config.momentum_rate * self.__momentum_b[i] + (1 - self.config.momentum_rate)*(self.__nabla_b[i] / batch_size) 
            self.__biaises[i] = numpy.array(self.__biaises[i]) - self.__momentum_b[i] * self.config.learning_rate
    
    def __update_ahead(self):
        for i in range(len(self.shape) - 1):
            self.__ahead_w[i] = numpy.array(self.__weights[i]) - self.config.momentum_rate * self.__momentum_w[i]
            self.__ahead_b[i] = numpy.array(self.__biaises[i]) - self.config.momentum_rate * self.__momentum_b[i]

    def __reset_momentum(self):
        self.__momentum_w = [numpy.full((len(w),len(w[0])) , 0.) for w in self.__weights]
        self.__momentum_b = [numpy.full(len(w), 0.) for w in self.__weights]

    def __reset_ahead(self):
        self.__ahead_w = [[] for l in range(len(self.shape) - 1)]
        self.__ahead_b = [[] for l in range(len(self.shape) - 1)]

    # ------------------------------------------------------
    # --- 3.3 ROOT MEAN SQUARE PROPAGATION METHODS ---
    # ------------------------------------------------------
    def __full_rms_prop(self, dataset:List):
        self.__reset_velocity()
        accuracy, loss = self.__back_propagation(dataset, self.__weights, self.__biaises)
        self.__update_velocity_weights(len(dataset))
        return self.__create_epoch_state(accuracy, loss)

    def __mini_rms_prop(self, dataset:List):
        accuracies:List = []
        losses:List = []

        self.__reset_velocity()
        batch = self.__prepare_batch(dataset)
        for b in range(len(batch)):
            accuracy, loss = self.__back_propagation(batch[b], self.__weights, self.__biaises)
            accuracies.append(accuracy)
            losses.append(loss)
            self.__update_velocity_weights(self.config.batch_size)
        return self.__create_epoch_state(
            sum(accuracies)/len(accuracies),
            sum(losses)/len(losses)
        )

    def __stochatic_rms_prop(self, dataset:List):
        accuracies:List = []
        losses:List = []

        self.__reset_velocity()
        for d in dataset:
            accuracy, loss = self.__back_propagation([d], self.__weights, self.__biaises)
            accuracies.append(accuracy)
            losses.append(loss)
            self.__update_velocity_weights(1)

    #
    # --- 3.2.X ROOT MEAN SQUARE PROPAGATION UTILS ---
    #
    def __update_velocity_weights(self, batch_size:int):
        for i in range(len(self.shape) - 1):
            self.__nabla_w[i] = numpy.array(self.__nabla_w[i], dtype=float)
            self.__nabla_b[i] = numpy.array(self.__nabla_b[i], dtype=float)
            self.__velocity_w[i] = numpy.array(self.__velocity_w[i], dtype=float)
            self.__velocity_b[i] = numpy.array(self.__velocity_b[i], dtype=float)
            self.__weights[i] = numpy.array(self.__weights[i], dtype=float)
            self.__biaises[i] = numpy.array(self.__biaises[i], dtype=float)

            self.__velocity_w[i] = self.config.velocity_rate * self.__velocity_w[i] + (1 - self.config.velocity_rate)*(numpy.power(self.__nabla_w[i]/batch_size, 2))
            self.__weights[i] -= (self.config.learning_rate / (numpy.sqrt(self.__velocity_w[i])+EPS)) * (self.__nabla_w[i] / batch_size)

            self.__velocity_b[i] = self.config.velocity_rate * self.__velocity_b[i] + (1 - self.config.velocity_rate)*(numpy.power(self.__nabla_b[i]/batch_size, 2))
            self.__biaises[i] -= (self.config.learning_rate / (numpy.sqrt(self.__velocity_b[i]) + EPS)) * (self.__nabla_b[i] / batch_size)

    def __reset_velocity(self):
        self.__velocity_w = [numpy.full((len(w),len(w[0])) , 0.0) for w in self.__weights]
        self.__velocity_b = [numpy.full(len(w), 0.) for w in self.__weights]


    # ------------------------------------------------------
    # --- 3.3 ADAM METHODS ---
    # ------------------------------------------------------
    def __full_adam(self, dataset:List):
        self.__reset_momentum()
        self.__reset_velocity()
        self.__reset_ahead()
        self.__update_ahead()
        accuracy, loss = self.__back_propagation(dataset, self.__ahead_w, self.__ahead_b)
        self.__update_momentum_velocity_weights(len(dataset))
        return self.__create_epoch_state(accuracy, loss)

    def __mini_adam(self, dataset:List):
        accuracies:List = []
        losses:List = []

        self.__reset_momentum()
        self.__reset_velocity()
        self.__reset_ahead()
        batch = self.__prepare_batch(dataset)
        for b in range(len(batch)):
            self.__update_ahead()
            accuracy, loss = self.__back_propagation(batch[b], self.__ahead_w, self.__ahead_b)
            accuracies.append(accuracy)
            losses.append(loss)
            self.__update_momentum_velocity_weights(self.config.batch_size)
        return self.__create_epoch_state(
            sum(accuracies)/len(accuracies),
            sum(losses)/len(losses)
        )

    def __stochatic_adam(self, dataset:List):
        accuracies:List = []
        losses:List = []

        self.__reset_momentum()
        self.__reset_velocity()
        self.__reset_ahead()
        for d in dataset:
            self.__update_ahead()
            accuracy, loss = self.__back_propagation([d], self.__ahead_w, self.__ahead_b)
            accuracies.append(accuracy)
            losses.append(loss)
            self.__update_momentum_velocity_weights(1)
        return self.__create_epoch_state(
            sum(accuracies)/len(accuracies),
            sum(losses)/len(losses)
        )

    #
    # --- 3.2.X ADAM UTILS ---
    #
    def __update_momentum_velocity_weights(self, batch_size:int):
        for i in range(len(self.shape) - 1):
            self.__nabla_w[i] = numpy.array(self.__nabla_w[i], dtype=float)
            self.__nabla_b[i] = numpy.array(self.__nabla_b[i], dtype=float)
            self.__velocity_w[i] = numpy.array(self.__velocity_w[i], dtype=float)
            self.__velocity_b[i] = numpy.array(self.__velocity_b[i], dtype=float)
            self.__momentum_w[i] = numpy.array(self.__momentum_w[i], dtype=float)
            self.__momentum_b[i] = numpy.array(self.__momentum_b[i], dtype=float)
            self.__weights[i] = numpy.array(self.__weights[i], dtype=float)
            self.__biaises[i] = numpy.array(self.__biaises[i], dtype=float)

            self.__momentum_w[i] = self.config.momentum_rate * self.__momentum_w[i] + (1 - self.config.momentum_rate) * (self.__nabla_w[i]/batch_size)
            self.__velocity_w[i] = self.config.velocity_rate * self.__velocity_w[i] + (1 - self.config.velocity_rate) * (numpy.power(self.__nabla_w[i]/batch_size, 2))
            self.__weights[i] -= self.__momentum_w[i]/(numpy.sqrt(self.__velocity_w[i] + EPS)) * self.config.learning_rate

            self.__momentum_b[i] = self.config.momentum_rate * self.__momentum_b[i] + (1 - self.config.momentum_rate) * (self.__nabla_b[i]/batch_size)
            self.__velocity_b[i] = self.config.velocity_rate * self.__velocity_b[i] + (1 - self.config.velocity_rate) * (numpy.power(self.__nabla_b[i]/batch_size, 2))
            self.__biaises[i] -= self.__momentum_b[i]/(numpy.sqrt(self.__velocity_b[i] + EPS)) * self.config.learning_rate

    # ------------------------------------------------------
    # --- 3.X TRAINING UTILS ---
    # ------------------------------------------------------
    
    def __back_propagation(self, dataset:List, weights:List, biaises:List) -> tuple[float, float]:
        accuracies:List = []
        losses:List = []

        self.__reset_nabla()
        for d in dataset:
            dn_w = []
            dn_b = []
            if len(d["label"]) != self.shape[-1]:
                logger.info("The label need to have the same size than output layer")
                raise Exception()
            out = self.__forward_pass(d["data"], weights, biaises)
            losses.append(self.__loss_fnc(out[-1], d["label"]))
            accuracies.append(1 if step(out[-1], 0.5) == d["label"] else 0)
            delta = self.__act_fnc.delta(out[-1], d["label"])
            dn_w.insert(0, numpy.outer(numpy.array(delta), numpy.array(out[-2])))
            dn_b.insert(0, delta)
            idx = len(self.shape)-3
            while idx >= 0:
                prime = self.__act_fnc.prime(out[idx+1])
                delta = numpy.dot(numpy.transpose(weights[idx+1]), delta) * prime
                dn_w.insert(0, numpy.outer(numpy.array(delta), numpy.array(out[idx])))
                dn_b.insert(0, delta)
                idx-=1
            self.__update_nabla(dn_w, dn_b)

        return sum(accuracies)/len(accuracies), sum(losses)/len(losses)

    def __prepare_batch(self, dataset:List):
        if len(dataset) < self.config.batch_size * 2:
            logger.error("data set to small to be used with this batch size")
            raise Exception()
        ds_len = int(len(dataset) / self.config.batch_size) * self.config.batch_size
        numpy.random.shuffle(dataset)
        batch = [[dataset[i] for i in range(j, j+self.config.batch_size)] for j in range(0, ds_len, self.config.batch_size)]
        return batch

    def __forward_pass(self, input:List, weights:List, biaises:List):
        out = []
        out.append(input)
        for l in range(len(self.shape) - 2):
            out.append(self.__act_fnc.activation(self.__fire_layer(weights[l], biaises[l], out[-1])))
        out.append(self.__output_activation.activation(self.__fire_layer(weights[-1], biaises[-1], out[-1])))
        return out
    
    def __fire_layer(self, weight:List, biaises:List, input:List):
        res = [numpy.dot(w, input) + b for w,b in zip(weight, biaises)]
        return res
    
    def __reset_nabla(self):
        self.__nabla_w = [numpy.full((len(w),len(w[0])) , 0.) for w in self.__weights]
        self.__nabla_b = [numpy.full(len(w), 0.) for w in self.__weights]

    def __update_nabla(self, dn_w:Sequence, dn_b:Sequence):
        for i in range(len(self.shape) - 1):
            self.__nabla_w[i] = numpy.array(numpy.array(self.__nabla_w[i]) + numpy.array(dn_w[i]))
            self.__nabla_b[i] = numpy.array(numpy.array(self.__nabla_b[i]) + numpy.array(dn_b[i]))


    # ======================================================
    # --- 7. UTILS ---
    # ======================================================
    def __checkNetwork(self):
        print("-- NETWORK --")
        print("General options:")
        print(f" -learning rate: {self.config.learning_rate}")
        print(f" -epochs: {self.config.epoch}")
        print(f" -batch size: {self.config.batch_size}")
        print(f" -optimisation: {self.config.optimisation_name}")
        print(f" -loss: {self.config.loss_name}")
        print()
        print("--LAYERS--")
        print(" -weights:")
        print(self.__weights)
        print(" -biaises:")
        print(self.__biaises)
        return
    
    def fire(self, input:numpy.array) -> numpy.array:
        act_input = input
        for l in range(len(self.shape) - 2):
            act_input = numpy.array(self.__act_fnc.activation(self.__fire_layer(self.__weights[l], self.__biaises[l], act_input)))
        act_input = numpy.array(self.__output_activation.activation(self.__fire_layer(self.__weights[-1], self.__biaises[-1], act_input)))
        return act_input
    
    def __save_training(
            self,
            accuracy:float=None,
            precision:float=None,
            recall:float=None,
            f1:float=None,
            time:float=None,
            min_training_loss:float=None,
            min_testing_loss:float=None
    ):
        save_to_history(
            optimizer=self.config.optimisation_name,
            activation_function=self.config.activation_name,
            loss_function=self.config.loss_name,
            epoch=self.config.epoch,
            learning_rate=self.config.learning_rate,
            network_shape=self.config.shape,
            min_training_loss=min_training_loss,
            min_testing_loss=min_testing_loss,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1=f1,
            time=time
        )
    
    def __print_epoch_state(
            self,
            epoch:int,
            training_accuracy:float,
            training_loss:float,
            testing_accuracy:float,
            testing_loss:float
    ):
        message = (
            "===============================\n"
            f"At epoch {epoch}/{self.config.epoch}"
            "=== Training ===\n"
            f"Accuracy: {training_accuracy},\n"
            f"Loss: {training_loss},\n"
            "=== Testing ===\n"
            f"Accuracy: {testing_accuracy},\n"
            f"Loss: {testing_loss},\n"
            "==============================\n"
        )
        logger.info(message)

    def __create_epoch_state(self, accuracy:float, loss:float) -> Dict:
        return {
            "accuracy": accuracy,
            "loss": loss
        }
        
    