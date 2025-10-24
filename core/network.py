
import yaml
import time
import math
import numpy
import matplotlib as plt
import ml_tools.utils as Utils
import ml_tools.activations as Activations


from typing import List, Dict
from utils.logger import Logger
logger = Logger()

EPS = 1e-8

# ==========================================================
# NETWORK CLASS
# ==========================================================
class Network:
    
    # ======================================================
    # --- 1. CLASS ATTRIBUTES / CONFIG DEFAULTS ---
    # =====================================================
    option_visu_training: bool = False
    option_visu_loss: bool = False
    option_visu_accuracy: bool = False

    learning_rate: float = None
    epoch: int = None
    batch_size: int = None
    momentum_rate:float = 0.9
    velocity_rate:float = 0.9

    error_threshold:float = 0. #A IMPLEMENTER OPTIONNAL

    # Components and functions
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

    # Network structure
    __weights:List = None
    __biaises:List = None
    __shape:List = None

    # ======================================================
    # --- 2. INITIALIZATION ---
    # ======================================================
    def __init__(self, init_file_path: str):
        """
        Load YAML config and initialize network structure and parameters.
        """
        config_general, config_archi = self.__load_config(init_file_path)

        logger.info("Network initialization starting...")
        logger.info("Configuration starting...")
        self.__init_mandatories(config_general)
        self.__init_optimisation(config_general)
        self.__init_evaluation(config_general)
        self.__init_loss(config_general)
        self.__init_activation(config_general)
        self.__init_initialisation(config_general)
        logger.info("Configuration complete...")
        logger.info("Layers initialization starting...")
        self.__init_layers(config_archi)
        logger.info("Layers initialization complete...")
        logger.info("Network initialization complete...")

    # ------------------------------------------------------
    # --- 2.1 CONFIGURATION INITIALISATION ---
    # ------------------------------------------------------
    def __load_config(self, path:str):
        try:
            f = open(path, 'r')
        except:
            logger.error(f"Can not open the file {path}")
            raise Exception()
        dataStr = f.read()
        try:
            config = yaml.safe_load(dataStr)
            config_general = config["general"]
            config_archi = config["architecture"]
        except:
            logger.error(f"The config file need to be a .yaml with atleast general and architecture keys")
            raise Exception()
        return config_general, config_archi

    def __init_mandatories(self, config: dict):
        try:
            self.learning_rate = float(config["learning rate"])
            self.epoch = int(config["epochs"])
        except KeyError:
            logger.error("Missing key in the config file")
            raise Exception()
        logger.info("Mandatories OK..")
        
    def __init_optimisation(self, config: dict):
        try:
            self.optimisation_name = '_'.join(str.lower(config["optimisation"]).split())
        except KeyError:
            self.optimisation_name = "gradient_descent"
        try:
            self.__optimisation_fnc = getattr(self, self.optimisation_name)
        except KeyError:
            logger.error(f"Optimisation function {self.optimisation_name} is unknown")
            raise Exception()
        try:
            self.batch_size = int(config["batch size"])
        except KeyError:
            if (self.optimisation_name == "stochastic_gradient_descent"):
                logger.error("Missing batch size for the SGD")
                raise Exception()
        try:
            self.momentum_rate = float(config["beta 1"])
        except KeyError:
            if self.optimisation_name == "nesterov momentum" or self.optimisation_name == "adam":
                logger.error("Missing beta 1 for the optimizer")
                raise Exception()
        try:
            self.beta_2 = float(config["beta 2"])
        except KeyError:
            if self.optimisation_name == "rms_prop" or self.optimisation_name == "adam":
                logger.error("Missing beta 2 for the optimizer")
                raise Exception()
        logger.info("Optimisation OK..")
        
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
        logger.info("Evaluation OK..")

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
        logger.info("Activation OK..")
        
    def __init_initialisation(self, config:Dict):
        try:
            self.initialisation_name = '_'.join(str.lower(config["initialisation"]).split())
        except KeyError:
            self.initialisation_name = "he_normal"
        try:
            import ml_tools.initialisations as Initialisations
            self.__initialisation_fnc = getattr(Initialisations, self.initialisation_name)
        except AttributeError:
            logger.error(f"Initialisation function {self.initialisation_name} is unknown")
        logger.info("Initialisation OK..")

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
        logger.info("loss OK..")

    # ------------------------------------------------------
    # --- 2.2 LAYER INITIALIZATION ---
    # ------------------------------------------------------
    def __init_layers(self, archi: dict):
        self.__weights = []
        self.__biaises = []
        self.__shape = []
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

    # ======================================================
    # --- 3. TRAINING METHODS ---
    # ======================================================
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
            self.__optimisation_fnc(ds_train)
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

    # ------------------------------------------------------
    # --- 3.1 GRADIENT DESCEND METHODS ---
    # ------------------------------------------------------
    def full_gd(self, dataset:List):
        nabla_w, nabla_b = self.back_propagation(dataset, self.__weights, self.__biaises)
        self.update_weights(nabla_w, nabla_b, len(dataset))
    
    def mini_gd(self, dataset:List):
        batch = self.prepare_batch(dataset)
        for b in range(len(batch)):
            nabla_w, nabla_b = self.back_propagation(batch[b], self.__weights, self.__biaises)
            self.update_weights(nabla_w, nabla_b, self.batch_size)

    def stochatic_gd(self, dataset:List):
        for d in dataset:
            nabla_w, nabla_b = self.back_propagation([d], self.__weights, self.__biaises)
            self.update_weights(nabla_w, nabla_b, 1)

    #
    # --- 3.1.X GRADIENT DESCEND UTILS ---
    #
    def update_weights(self, nabla_w:List, nabla_b:List, batch_size:int):
        for i in range(len(self.__shape) - 1):
            self.__weights[i] = numpy.array(self.__weights[i]) - self.learning_rate * (numpy.array(nabla_w[i]) / batch_size)
            self.__biaises[i] = numpy.array(self.__biaises[i]) - self.learning_rate * (numpy.array(nabla_b[i]) / batch_size)
    
    # ------------------------------------------------------
    # --- 3.2 GRADIENT ACCELERATED METHODS ---
    # ------------------------------------------------------
    def full_nag(self, dataset:List):
        momentum_w = [numpy.full((len(w),len(w[0])) , 0.) for w in self.__weights]
        momentum_b = [numpy.full(len(w), 0.) for w in self.__weights]
        ahead_w = [[] for l in range(len(self.__shape) - 1)]
        ahead_b = [[] for l in range(len(self.__shape) - 1)]
        for i in range(len(self.__shape) - 1):
            ahead_w[i] = numpy.array(self.__weights[i]) - self.momentum_rate * momentum_w[i]
            ahead_b[i] = numpy.array(self.__biaises[i]) - self.momentum_rate * momentum_b[i]
        nabla_w, nabla_b = self.back_propagation(dataset, ahead_w, ahead_b)
        momentum_w, momentum_b = self.update_momentum_weights(momentum_w, momentum_b, nabla_w, nabla_b, len(dataset))
    
    def mini_nag(self, dataset:List):
        momentum_w = [numpy.full((len(w),len(w[0])) , 0.) for w in self.__weights]
        momentum_b = [numpy.full(len(w), 0.) for w in self.__weights]
        ahead_w = [[] for l in range(len(self.__shape) - 1)]
        ahead_b = [[] for l in range(len(self.__shape) - 1)]
        batch = self.prepare_batch(dataset)
        for b in range(len(batch)):
            for i in range(len(self.__shape) - 1):
                ahead_w[i] = numpy.array(self.__weights[i]) - self.momentum_rate * momentum_w[i]
                ahead_b[i] = numpy.array(self.__biaises[i]) - self.momentum_rate * momentum_b[i]
            nabla_w, nabla_b = self.back_propagation(batch[b], ahead_w, ahead_b)
            momentum_w, momentum_b = self.update_momentum_weights(momentum_w, momentum_b, nabla_w, nabla_b, self.batch_size)

    def stochatic_nag(self, dataset:List):
        momentum_w = [numpy.full((len(w),len(w[0])) , 0.) for w in self.__weights]
        momentum_b = [numpy.full(len(w), 0.) for w in self.__weights]
        ahead_w = [[] for l in range(len(self.__shape) - 1)]
        ahead_b = [[] for l in range(len(self.__shape) - 1)]
        for d in dataset:
            for i in range(len(self.__shape) - 1):
               ahead_w[i] = numpy.array(self.__weights[i]) - self.momentum_rate * momentum_w[i]
               ahead_b[i] = numpy.array(self.__biaises[i]) - self.momentum_rate * momentum_b[i]
            nabla_w, nabla_b = self.back_propagation([d], ahead_w, ahead_b)
            momentum_w, momentum_b = self.update_momentum_weights(momentum_w, momentum_b, nabla_w, nabla_b, 1)

    #
    # --- 3.2.X GRADIENT ACCELERATED UTILS ---
    #
    def update_momentum_weights(self, momentum_w:List, momentum_b:List, nabla_w:List, nabla_b:List, batch_size:int):
        for i in range(len(self.__shape) - 1):
            nabla_w[i] = numpy.array(nabla_w[i], dtype=float)
            nabla_b[i] = numpy.array(nabla_b[i], dtype=float)
            momentum_w[i] = numpy.array(momentum_w[i], dtype=float)
            momentum_b[i] = numpy.array(momentum_b[i], dtype=float)
            self.__weights[i] = numpy.array(self.__weights[i], dtype=float)
            self.__biaises[i] = numpy.array(self.__biaises[i], dtype=float)

            momentum_w[i] = self.momentum_rate * momentum_w[i] + (1 - self.momentum_rate)*(nabla_w[i] / batch_size) 
            self.__weights[i] = numpy.array(self.__weights[i]) - momentum_w[i] * self.learning_rate

            momentum_b[i] = self.momentum_rate * momentum_b[i] + (1 - self.momentum_rate)*(nabla_b[i] / batch_size) 
            self.__biaises[i] = numpy.array(self.__biaises[i]) - momentum_b[i] * self.learning_rate
        
        return momentum_w, momentum_b
    
    # ------------------------------------------------------
    # --- 3.3 ROOT MEAN SQUARE PROPAGATION METHODS ---
    # ------------------------------------------------------
    def full_rms_prop(self, dataset:List):
        velocity_w = [numpy.full((len(w),len(w[0])) , 0.0) for w in self.__weights]
        velocity_b = [numpy.full(len(w), 0.) for w in self.__weights]
        nabla_w, nabla_b = self.back_propagation(dataset, self.__weights, self.__biaises)
        velocity_w, velocity_b = self.update_velocity_weights(velocity_w, velocity_b, nabla_w, nabla_b, len(dataset))

    def mini_rms_prop(self, dataset:List):
        velocity_w = [numpy.full((len(w),len(w[0])) , 0.) for w in self.__weights]
        velocity_b = [numpy.full(len(w), 0.) for w in self.__weights]
        batch = self.prepare_batch(dataset)
        for b in range(len(batch)):
            nabla_w, nabla_b = self.back_propagation(batch[b], self.__weights, self.__biaises)
            velocity_w, velocity_b = self.update_velocity_weights(velocity_w, velocity_b, nabla_w, nabla_b, self.batch_size)

    def stochatic_rms_prop(self, dataset:List):
        velocity_w = [numpy.full((len(w),len(w[0])) , 0.) for w in self.__weights]
        velocity_b = [numpy.full(len(w), 0.) for w in self.__weights]
        for d in dataset:
            nabla_w, nabla_b = self.back_propagation([d], self.__weights, self.__biaises)
            velocity_w, velocity_b = self.update_velocity_weights(velocity_w, velocity_b, nabla_w, nabla_b, 1)

    #
    # --- 3.2.X ROOT MEAN SQUARE PROPAGATION UTILS ---
    #
    def update_velocity_weights(self, velo_w:List, velo_b:List, nabla_w:List, nabla_b:List, batch_size:int):
        for i in range(len(self.__shape) - 1):
            nabla_w[i] = numpy.array(nabla_w[i], dtype=float)
            nabla_b[i] = numpy.array(nabla_b[i], dtype=float)
            velo_w[i] = numpy.array(velo_w[i], dtype=float)
            velo_b[i] = numpy.array(velo_b[i], dtype=float)
            self.__weights[i] = numpy.array(self.__weights[i], dtype=float)
            self.__biaises[i] = numpy.array(self.__biaises[i], dtype=float)

            velo_w[i] = self.velocity_rate * velo_w[i] + (1 - self.velocity_rate)*(numpy.power(nabla_w[i]/batch_size, 2))
            self.__weights[i] -= (self.learning_rate / (numpy.sqrt(velo_w[i])+EPS)) * (nabla_w[i] / batch_size)

            velo_b[i] = self.velocity_rate * velo_b[i] + (1 - self.velocity_rate)*(numpy.power(nabla_b[i]/batch_size, 2))
            self.__biaises[i] -= (self.learning_rate / (numpy.sqrt(velo_b[i]) + EPS)) * (nabla_b[i] / batch_size)

        return velo_w, velo_b

    # ------------------------------------------------------
    # --- 3.3 ADAM METHODS ---
    # ------------------------------------------------------
    def full_adam(self, dataset:List):
        momentum = {"w":[numpy.full((len(w),len(w[0])) , 0.) for w in self.__weights], "b":[numpy.full(len(w), 0.) for w in self.__weights]}
        velocity = {"w":[numpy.full((len(w),len(w[0])) , 0.) for w in self.__weights], "b":[numpy.full(len(w), 0.) for w in self.__weights]}
        a_head = {"w":[[] for l in range(len(self.__shape) - 1)],"b":[[] for l in range(len(self.__shape) - 1)]}
        for i in range(len(self.__shape) - 1):
            a_head["w"][i] = numpy.array(self.__weights[i]) - self.momentum_rate * momentum["w"][i]
            a_head["b"][i] = numpy.array(self.__biaises[i]) - self.momentum_rate * momentum["b"][i]
        nabla_w, nabla_b = self.back_propagation(dataset, a_head["w"], a_head["b"])
        momentum, velocity = self.update_momentum_velocity_weights(momentum, velocity, {"w":nabla_w,"b":nabla_b}, len(dataset))

    def mini_adam(self, dataset:List):
        momentum = {"w":[numpy.full((len(w),len(w[0])) , 0.) for w in self.__weights], "b":[numpy.full(len(w), 0.) for w in self.__weights]}
        velocity = {"w":[numpy.full((len(w),len(w[0])) , 0.) for w in self.__weights], "b":[numpy.full(len(w), 0.) for w in self.__weights]}
        a_head = {"w":[[] for l in range(len(self.__shape) - 1)],"b":[[] for l in range(len(self.__shape) - 1)]}
        batch = self.prepare_batch(dataset)
        for b in range(len(batch)):
            for i in range(len(self.__shape) - 1):
                a_head["w"][i] = numpy.array(self.__weights[i]) - self.momentum_rate * momentum["w"][i]
                a_head["b"][i] = numpy.array(self.__biaises[i]) - self.momentum_rate * momentum["b"][i]
            nabla_w, nabla_b = self.back_propagation(batch[b], a_head["w"], a_head["b"])
            momentum, velocity = self.update_momentum_velocity_weights(momentum, velocity, {"w":nabla_w,"b":nabla_b}, self.batch_size)

    def stochatic_adam(self, dataset:List):
        momentum = {"w":[numpy.full((len(w),len(w[0])) , 0.) for w in self.__weights], "b":[numpy.full(len(w), 0.) for w in self.__weights]}
        velocity = {"w":[numpy.full((len(w),len(w[0])) , 0.) for w in self.__weights], "b":[numpy.full(len(w), 0.) for w in self.__weights]}
        a_head = {"w":[[] for l in range(len(self.__shape) - 1)],"b":[[] for l in range(len(self.__shape) - 1)]}
        for d in dataset:
            for i in range(len(self.__shape) - 1):
                a_head["w"][i] = numpy.array(self.__weights[i]) - self.momentum_rate * momentum["w"][i]
                a_head["b"][i] = numpy.array(self.__biaises[i]) - self.momentum_rate * momentum["b"][i]
            nabla_w, nabla_b = self.back_propagation([d], a_head["w"], a_head["b"])
            momentum, velocity = self.update_momentum_velocity_weights(momentum, velocity, {"w":nabla_w,"b":nabla_b}, 1)
    #
    # --- 3.2.X ADAM UTILS ---
    #
    def update_momentum_velocity_weights(self, momentum:Dict, velo:Dict, nabla:Dict, batch_size:int):
        for i in range(len(self.__shape) - 1):
            nabla["w"][i] = numpy.array(nabla["w"][i], dtype=float)
            nabla["b"][i] = numpy.array(nabla["b"][i], dtype=float)
            velo["w"][i] = numpy.array(velo["w"][i], dtype=float)
            velo["b"][i] = numpy.array(velo["b"][i], dtype=float)
            momentum["w"][i] = numpy.array(momentum["w"][i], dtype=float)
            momentum["b"][i] = numpy.array(momentum["b"][i], dtype=float)
            self.__weights[i] = numpy.array(self.__weights[i], dtype=float)
            self.__biaises[i] = numpy.array(self.__biaises[i], dtype=float)

            momentum["w"][i] = self.momentum_rate * momentum["w"][i] + (1 - self.momentum_rate) * (nabla["w"][i]/batch_size)
            velo["w"][i] = self.velocity_rate * velo["w"][i] + (1 - self.velocity_rate) * (numpy.power(nabla["w"][i]/batch_size, 2))
            self.__weights[i] -= momentum["w"][i]/(numpy.sqrt(velo["w"][i] + EPS)) * self.learning_rate

            momentum["b"][i] = self.momentum_rate * momentum["b"][i] + (1 - self.momentum_rate) * (nabla["b"][i]/batch_size)
            velo["b"][i] = self.velocity_rate * velo["b"][i] + (1 - self.velocity_rate) * (numpy.power(nabla["b"][i]/batch_size, 2))
            self.__biaises[i] -= momentum["b"][i]/(numpy.sqrt(velo["b"][i] + EPS)) * self.learning_rate

        return momentum, velo

    # ------------------------------------------------------
    # --- 3.X TRAINING UTILS ---
    # ------------------------------------------------------
    
    def back_propagation(self, dataset:List, weights:List, biaises:List):
        nabla_w = [numpy.full((len(w),len(w[0])) , 0.) for w in self.__weights]
        nabla_b = [numpy.full(len(w), 0.) for w in self.__weights]
        count = 0
        for d in dataset:
            dn_w = []
            dn_b = []
            if len(d["label"]) != self.__shape[-1]:
                logger.info("The label need to have the same size than output layer")
                raise Exception()
            out = self.forward_pass(d["data"], weights, biaises)
            delta = self.__activation.delta(out[-1], d["label"])
            dn_w.insert(0, numpy.outer(numpy.array(delta), numpy.array(out[-2])))
            dn_b.insert(0, delta)
            idx = len(self.__shape)-3
            while idx >= 0:
                prime = self.__activation.prime(out[idx+1])
                delta = numpy.dot(numpy.transpose(weights[idx+1]), delta) * prime
                dn_w.insert(0, numpy.outer(numpy.array(delta), numpy.array(out[idx])))
                dn_b.insert(0, delta) 
                idx-=1
            for i in range(len(self.__shape) - 1):
                nabla_w[i] = numpy.array(numpy.array(nabla_w[i]) + numpy.array(dn_w[i]))
                nabla_b[i] = numpy.array(numpy.array(nabla_b[i]) + numpy.array(dn_b[i]))
        return nabla_w, nabla_b

    def prepare_batch(self, dataset:List):
        if len(dataset) < self.batch_size * 2:
            logger.error("data set to small to be used with this batch size")
            raise Exception()
        ds_len = int(len(dataset) / self.batch_size) * self.batch_size
        numpy.random.shuffle(dataset)
        batch = [[dataset[i] for i in range(j, j+self.batch_size)] for j in range(0, ds_len, self.batch_size)]
        return batch

    def forward_pass(self, input:List, weights:List, biaises:List):
        out = []
        out.append(input)
        for l in range(len(self.__shape) - 2):
            out.append(self.__activation.activation(self.fire_layer(weights[l], biaises[l], out[-1])))
        out.append(self.__output_activation.activation(self.fire_layer(weights[-1], biaises[-1], out[-1])))
        return out
    
    def fire_layer(self, weight:List, biaises:List, input:List):
        res = [numpy.dot(w, input) + b for w,b in zip(weight, biaises)]
        return res


    # ======================================================
    # --- 7. DEBUG / CHECK ---
    # ======================================================
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
    
    def fire(self, input:numpy.array) -> numpy.array:
        act_input = input
        for l in range(len(self.__shape) - 2):
            act_input = numpy.array(self.__activation.activation(self.fire_layer(self.__weights[l], self.__biaises[l], act_input)))
        act_input = numpy.array(self.__output_activation.activation(self.fire_layer(self.__weights[-1], self.__biaises[-1], act_input)))
        return act_input
    


    



        
    