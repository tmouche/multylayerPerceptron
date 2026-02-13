from core.layer import Layer
from dataclasses import dataclass
from ml_tools.utils import step
from ml_tools.activations import __Activation
from typing import (
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
)
from utils.decorator import call_decorator
from utils.exception import (
    Format,
    NetworkException,
    NetworkInit,
    NetworkLayerCount,
    LayerActivation,
    LayerInit,
    LayerInitializer,
    UnexpectedException
)
from utils.history import save_to_history
from utils.logger import Logger
from utils.types import ArrayF, FloatT

import inspect
import ml_tools.initialisations as Initialisations
import ml_tools.activations as Activations
import ml_tools.losses as Losses
import numpy as np
import numpy.typing as npt
import time
import json

logger = Logger()

EPS = 1e-8

@dataclass
class NetworkConfig:

    # Obligatoires
    shape: List[int]
    evaluation: callable
    activation_name: str
    loss_name: str
    optimisation_name: str
    output_activation_name: str
    initialisation_name: str

    # Optionnels
    learning_rate: float = 0.001
    epoch: int = 50
    batch_size: Optional[int] = None

    # momentum_rate: float = 0.9
    # velocity_rate: float = 0.9

    loss_threshold: Optional[float] = None



# ==========================================================
# NETWORK CLASS
# ==========================================================
class Network:
    
    # ======================================================
    # --- 1. CLASS ATTRIBUTES / CONFIG DEFAULTS ---
    # =====================================================
    option_visu_training: bool = False

    config:NetworkConfig
    _is_apply:bool = False

    # Components and functions
    _opti_fnc:callable = None
    _eval_fnc:callable = None
    _loss_fnc:callable = None
    _act_obj = None
    _output_act_obj = None
    _init_fnc:callable = None

    # Network structure
    layers: List[Layer]

    batch_size: int

    learning_rate: FloatT
    momentum_rate: FloatT = 0.9
    velocity_rate: FloatT = 0.9

    weights: npt.ArrayLike[npt.ArrayLike[ArrayF]]
    biaises: npt.ArrayLike[ArrayF]

    _nabla_w:Sequence = None
    _nabla_b:Sequence = None

    _momentum_w:Sequence = None
    _momentum_b:Sequence = None

    _ahead_w:Sequence = None
    _ahead_b:Sequence = None

    _velocity_w:Sequence = None
    _velocity_b:Sequence = None

    # ======================================================
    # --- 2. INITIALIZATION ---
    # ======================================================

    # def __init__(self, config:NetworkConfig):
    #     self.config = config
    #     self._apply_config()

    # def _apply_config(self):
    #     logger.info("Network initialization starting...")
    #     logger.info("Configuration starting...")
    #     self._check_mandatories()
    #     self._check_optimisation()
    #     self._check_activation()
    #     self._init_initialisation()
    #     self._eval_fnc = self.config.evaluation
    #     logger.info("Configuration complete...")
    #     logger.info("Layers initialization starting...")
    #     self._init_layers()
    #     logger.info("Layers initialization complete...")
    #     logger.info("Network initialization complete...")
    #     self._is_apply = True


    def __init__(self, layers :List[Layer]):
        
        for a in layers:
            if isinstance(a, Layer) == False:
                logger.error(f"Unrecognized argument passed: {a}")
                raise Format(context="Layer")
            self.layers.append(a)

        try:
            self.__init_layers()
        except (LayerInit, NetworkInit) as iniErr:
            logger.error(iniErr)
            raise NetworkInit(str(iniErr))
        except Exception as e:
            logger.error(f"Unexpected exception: {e}")
            raise UnexpectedException()

            

    # ------------------------------------------------------
    # --- 2.1 CONFIGURATION INITIALISATION ---
    # ------------------------------------------------------

    def _check_mandatories(self):
        if self.config.learning_rate is None or self.config.learning_rate <= 0:
            logger.error("Learning rate cannot be negative or egal to 0")
            raise Exception()
        if self.config.epoch is None or self.config.epoch <= 0:
            logger.error("The number of epoch cannot be negative or egal to 0")
            raise Exception()
        
    def _check_optimisation(self):
        opti_name = '_' + '_'.join(str.lower(self.config.optimisation_name).split())
        try:
            self._opti_fnc = getattr(self, opti_name)
        except AttributeError as e:
            logger.error(f"Optimisation function {self.config.optimisation_name} is unknown")
            raise e
        
        if self.config.batch_size is None:
            if "mini" in self._opti_name:
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
    
    def _check_activation(self):
        loss_name = '_'.join(str.lower(self.config.loss_name).split())
        act_name = '_'.join(str.lower(self.config.activation_name).split())
        act_output_name = '_'.join(str.lower(self.config.output_activation_name).split())
        if act_name == "softmax":
            logger.error("Softmax can not be used as activation for hidden layers")
            raise Exception()
        try:
            self._loss_fnc = getattr(Losses, loss_name)
        except AttributeError:
            logger.error(f"Activation function {act_name} unknown")
            raise Exception()
        try:
            self._act_obj = getattr(Activations, act_name)(loss_name)
        except AttributeError:
            logger.error(f"Activation function {act_name} unknown")
            raise Exception()
        try:
            self._output_act_obj = getattr(Activations, act_output_name)(loss_name)
        except AttributeError:
            logger.error(f"Activation function {act_output_name} unknown")
            raise Exception()
        
    def _init_initialisation(self):
        init_name = '_'.join(str.lower(self.config.initialisation_name).split())
        try:
            self._init_fnc = getattr(Initialisations, init_name)
        except AttributeError:
            logger.error(f"Initialisation function {init_name} is unknown")

    # ------------------------------------------------------
    # --- 2.2 LAYER INITIALIZATION ---
    # ------------------------------------------------------
    # def _init_layers(self):
    #     self._weights = []
    #     self._biaises = []
        
    #     if len(self.config.shape) < 3:
    #         logger.error("The Network has to have at least 3 layers")
    #         raise Exception()

    #     for i in range(1, len(self.config.shape)):
    #         if self.config.shape[i] <= 0 or self.config.shape[i-1] <= 0:
    #             logger.error("A layer size can not be negativ or egal to 0")
    #             raise Exception()
    #         self._create_layers(self.config.shape[i], self.config.shape[i-1])
                
    #     try:
    #         previous_size = self.config.shape[0]
    #         output_size = self.config.shape[-1]
    #     except KeyError:
    #         logger.error(f"Missing mandatories keys (architecture.input.size, architecture.output.size)")
    #         raise Exception()
    #     if previous_size <= 0 or output_size <= 0:
    #         logger.error(f"The input size and the output size cannot be lower than 1")
    #         raise Exception()
    #     self.output_activation_name = '_'.join(str.lower(self.config.output_activation_name).split())
    #     if self.output_activation_name == "softmax" and output_size < 2:
    #         logger.error("Softmax needs atleast two output neurons")
    #         raise Exception()
    
    # def _create_layers(self, size:int, prev_size:int):
    #     self._weights.append(self._init_fnc(shape=(size,prev_size)))
    #     self._biaises.append(self._init_fnc(shape=(size)))

    def __init_layers(self):
        """
        Initialize network weights and biases for all layers.

        Raises:
            NetworkLayerCount:
                If the network contains fewer than three layers.

            LayerActivation:
                If a layer is missing an activation function.

            LayerInitializer:
                If a layer is missing a weight initializer.

        Logs:
            - Error if the network does not contain enough layers.
            - Error if activation or initializer is missing for a layer.
        """
        self.weights = np.ndarray(0)
        self.biaises = np.ndarray(0)


        if len(self.layers) < 3:
            logger.error(f"Not enough layers")
            raise NetworkLayerCount()
        
        for i in range(1, len(self.layers)):
            previous_size: int = self.layers[i-1]
            size: int = self.layers[i]

            if not self.layers[i].activation:
                logger.error(f"Missing Activation in layer n*{i+1}")
                raise LayerActivation(context="Missing")
            if not self.layers[i].initializer:
                logger.error(f"Missing Initializer in layer n*{i+1}")
                raise LayerInitializer(context="Missing")
            self.weights = np.append(self.weights, self.layers[i].initializer(shape=(size, previous_size)))
            self.biaises = np.append(self.biaises, self.layers[i].initializer(shape=(size)))

            
    

    # ======================================================
    # --- 3. TRAINING METHODS ---
    # ======================================================
    @call_decorator
    def train(
            self,
            ds_train:List[Dict[str, List[float]]], 
            ds_test:List[Dict[str, List[float]]]
        ) -> tuple[Sequence, Sequence]:
        accuracies:Dict = {'testing': [], 'training': []}
        losses:Dict = {'testing': [], 'training': []}
        
        if len(ds_train[0]["data"]) != self.config.shape[0]:
            logger.error(f"Missmatch between the number of input ({len(ds_train[0]['data'])}) and the number of expected input ({self.config.shape[0]})")
            raise Exception()
        if len(ds_train[0]["label"]) != self.config.shape[-1]:
            logger.error(f"Missmatch between the number of output ({len(ds_train[0]['label'])}) and the number of expected output ({self.config.shape[-1]})")
            raise Exception()

        logger.info("Starting training...")
        start_time = time.perf_counter()
        for e in range(self.config.epoch):
            training:Dict = self._opti_fnc(ds_train)
            testing:Dict = self._eval_fnc(self, self._loss_fnc, ds_test)
            
            accuracies['testing'].append(testing.get("accuracy"))
            accuracies['training'].append(training.get("accuracy"))
            losses['testing'].append(testing.get("loss"))
            losses['training'].append(training.get("loss"))
            
            if self.option_visu_training:
                self._print_epoch_state(
                    epoch=e,
                    training_accuracy=training.get("accuracy"),
                    training_loss=training.get("loss"),
                    testing_accuracy= testing.get("accuracy"),
                    testing_loss= testing.get("loss")
                )
            if self.config.loss_threshold and abs(testing["loss"]) < self.config.loss_threshold:
                break
        time_stamp = time.perf_counter() - start_time    
        self._save_training(
            accuracy=testing["accuracy"],
            precision=testing["precision"],
            recall=testing["recall"], 
            f1=testing["f1"], 
            time=time_stamp, 
            min_training_loss=min(losses["training"]),
            min_testing_loss=min(losses["testing"])
        )
        return accuracies, losses

    # ------------------------------------------------------
    # --- 3.1 GRADIENT DESCEND METHODS ---
    # ------------------------------------------------------
    def _full_gd(self, dataset:List) -> Dict:
        accuracy, loss = self._back_propagation(dataset, self.weights, self.biaises)
        self._gd_update_weights(len(dataset))
        return self._create_epoch_state(accuracy, loss)
    
    def _mini_gd(self, dataset:List) -> Dict:
        accuracies:List = []
        losses:List = []

        batch = self._prepare_batch(dataset)
        for b in range(len(batch)):
            accuracy, loss = self._back_propagation(batch[b], self.weights, self.biaises)
            accuracies.append(accuracy)
            losses.append(loss)
            self._gd_update_weights(self.config.batch_size)
        return self._create_epoch_state(
            sum(accuracies)/len(accuracies),
            sum(losses)/len(losses)
        )

    def _stochastic_gd(self, dataset:List) -> Dict:
        accuracies:List = []
        losses:List = []

        # IL FAUT RAJOUTER LE STOCHATIC ICI 
        for d in dataset:
            accuracy, loss = self._back_propagation([d], self.weights, self.biaises)
            accuracies.append(accuracy)
            losses.append(loss)
            self._gd_update_weights(1)
        return self._create_epoch_state(
            sum(accuracies)/len(accuracies),
            sum(losses)/len(losses)
        )

    #
    # --- 3.1.X GRADIENT DESCEND UTILS ---
    #
    def _gd_update_weights(self, batch_size:int):
        for i in range(len(self.config.shape) - 1):
            self.weights[i] -= (self.config.learning_rate * (self._nabla_w[i] / batch_size))
            self.biaises[i] -= (self.config.learning_rate * (self._nabla_b[i] / batch_size))


    # ------------------------------------------------------
    # --- 3.2 GRADIENT ACCELERATED METHODS ---
    # ------------------------------------------------------

    def _full_nag(self, dataset:List):
        self._nag_init_momentum()
        self._nag_init_ahead()
        self._update_ahead()
        accuracy, loss = self._back_propagation(dataset, self._ahead_w, self._ahead_b)
        self._nag_update_weights(len(dataset))
        return self._create_epoch_state(accuracy, loss)
    
    def _mini_nag(self, dataset:List):
        accuracies:List = []
        losses:List = []

        self._nag_init_momentum()
        self._nag_init_ahead()
        batch = self._prepare_batch(dataset)
        for b in range(len(batch)):
            self._nag_update_ahead()
            accuracy, loss = self._back_propagation(batch[b], self._ahead_w, self._ahead_b)
            accuracies.append(accuracy)
            losses.append(loss)
            self._nag_update_weights(self.config.batch_size)
        return self._create_epoch_state(
            sum(accuracies)/len(accuracies),
            sum(losses)/len(losses)
        )

    def _stochatic_nag(self, dataset:List):
        accuracies:List = []
        losses:List = []
        
        self._nag_init_momentum()
        self._nag_init_ahead()
        for d in dataset:
            self.__nag_update_ahead()
            accuracy, loss = self._back_propagation([d], self._ahead_w, self._ahead_b)
            accuracies.append(accuracy)
            losses.append(loss)
            self._nag_update_weights(1)
        return self._create_epoch_state(
            sum(accuracies)/len(accuracies),
            sum(losses)/len(losses)
        )

    #
    # --- 3.2.X GRADIENT ACCELERATED UTILS ---
    #

    def _nag_init_momentum(self):
        self._momentum_w = [numpy.full((len(w),len(w[0])) , 0.) for w in self.weights]
        self._momentum_b = [numpy.full(len(w), 0.) for w in self.weights]

    def _nag_init_ahead(self):
        self._ahead_w = [[] for l in range(len(self.config.shape) - 1)]
        self._ahead_b = [[] for l in range(len(self.config.shape) - 1)]

    def _nag_update_ahead(self):
        for i in range(len(self.config.shape) - 1):
            self._ahead_w[i] = numpy.array(self.weights[i]) - (self.config.momentum_rate * self.config.learning_rate * self._momentum_w[i])
            self._ahead_b[i] = numpy.array(self.biaises[i]) - (self.config.momentum_rate * self.config.learning_rate * self._momentum_b[i])

    def _nag_update_weights(self, batch_size:int):
        for i in range(len(self.config.shape) - 1):
            self._momentum_w[i] = self.config.momentum_rate * self._momentum_w[i] + (self._nabla_w[i] / batch_size)
            self.weights[i] = numpy.array(self.weights[i]) - (self.config.learning_rate * self._momentum_w[i])

            self._momentum_b[i] = self.config.momentum_rate * self._momentum_b[i] + (self._nabla_b[i] / batch_size) 
            self.biaises[i] = numpy.array(self.biaises[i]) - (self.config.learning_rate * self._momentum_b[i])


    # ------------------------------------------------------
    # --- 3.3 ROOT MEAN SQUARE PROPAGATION METHODS ---
    # ------------------------------------------------------
    def _full_rms_prop(self, dataset:List):
        self._rms_init_velocity()
        accuracy, loss = self._back_propagation(dataset, self.weights, self.biaises)
        self._rms_update_weights(len(dataset))
        return self._create_epoch_state(accuracy, loss)

    def _mini_rms_prop(self, dataset:List):
        accuracies:List = []
        losses:List = []

        self._rms_init_velocity()
        batch = self._prepare_batch(dataset)
        for b in range(len(batch)):
            accuracy, loss = self._back_propagation(batch[b], self.weights, self.biaises)
            accuracies.append(accuracy)
            losses.append(loss)
            self._rms_update_weights(self.config.batch_size)
        return self._create_epoch_state(
            sum(accuracies)/len(accuracies),
            sum(losses)/len(losses)
        )

    def _stochatic_rms_prop(self, dataset:List):
        accuracies:List = []
        losses:List = []

        self._rms_init_velocity()
        for d in dataset:
            accuracy, loss = self._back_propagation([d], self.weights, self.biaises)
            accuracies.append(accuracy)
            losses.append(loss)
            self._rms_update_weights(1)

    #
    # --- 3.2.X ROOT MEAN SQUARE PROPAGATION UTILS ---
    #

    def _rms_init_velocity(self):
        self._velocity_w = [numpy.full((len(w),len(w[0])) , 0.0) for w in self.weights]
        self._velocity_b = [numpy.full(len(w), 0.) for w in self.weights]

    def _rms_update_weights(self, batch_size:int):
        for i in range(len(self.config.shape) - 1):
            self._velocity_w[i] = self.config.velocity_rate * self._velocity_w[i] + (1 - self.config.velocity_rate)*(numpy.power(self._nabla_w[i]/batch_size, 2))
            self.weights[i] -= (self.config.learning_rate / (numpy.sqrt(self._velocity_w[i])+EPS)) * (self._nabla_w[i] / batch_size)

            self._velocity_b[i] = self.config.velocity_rate * self._velocity_b[i] + (1 - self.config.velocity_rate)*(numpy.power(self._nabla_b[i]/batch_size, 2))
            self.biaises[i] -= (self.config.learning_rate / (numpy.sqrt(self._velocity_b[i]) + EPS)) * (self._nabla_b[i] / batch_size)


    # ------------------------------------------------------
    # --- 3.3 ADAM METHODS ---
    # ------------------------------------------------------
    def _full_adam(self, dataset:List):
        self._nag_init_momentum()
        self._rms_init_velocity()
        self._nag_init_ahead()
        self._nag_update_ahead()
        accuracy, loss = self._back_propagation(dataset, self._ahead_w, self._ahead_b)
        self._adam_update_weights(len(dataset))
        return self._create_epoch_state(accuracy, loss)

    def _mini_adam(self, dataset:List):
        accuracies:List = []
        losses:List = []

        self._nag_init_momentum()
        self._rms_init_velocity()
        self._nag_init_ahead()
        batch = self._prepare_batch(dataset)
        for b in range(len(batch)):
            self._nag_update_ahead()
            accuracy, loss = self._back_propagation(batch[b], self._ahead_w, self._ahead_b)
            accuracies.append(accuracy)
            losses.append(loss)
            self._adam_update_weights(self.config.batch_size)
        return self._create_epoch_state(
            sum(accuracies)/len(accuracies),
            sum(losses)/len(losses)
        )

    def _stochatic_adam(self, dataset:List):
        accuracies:List = []
        losses:List = []

        self._nag_init_momentum()
        self._rms_init_velocity()
        self._nag_init_ahead()
        for d in dataset:
            self._nag_update_ahead()
            accuracy, loss = self._back_propagation([d], self._ahead_w, self._ahead_b)
            accuracies.append(accuracy)
            losses.append(loss)
            self._adam_update_weights(1)
        return self._create_epoch_state(
            sum(accuracies)/len(accuracies),
            sum(losses)/len(losses)
        )

    #
    # --- 3.2.X ADAM UTILS ---
    #
    def _adam_update_weights(self, batch_size:int):
        for i in range(len(self.config.shape) - 1):
            self._momentum_w[i] = self.config.momentum_rate * self._momentum_w[i] + (1 - self.config.momentum_rate) * (self._nabla_w[i]/batch_size)
            self._velocity_w[i] = self.config.velocity_rate * self._velocity_w[i] + (1 - self.config.velocity_rate) * (numpy.power(self._nabla_w[i]/batch_size, 2))
            self.weights[i] -= self._momentum_w[i]/(numpy.sqrt(self._velocity_w[i] + EPS)) * self.config.learning_rate

            self._momentum_b[i] = self.config.momentum_rate * self._momentum_b[i] + (1 - self.config.momentum_rate) * (self._nabla_b[i]/batch_size)
            self._velocity_b[i] = self.config.velocity_rate * self._velocity_b[i] + (1 - self.config.velocity_rate) * (numpy.power(self._nabla_b[i]/batch_size, 2))
            self.biaises[i] -= self._momentum_b[i]/(numpy.sqrt(self._velocity_b[i] + EPS)) * self.config.learning_rate


    # ------------------------------------------------------
    # --- 3.X TRAINING UTILS ---
    # ------------------------------------------------------
    
    def _back_propagation(self, dataset:List, weights:List, biaises:List) -> tuple[float, float]:
        accuracies:List = []
        losses:List = []
        
        self._nabla_w = [numpy.full((len(w),len(w[0])) , 0.) for w in self.weights]
        self._nabla_b = [numpy.full(len(w), 0.) for w in self.weights]
        for d in dataset:
            dn_w = []
            dn_b = []
            if len(d["label"]) != self.config.shape[-1]:
                logger.info("The label need to have the same size than output layer")
                raise Exception()
            out = self._forward_pass(d["data"], weights, biaises)
            losses.append(self._loss_fnc(out[-1], d["label"]))
            accuracies.append(1 if step(out[-1], 0.5) == d["label"] else 0)
            delta = self._output_act_obj.delta(out[-1], d["label"])
            dn_w.insert(0, numpy.outer(numpy.array(delta), numpy.array(out[-2])))
            dn_b.insert(0, delta)
            idx = len(self.config.shape)-3
            while idx >= 0:
                prime = self._act_obj.prime(out[idx+1])
                delta = numpy.dot(numpy.transpose(weights[idx+1]), delta) * prime
                dn_w.insert(0, numpy.outer(numpy.array(delta), numpy.array(out[idx])))
                dn_b.insert(0, delta)
                idx-=1
            self._update_nabla(dn_w, dn_b)
        
        return sum(accuracies)/len(accuracies), sum(losses)/len(losses)

    def _prepare_batch(self, dataset:List):
        if len(dataset) < self.config.batch_size * 2:
            logger.error("data set to small to be used with this batch size")
            raise Exception()
        ds_len = int(len(dataset) / self.config.batch_size) * self.config.batch_size
        numpy.random.shuffle(dataset)
        batch = [[dataset[i] for i in range(j, j+self.config.batch_size)] for j in range(0, ds_len, self.config.batch_size)]
        return batch

    def _forward_pass(self, input:List, weights:List, biaises:List):
        out = []
        out.append(input)
        for l in range(len(self.config.shape) - 2):
            out.append(self._act_obj.activation(self._fire_layer(weights[l], biaises[l], out[-1])))
        out.append(self._output_act_obj.activation(self._fire_layer(weights[-1], biaises[-1], out[-1])))
        return out
    
    def _fire_layer(self, weight:List, biaises:List, input:List):
        res = [numpy.dot(w, input) + b for w,b in zip(weight, biaises)]
        return res

    def _update_nabla(self, dn_w:Sequence, dn_b:Sequence):
        for i in range(len(self.config.shape) - 1):
            self._nabla_w[i] += numpy.array(dn_w[i], float)
            self._nabla_b[i] += numpy.array(dn_b[i], float)


    # ======================================================
    # --- 7. UTILS ---
    # ======================================================
    def _checkNetwork(self):
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
        print(self.weights)
        print(" -biaises:")
        print(self.biaises)
        return
    
    def fire(self, input:numpy.array) -> numpy.array:
        act_input = input
        for l in range(len(self.config.shape) - 2):
            act_input = numpy.array(self._act_obj.activation(self._fire_layer(self.weights[l], self.biaises[l], act_input)))
        act_input = numpy.array(self._output_act_obj.activation(self._fire_layer(self.weights[-1], self.biaises[-1], act_input)))
        return act_input
    
    def _save_training(
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
            batch_size=self.config.batch_size,
            min_training_loss=min_training_loss,
            min_testing_loss=min_testing_loss,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1=f1,
            time=time
        )
    
    def _print_epoch_state(
            self,
            epoch:int,
            training_accuracy:float,
            training_loss:float,
            testing_accuracy:float,
            testing_loss:float
    ):
        message = (
            "===============================\n"
            f"At epoch {epoch}/{self.config.epoch}\n"
            "=== Training ===\n"
            f"Accuracy: {training_accuracy},\n"
            f"Loss: {training_loss},\n"
            "=== Testing ===\n"
            f"Accuracy: {testing_accuracy},\n"
            f"Loss: {testing_loss},\n"
            "==============================\n"
        )
        logger.info(message)

    def _create_epoch_state(self, accuracy:float, loss:float) -> Dict:
        return {
            "accuracy": accuracy,
            "loss": loss
        }
        
    def save_model(self, path_to_file: str):
        raw_data: Dict = {
            "shape": self.config.shape,
            "activation": self.config.activation_name,
            "output activation": self.config.output_activation_name,
            "weight": self.weights,
            "biaises": self.biaises
        }    
    