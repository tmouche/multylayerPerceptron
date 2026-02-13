from abc import ABC, abstractmethod
from core.network import Network
from core.layer import Layer
from math import floor
from ml_tools.fire import Fire
from typing import Dict, List
from utils.logger import Logger
from utils.types import ArrayF, FloatT


import numpy as np
import numpy.typing as npt

logger = Logger()

EPS: FloatT = 1e-8

class Optimizer(ABC):

    in_use_weights: List[npt.ArrayLike[ArrayF]]
    in_use_biaises: List[ArrayF]

    fire: Fire
    net: Network

    accuracies: List[FloatT]
    losses: List[FloatT]

    def __init__(
            self,
            fire: Fire,
            network: Network
        ):
        self.accuracies = list()
        self.losses = list()

        self.fire = fire
        self.net = network
        self.in_use_weights = network.weights
        self.in_use_biaises = network.biaises


    def full(self, dataset: List[Dict[str, ArrayF]]):
        self._reset()
        self._learn([dataset], len(dataset))


    def deterministic(self, dataset: List[Dict[str, ArrayF]]) -> Dict:
        self._reset()
        new_ds_len: int = floor(len(dataset) / self.net.batch_size) * self.net.batch_size
        batch: List[List[Dict[str, ArrayF]]] = [[dataset[i] for i in range(j, j + self.net.batch_size)] for j in range(0, new_ds_len, self.net.batch_size)]
        self._learn(batch, self.net.batch_size)


    def stochastic(self, dataset: List[Dict[str, ArrayF]]) -> Dict:
        self._reset()
        random_dataset: List[Dict[str, ArrayF]] = dataset.copy() 
        np.random.shuffle(random_dataset)
        batch: List[List[Dict[str, ArrayF]]] = self.prepare_batch(random_dataset)
        self._learn(batch, self.net.batch_size)

    def _learn(self, batch: List[List[Dict[str, ArrayF]]], batch_size: int):
        for b in range(len(batch)):
            self.fire.backward(list(b) if self.net.batch_size == 1 else b, self.in_use_weights, self.in_use_biaises)
            self._update(batch_size)
            self.fire._reset()
    
    @abstractmethod
    def _update(self, batch_size: int):
        pass

    @abstractmethod
    def _reset(self):
        pass


class Gradient_Descent(Optimizer):

    def __init__(
            self,
            fire: Fire,
            network: Network
        ):
            super().__init__(fire=fire, network=network)

    def _update(self, batch_size: int):
        for i in range(len(self.net.weights) - 1):
            self.net.weights[i] -= (self.net.learning_rate * (self.fire.nabla_w[i] / batch_size))
            self.net.biaises[i] -= (self.net.learning_rate * (self.fire.nabla_b[i] / batch_size))

    def _reset(self):
        pass


class RMS_Propagation(Optimizer):

    velocity_rate: FloatT

    velocity_w: List[npt.ArrayLike[ArrayF]]
    velocity_b: List[ArrayF]

    __r_velocity_w: List[npt.ArrayLike[ArrayF]]
    __r_velocity_b: List[ArrayF]

    def __init__(
            self,
            fire: Fire,
            network: Network,
            velocity_rate: FloatT
        ):
            super().__init__(fire=fire, network=network)
            self.velocity_rate = velocity_rate
            self.__r_velocity_w = list(np.full((len(w),len(w[0])) , 0.) for w in self.net.weights)
            self.__r_velocity_b = list(np.full(len(w), 0.) for w in self.net.weights)
            self._reset()

    def _update(self, batch_size:int):
        for i in range(len(self.net.weights)):
            self.velocity_w[i] = self.velocity_rate * self.velocity_w[i] + (1 - self.velocity_rate)*(np.power(self.fire.nabla_w[i]/batch_size, 2))
            self.net.weights[i] -= (self.net.learning_rate / (np.sqrt(self.velocity_w[i]) + EPS)) * (self.fire.nabla_w[i] / batch_size)

            self.velocity_b[i] = self.velocity_rate * self.velocity_b[i] + (1 - self.velocity_rate)*(np.power(self.fire.nabla_b[i]/batch_size, 2))
            self.net.biaises[i] -= (self.net.learning_rate / (np.sqrt(self.velocity_b[i]) + EPS)) * (self.fire.nabla_b[i] / batch_size)

    def _reset(self):
        self.velocity_w = self.__r_velocity_w.copy()
        self.velocity_b = self.__r_velocity_b.copy()


class Nesterov_Accelerated_Gradient(Optimizer):

    momentum_rate: FloatT

    momentum_w: List[npt.ArrayLike[ArrayF]]
    momentum_b: List[ArrayF]

    ahead_w: List[npt.ArrayLike[ArrayF]]
    ahead_b: List[ArrayF]

    __r_momentum_w: List[npt.ArrayLike[ArrayF]]
    __r_momentum_b: List[ArrayF]


    def __init__(
            self,
            fire: Fire,
            network: Network,
            momentum_rate: FloatT
    ):
        super().__init__(fire=fire, network=network)
        self.momentum_rate = momentum_rate
        self.__r_momentum_w = list(np.full((len(w),len(w[0])) , 0.) for w in self.net.weights)
        self.__r_momentum_b = list(np.full(len(w), 0.) for w in self.net.weights)
        self._reset()
        self.ahead_w = self.net.weights.copy()
        self.ahead_b = self.net.biaises.copy()
        self.in_use_weights = self.ahead_w
        self.in_use_biaises = self.ahead_b

    def _reset(self):
        self.momentum_w = self.__r_momentum_w.copy()
        self.momentum_b = self.__r_momentum_b.copy()

    def __update(self, batch_size: int):
        for i in range(len(self.net.weights)):
            self.ahead_w[i] = self.net.weights[i] - (self.momentum_rate * self.net.learning_rate * self.momentum_w[i])
            self.ahead_b[i] = self.net.biaises[i] - (self.momentum_rate * self.net.learning_rate * self.momentum_b[i])
            self.momentum_w[i] = self.momentum_rate * self.momentum_w[i] + (self.fire.nabla_w[i] / batch_size)
            self.net.weights[i] = self.weights[i] - (self.config.learning_rate * self._momentum_w[i])
            self._momentum_b[i] = self.config.momentum_rate * self._momentum_b[i] + (self._nabla_b[i] / batch_size) 
            self.biaises[i] = numpy.array(self.biaises[i]) - (self.config.learning_rate * self._momentum_b[i])

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