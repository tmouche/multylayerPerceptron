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

    in_use_weights: List[List[ArrayF]]
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
        return self._metric()

    def deterministic(self, dataset: List[Dict[str, ArrayF]]) -> Dict:
        self._reset()
        new_ds_len: int = floor(len(dataset) / self.net.batch_size) * self.net.batch_size
        batch: List[List[Dict[str, ArrayF]]] = [[dataset[i] for i in range(j, j + self.net.batch_size)] for j in range(0, new_ds_len, self.net.batch_size)]
        self._learn(batch, self.net.batch_size)
        return self._metric()

    def stochastic(self, dataset: List[Dict[str, ArrayF]]) -> Dict:
        self._reset()
        random_dataset: List[Dict[str, ArrayF]] = dataset.copy() 
        np.random.shuffle(random_dataset)
        batch: List[List[Dict[str, ArrayF]]] = self.prepare_batch(random_dataset)
        self._learn(batch, self.net.batch_size)
        return self._metric()

    def _learn(self, batch: List[List[Dict[str, ArrayF]]], batch_size: int):
        for i in range(len(batch)):
            self.fire.backward(list(batch[i]) if self.net.batch_size == 1 else batch[i], self.in_use_weights, self.in_use_biaises)
            self._update(batch_size)
            self.fire._reset()
        self.accuracies.append(np.mean(self.fire.accuracies[-len(batch):], dtype=FloatT))
        self.losses.append(np.mean(self.fire.losses[-len(batch):], dtype=FloatT))

    
    def _metric(self) -> Dict[str, List[FloatT]]:
        return dict(accuracy=self.accuracies[-1], loss=self.losses[-1])

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
        for i in range(len(self.net.weights)):
            self.net.weights[i] -= (self.net.learning_rate * (self.fire.nabla_w[i] / batch_size))
            self.net.biaises[i] -= (self.net.learning_rate * (self.fire.nabla_b[i] / batch_size))

    def _reset(self):
        pass


class RMS_Propagation(Optimizer):

    velocity_rate: FloatT

    velocity_w: List[List[ArrayF]]
    velocity_b: List[ArrayF]

    __r_velocity_w: List[List[ArrayF]]
    __r_velocity_b: List[ArrayF]

    def __init__(
            self,
            fire: Fire,
            network: Network,
            velocity_rate: FloatT
        ):
            super().__init__(fire=fire, network=network)
            self.velocity_rate = velocity_rate
            self.__r_velocity_w = list(np.full((len(w),len(w[0])) , 0., dtype=FloatT) for w in self.net.weights)
            self.__r_velocity_b = list(np.full(len(w), 0., dtype=FloatT) for w in self.net.weights)
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

    momentum_w: List[List[ArrayF]]
    momentum_b: List[ArrayF]

    ahead_w: List[List[ArrayF]]
    ahead_b: List[ArrayF]

    __r_momentum_w: List[List[ArrayF]]
    __r_momentum_b: List[ArrayF]


    def __init__(
            self,
            fire: Fire,
            network: Network,
            momentum_rate: FloatT
    ):
        super().__init__(fire=fire, network=network)
        self.momentum_rate = momentum_rate
        self.__r_momentum_w = list(np.full((len(w),len(w[0])) , 0., dtype=FloatT) for w in self.net.weights)
        self.__r_momentum_b = list(np.full(len(w), 0., dtype=FloatT) for w in self.net.weights)
        self._reset()
        self.ahead_w = self.net.weights.copy()
        self.ahead_b = self.net.biaises.copy()
        self.in_use_weights = self.ahead_w
        self.in_use_biaises = self.ahead_b

    def _reset(self):
        self.momentum_w = self.__r_momentum_w.copy()
        self.momentum_b = self.__r_momentum_b.copy()

    def _update(self, batch_size: int):
        for i in range(len(self.net.weights)):
            self.momentum_w[i] = self.momentum_rate * self.momentum_w[i] + (self.fire.nabla_w[i] / batch_size)
            self.net.weights[i] = self.net.weights[i] - (self.net.learning_rate * self.momentum_w[i])
            self.momentum_b[i] = self.momentum_rate * self.momentum_b[i] + (self.fire.nabla_b[i] / batch_size) 
            self.net.biaises[i] = self.net.biaises[i] - (self.net.learning_rate * self.momentum_b[i])
            self.ahead_w[i] = self.net.weights[i] - (self.momentum_rate * self.net.learning_rate * self.momentum_w[i])
            self.ahead_b[i] = self.net.biaises[i] - (self.momentum_rate * self.net.learning_rate * self.momentum_b[i])
