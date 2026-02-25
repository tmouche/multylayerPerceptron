from core.layer import Layer
from ml_tools.activations import Activation
from ml_tools.utils import step
from utils.logger import Logger
from utils.types import ArrayF, FloatT
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
import numpy.typing as npt

logger = Logger()

class Fire:

    accuracies: ArrayF
    losses: ArrayF
    
    layers: List[Layer]

    nabla_w: List[List[ArrayF]]
    nabla_b: List[ArrayF]

    __r_nabla_w: List[List[ArrayF]]
    __r_nabla_b: List[ArrayF]

    def __init__(
            self,
            layers: List[Layer]
        ):
        self.accuracies = np.ndarray(0)
        self.losses = np.ndarray(0)

        self.layers = layers
        
        self.__r_nabla_w = list()
        self.__r_nabla_b = list()

        for i in range(1, len(self.layers)):
            previous_size: int = self.layers[i-1].shape
            size: int = self.layers[i].shape
            self.__r_nabla_w.append(list(np.full((size,previous_size) , 0., dtype=FloatT)))
            self.__r_nabla_b.append(list(np.full(size, 0., dtype=FloatT)))

        self._reset()


    def backward(
            self,
            dataset: List[Dict[str, ArrayF]],
            weights: List[List[ArrayF]],
            biaises: List[ArrayF]
        ):
        ds_size: int = len(dataset)
        e_accuracies: ArrayF = np.ndarray(ds_size)
        e_losses: ArrayF = np.ndarray(ds_size)

        for i in range(ds_size):
            out: List[ArrayF] = self.forward(dataset[i].get("data"), weights, biaises)
            e_losses[i] = self.layers[-1].activation.loss(out[-1], dataset[i].get("label"))
            e_accuracies[i] = 1 if step(out[-1], 0.5) == dataset[i].get("label").tolist() else 0
            delta: ArrayF = self.layers[-1].activation.delta(out[-1], dataset[i].get("label"))
            self.nabla_w[-1] += np.outer(delta, out[-2])
            self.nabla_b[-1] += delta
            idx: int = len(weights)-2
            while idx >= 0:
                prime: ArrayF = self.layers[idx+1].activation.prime(out[idx+1])
                delta: ArrayF = np.dot(np.transpose(weights[idx+1]), delta) * prime
                self.nabla_w[idx] += np.outer(np.array(delta), np.array(out[idx]))
                self.nabla_b[idx] += delta
                idx-=1

        self.accuracies = np.append(self.accuracies, np.mean(e_accuracies))
        self.losses = np.append(self.losses, np.mean(e_losses))

    

    def forward(
            self,
            input: ArrayF,
            weights: List[List[ArrayF]],
            biaises: List[ArrayF]
        ) -> List[ArrayF]:
        out: List[ArrayF] = [input]
        for i in range(len(weights) - 1):
            out.append(self.layers[i + 1].activation.activation(self.layer(out[-1], weights[i], biaises[i])))
        out.append(self.layers[-1].activation.activation(self.layer(out[-1], weights[-1], biaises[-1])))
        return out


    def layer(
            self,
            input: ArrayF,
            weights: List[ArrayF],
            biaises: ArrayF
        ) -> ArrayF:
        res: ArrayF = np.ndarray(0)
        for w, b in zip(weights, biaises):
            res = np.append(res, np.dot(w, input) + b)
        return res


    def full(
            self,
            input: ArrayF,
            weights: List[List[ArrayF]],
            biaises: List[ArrayF]
        ) -> ArrayF:
        return self.forward(input, weights, biaises)[-1]
    
    def _reset(self):
        self.nabla_w = self.__r_nabla_w.copy()
        self.nabla_b = self.__r_nabla_b.copy()