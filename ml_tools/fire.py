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

    accuracies: List[FloatT]
    losses: List[FloatT]
    
    layers: List[Layer]

    w_shape: List[Tuple[int, int]]
    b_shape: List[int]

    nabla_w: List[npt.ArrayLike[ArrayF]]
    nabla_b: List[ArrayF]

    __r_nabla_w: List[npt.ArrayLike[ArrayF]]
    __r_nabla_b: List[ArrayF]

    def __init__(
            self,
            layers: List[Layer]
        ):
        self.accuracies = list()
        self.losses = list()

        self.layers = layers

        for i in range(1, len(self.layers)):
            previous_size: int = self.layers[i-1]
            size: int = self.layers[i]
            self.weights = np.append(self.weights, self.layers[i].initializer(shape=(size, previous_size)))
            self.biaises = np.append(self.biaises, self.layers[i].initializer(shape=(size)))

        self.__r_nabla_w = list(np.full((len(w),len(w[0])) , 0.) for w in self.weights)
        self.__r_nabla_b = list(np.full(len(w), 0.) for w in self.weights)
        self._reset()


    def backward(
            self,
            dataset: List[Dict[str, ArrayF]],
            weights: List[npt.ArrayLike[ArrayF]],
            biaises: List[ArrayF]
        ):
        e_accuracies: List[FloatT] = list()
        e_losses: List[FloatT] = list()

        for d in dataset:
            out: npt.ArrayLike[ArrayF] = self._forward_pass(d["data"], weights, biaises)
            e_losses.append(self.layers[-1].activation.loss(out[-1], d["label"]))
            e_accuracies.append(1 if step(out[-1], 0.5) == d["label"] else 0)
            delta: ArrayF = self.layers[-1].activation.delta(out[-1], d["label"])
            self.nabla_w[-1] += np.outer(delta, out[-2])
            self.nabla_b[-1] += delta
            idx: int = len(self.w_shape)-2
            while idx >= 0:
                prime: ArrayF = self.layers[idx].activation.prime(out[idx+1])
                delta: ArrayF = np.dot(np.transpose(weights[idx+1]), delta) * prime
                self.nabla_w[idx] += np.outer(np.array(delta), np.array(out[idx]))
                self.nabla_b += delta
                idx-=1
        
        self.accuracies(np.mean(e_accuracies))
        self.losses(np.mean(e_losses))
    

    def forward(
            self,
            input: ArrayF,
            weights: List[npt.ArrayLike[ArrayF]],
            biaises: List[ArrayF]
        ) -> npt.ArrayLike[ArrayF]:
        out: npt.ArrayLike[ArrayF] = np.ndarray(0)
        out = np.append(out, input)
        for i in range(len(weights) - 1):
            out = np.append(out, self.layers[i].activation.activation(self.layer(weights[i], biaises[i], out[-1])))
        out = np.append(out, self.layers[-1].activation.activation(self.layer(weights[-1], biaises[-1], out[-1])))
        return out


    def layer(
            self,
            input: ArrayF,
            weights: npt.ArrayLike[ArrayF],
            biaises: ArrayF
        ) -> ArrayF:
        res: ArrayF = np.ndarray()
        for w, b in zip(weights, biaises):
            res = res.append(res, np.dot(w, input) + b)
        return res


    def full(
            self,
            input: ArrayF,
            weights: List[npt.ArrayLike[ArrayF]],
            biaises: List[ArrayF]
        ) -> ArrayF:
        return self.forward(input, weights, biaises)[-1]
    
    def _reset(self):
        self.nabla_w = self.__r_nabla_w.copy()
        self.nabla_b = self.__r_nabla_b.copy()