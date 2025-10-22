
import numpy as np

from typing import List, Dict
from core.network import Network
from utils.loading import progress_bar

from utils.logger import Logger
logger = Logger()

def gradient_descent(network:Network, dataset:List):
    for d in dataset:
        network.backpropagation(d["data"], d["label"])
    network.update_weights(len(dataset), network.learning_rate)
    network.update_biaises(len(dataset), network.learning_rate)
    return

def stochastic_gradient_descent(network:Network, dataset:List):
    if len(dataset) < network.batch_size * 2:
        logger.error("data set to small to be used with this batch size")
        raise Exception()
    ds_len = int(len(dataset) / network.batch_size) * network.batch_size
    np.random.shuffle(dataset)
    batch = [[dataset[i] for i in range(j, j+network.batch_size)] for j in range(0, ds_len, network.batch_size)]
    for b in range(len(batch)):
        for l in range(network.batch_size):
            network.backpropagation(batch[b][l]["data"], batch[b][l]["label"])
        network.update_weights(network.batch_size, network.learning_rate)
        network.update_biaises(network.batch_size, network.learning_rate)

def nesterov_momentum(dataset:List, config:Dict):
    pass

def rms_prop(dataset:List, config:Dict):
    pass

def adam(dataset:List, config:Dict):
    pass

