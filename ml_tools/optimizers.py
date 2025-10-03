
import time
import numpy as np

from typing import List, Dict
from core.network import Network
from utils.loading import progress_bar

from utils.logger import Logger
logger = Logger()

def gradient_descent(network:Network, dataset:List):
    total = len(dataset)
    progress = 0
    start_time = time.perf_counter()
    for d in dataset:
        network.backpropagation(d["data"], d["label"])
        # if network.option_visu_training:
        #     progress += 1
        #     progress_bar(progress, total, start_time)
    network.update_weights(len(dataset), network.learning_rate)
    network.update_biaises(len(dataset), network.learning_rate)
    return

def stochastic_gradient_descent(dataset:List, config:Dict):
    # try:
    #     batch_size = config["batch_size"]
    # except KeyError:
    #     raise Exception(f"Error log: Gradient Descent ")
    
    # def train(self, ds_train: np.array, ds_test: np.array):

    #     mean_loss_epochs = []
    #     accuracy_epochs = []
    #     if len(ds_train) < self.__batch_size * 2:
    #         raise Exception(f"Error log: data set to small to be used with this batch size")
    #     ds_len = int(len(ds_train) / self.__batch_size) * self.__batch_size
    #     # il faut check la batchsize en amont
    #     for e in range(self.__epoch):
    #         np.random.shuffle(ds_train)
    #         batch = [[ds_train[i] for i in range(j, j+self.__batch_size)] for j in range(0, ds_len, self.__batch_size)]
    #         for b in range(len(batch)):
    #             for l in range(self.__batch_size):
    #                 self.backpropagation(batch[b][l][1], batch[b][l][0])
    pass

def nesterov_momentum(dataset:List, config:Dict):
    pass

def rms_prop(dataset:List, config:Dict):
    pass

def adam(dataset:List, config:Dict):
    pass

