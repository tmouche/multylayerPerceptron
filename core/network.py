from core.layer import Layer
from ml_tools.fire import Fire
from typing import (
    Dict,
    List,
)
from utils.exception import (
    Format,
    LayerInit,
    LayerInitializer,
    NetworkBatchSize,
    NetworkInit,
    NetworkLayerCount,
    NetworkLearningRate,
    UnexpectedException
)
from utils.logger import Logger
from utils.types import ArrayF, FloatT

logger = Logger()

class Network:

    layers: List[Layer]

    fire: Fire

    batch_size: int

    learning_rate: FloatT

    weights: List[List[ArrayF]]
    biaises: List[ArrayF]

    def __init__(
            self,
            layers: List[Layer],
            learning_rate: FloatT,
            batch_size: int
        ):
        
        if learning_rate is None or learning_rate <= 0.:
            logger.error(f"Learning rate: {learning_rate}")
            raise NetworkLearningRate()
        self.learning_rate = learning_rate

        if batch_size is None or batch_size <= 0.:
            logger.error(f"Batch size: {batch_size}")
            raise NetworkBatchSize()
        self.batch_size = batch_size

        self.layers = list()
        for l in layers:
            if isinstance(l, Layer) == False:
                logger.error(f"Unrecognized argument passed: {l}")
                raise Format(context="Layer")
            self.layers.append(l)

        try:
            self.__init_layers()
        except (LayerInit, NetworkInit) as iniErr:
            logger.error(iniErr)
            raise NetworkInit(str(iniErr))
        except Exception as e:
            logger.error(f"Unexpected exception: {e}")
            raise UnexpectedException()

        self.fire = Fire(self.layers)
            
        
    def __init_layers(self):
        
        self.weights = list()
        self.biaises = list()

        if len(self.layers) < 3:
            logger.error(f"Not enough layers")
            raise NetworkLayerCount()
        for i in range(1, len(self.layers)):
            previous_size: int = self.layers[i-1].shape
            size: int = self.layers[i].shape

            if not self.layers[i].initializer:
                logger.error(f"Missing Initializer in layer n*{i+1}")
                raise LayerInitializer(context="Missing")
            self.weights.append(self.layers[i].initializer(shape=(size, previous_size)))
            self.biaises.append(self.layers[i].initializer(shape=(size)))

    
    def save_model(self, path_to_file: str):
        raw_data: Dict = {
            "shape": self.config.shape,
            "activation": self.config.activation_name,
            "output activation": self.config.output_activation_name,
            "weight": self.weights,
            "biaises": self.biaises
        }    
    