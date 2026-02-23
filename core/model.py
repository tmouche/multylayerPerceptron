from core.layer import Layer
from core.network import Network
from ml_tools.evaluations import binary_classification
from ml_tools.optimizers import Optimizer
from multipledispatch import dispatch
from time import perf_counter
from typing import Callable, Dict, List, Tuple
from utils.constant import POSITIV
from utils.exception import (
    LayerActivation,
    ModelActivation,
    ModelException,
    ModelLoss,
    ModelNetCreate,
    ModelOptimizer,
    NetworkException,
    UnexpectedException,
)
from utils.history import save_to_history
from utils.logger import Logger
from utils.types import ArrayF, FloatT

import ml_tools.activations as Activation
import ml_tools.losses as Loss

logger = Logger()

class Model:

    accuracies: List[Dict[str, List[FloatT]]]
    losses: List[Dict[str, List[FloatT]]]

    network: Network

    def __init__(self):
        
        self.accuracies = list()
        self.losses = list()

        self.network = None


    def create_network(
        self,
        layers: List[Layer],
        learning_rate: FloatT,
        batch_size: int
    ):
        
        try:
            new_net: Network = Network(
                layers=layers,
                learning_rate=learning_rate,
                batch_size=batch_size
            )
        except NetworkException as netErr:
            logger.error(netErr)
            raise ModelNetCreate(netErr)
        self.network = new_net


    def load_network(self, path_to_init: str) -> Network:
        pass

        
    def fit(
        self,
        optimizer: Callable,
        ds_train: List[Dict[str, ArrayF]],
        ds_test: List[Dict[str, ArrayF]],
        loss: str,
        epochs: int,
        early_stoper: FloatT = 0.,
        print_training_state: bool = True,
        history_save: bool = False,
    ) -> Tuple[Dict[str, List[FloatT]], Dict[str, List[FloatT]]]:
        accuracies: Dict[str, List[FloatT]] = dict(testing=list(), training=list())
        losses: Dict[str, List[FloatT]] = dict(testing=list(), training=list())

        self.load_layers(loss)
        
        logger.info("Starting training...")
        start_time: FloatT = perf_counter()
        for i_epoch in range(epochs):
            try:
                training: Dict[str, FloatT] = optimizer(ds_train)
                testing: Dict[str, FloatT] = binary_classification(self.fire, self.network.layers[-1].activation.loss, ds_test, POSITIV)

                accuracies.get('testing').append(testing.get("accuracy"))
                accuracies.get('training').append(training.get("accuracy"))
                losses.get('testing').append(testing.get("loss"))
                losses.get('training').append(training.get("loss"))

                if print_training_state:
                    Model.print_epoch_state(
                        epoch=i_epoch,
                        training_accuracy=training.get("accuracy"),
                        training_loss=training.get("loss"),
                        testing_accuracy= testing.get("accuracy"),
                        testing_loss= testing.get("loss")
                    )
                if testing.get("loss") < early_stoper:
                    break
            except KeyboardInterrupt:
                logger.info("Trainning Stopped")
                break
            except Exception as e:
                logger.error(f"Unexpected exception: {e}")
                raise UnexpectedException()
        time_stamp: FloatT = perf_counter() - start_time
        if history_save:
            save_to_history(
                optimizer=optimizer,
                activation_function=self.network.layers[-1].output_activation,
                loss_function=loss,
                epoch=epochs,
                learning_rate=self.network.learning_rate,
                network_shape=[self.network.layers[i].shape for i in range(len(self.network.layers))],
                batch_size=self.network.batch_size,
                min_training_loss=min(losses.get('training')),
                min_testing_loss=min(losses.get('testing')),
                accuracy=testing.get('accuracy'),
                precision=testing.get('precision'),
                recall=testing.get('recall'),
                f1=testing.get('f1'),
                time=time_stamp
            )
        return accuracies, losses
        
    def get_loss(func_name: str) -> Callable:
        try:
            return getattr(Loss, func_name)
        except AttributeError as attrErr:
            logger.error(f"Loss {func_name} not found")
            raise ModelLoss(attrErr)
        except Exception as e:
            logger.error(f"Unexpected exception: {e}")
            raise UnexpectedException()

    def load_layers(self, loss: str):
        
        for i in range(1, len(self.network.layers)):
            try:
                self.network.layers[i].activation = getattr(Activation, self.network.layers[i].output_activation)(loss)
            except AttributeError:
                logger.error(f"Activation {self.network.layers[i].output_activation} not found")
                raise ModelActivation(self.network.layers[i].output_activation)
            except Exception as e:
                logger.error(f"Unexpected exception: {e}")
                raise UnexpectedException()