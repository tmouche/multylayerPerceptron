from core.layer import Layer
from core.network import Network
from ml_tools.evaluations import binary_classification
from ml_tools.optimizers import Optimizer
from multipledispatch import dispatch
from time import perf_counter
from typing import Callable, Dict, List, Tuple
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


    @dispatch(List[Layer])
    @staticmethod
    def create_network(layers: List[Layer]) -> Network:
        
        try:
            new_net: Network = Network(layers=layers)
        except NetworkException as netErr:
            logger.error(netErr)
            raise ModelNetCreate(netErr)

        return new_net


    @dispatch(str)
    @staticmethod
    def create_network(path_to_init: str) -> Network:
        pass
    

    def __init__(self):
        
        self.accuracies = list()
        self.losses = list()

        
    @staticmethod
    def fit(
        network: Network,
        ds_train: List[Dict[str, ArrayF]],
        ds_test: List[Dict[str, ArrayF]],
        loss: str,
        learning_rate: FloatT,
        epochs: int,
        optimizer,
        batch_size: int = 1,
        early_stoper: FloatT = 0.,
        print_training_state: bool = True,
        history_save: bool = False,
    ) -> Tuple[Dict[str, List[FloatT]], Dict[str, List[FloatT]]]:
        accuracies: Dict[str, List[FloatT]] = dict(testing=list(), training=list())
        losses: Dict[str, List[FloatT]] = dict(testing=list(), training=list())

        try:
            optimizer_fnc: Callable = Model.get_optimizer(optimizer, network)
  
            loss_fnc: Callable = Model.get_loss(loss)
        except ModelException as modErr:
            raise ModelException(modErr)

        Model.load_layers(network, loss)
        
        network.learning_rate = learning_rate
        network.batch_size = batch_size

        logger.info("Starting training...")
        start_time: FloatT = perf_counter()
        for i_epoch in range(epochs):
            try:
                training: Dict[str, FloatT] = optimizer_fnc(ds_train)
                testing: Dict[str, FloatT] = binary_classification(network, loss_fnc, ds_test, )

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
                activation_function=network.layers[-1].output_activation,
                loss_function=loss,
                epoch=epochs,
                learning_rate=learning_rate,
                network_shape=[network.layers[i].shape for i in range(len(network.layers))],
                batch_size=batch_size,
                min_training_loss=min(losses.get('training')),
                min_testing_loss=min(losses.get('testing')),
                accuracy=testing.get('accuracy'),
                precision=testing.get('precision'),
                recall=testing.get('recall'),
                f1=testing.get('f1'),
                time=time_stamp
            )
        return accuracies, losses


    @staticmethod
    def get_optimizer(func_name: str, network: Network) -> Callable:
        try:
            return getattr(network, func_name)
        except AttributeError as attrErr:
            logger.error(f"Optimizer {func_name} not found")
            raise ModelOptimizer(attrErr)
        except Exception as e:
            logger.error(f"Unexpected exception: {e}")
            raise UnexpectedException()
        
    @staticmethod
    def get_loss(func_name: str) -> Callable:
        try:
            return getattr(Loss, func_name)
        except AttributeError as attrErr:
            logger.error(f"Loss {func_name} not found")
            raise ModelLoss(attrErr)
        except Exception as e:
            logger.error(f"Unexpected exception: {e}")
            raise UnexpectedException()

    @staticmethod
    def load_layer(network: Network, loss: str):
        
        for layer in network.layers:
            try:
                    layer.activation = getattr(Activation, layer.output_activation)(loss)
            except AttributeError:
                logger.error(f"Activation {layer.output_activation} not found")
                raise ModelActivation(layer.output_activation)
            except Exception as e:
                logger.error(f"Unexpected exception: {e}")
                raise UnexpectedException()