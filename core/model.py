from core.layer import Layer
from core.network import Network
from ml_tools.evaluations import binary_classification
from ml_tools.fire import Fire
from time import perf_counter
from typing import Callable, Dict, List
from utils.constant import POSITIV
from utils.exception import (
    ModelActivation,
    ModelLoss,
    ModelNetCreate,
    NetworkException,
    UnexpectedException,
)
from utils.history import save_to_history
from utils.logger import Logger
from utils.types import ArrayF, FloatT
import ml_tools.activations as Activation
import ml_tools.losses as Loss
import numpy as np

logger = Logger()

class Model:

    accuracies: Dict[str, ArrayF]
    losses: Dict[str, ArrayF]

    fire: Fire
    network: Network

    def __init__(self):
        
        self.accuracies = dict(testing=np.ndarray(0), training=np.ndarray(0))
        self.losses = dict(testing=np.ndarray(0), training=np.ndarray(0))

        self.fire = None
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
        self.fire = Fire(self.network.layers) 


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
    ):

        self.load_layers(loss)
        
        logger.info("Starting training...")
        start_time: FloatT = perf_counter()
        max_epoch: int = 0
        for i_epoch in range(epochs):
            try:
                training: Dict[str, FloatT] = optimizer(ds_train)
                testing: Dict[str, FloatT] = binary_classification(self.network, self.network.layers[-1].activation.loss, ds_test, POSITIV)

                self.accuracies["testing"] = np.append(self.accuracies.get("testing"), testing.get("accuracy"))
                self.losses["testing"] = np.append(self.losses.get("testing"), testing.get("loss"))
                self.accuracies["training"] = np.append(self.accuracies.get("training"), training.get("accuracy"))
                self.losses["training"] = np.append(self.losses.get("training"), training.get("loss"))

                if print_training_state:
                    Model._print_epoch_state(
                        self,
                        act_epoch=i_epoch,
                        epoch=epochs,
                        training_accuracy=training.get("accuracy"),
                        training_loss=training.get("loss"),
                        testing_accuracy= testing.get("accuracy"),
                        testing_loss= testing.get("loss"),
                    )
                if testing.get("loss") < early_stoper:
                    max_epoch = i_epoch
                    break
            except KeyboardInterrupt:
                max_epoch = i_epoch
                logger.info("Trainning Stopped")
                break
            except Exception as e:
                logger.error(f"Unexpected exception: {e}")
                raise UnexpectedException()
        time_stamp: FloatT = perf_counter() - start_time
        if not max_epoch: max_epoch = epochs
        if history_save:
            save_to_history(
                optimizer=f"{optimizer.__self__.__class__.__qualname__}.{optimizer.__name__}",
                activation_function=self.network.layers[-1].output_activation,
                loss_function=loss,
                epoch=max_epoch,
                learning_rate=self.network.learning_rate,
                network_shape=[self.network.layers[i].shape for i in range(len(self.network.layers))],
                batch_size=self.network.batch_size,
                min_training_loss=min(self.losses.get('training')),
                min_testing_loss=min(self.losses.get('testing')),
                accuracy=testing.get('accuracy'),
                precision=testing.get('precision'),
                recall=testing.get('recall'),
                f1=testing.get('f1'),
                time=time_stamp
            )
        
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
            
    def _print_epoch_state(
            self,
            act_epoch:int,
            epoch:int,
            training_accuracy:float,
            training_loss:float,
            testing_accuracy:float,
            testing_loss:float,
    ):
        message = (
            "\n===============================\n"
            f"At epoch {act_epoch}/{epoch}\n"
            "=== Training ===\n"
            f"Accuracy: {training_accuracy},\n"
            f"Loss: {training_loss},\n"
            "=== Testing ===\n"
            f"Accuracy: {testing_accuracy},\n"
            f"Loss: {testing_loss},\n"
            "==============================\n"
        )
        logger.info(message)