from typing import Callable
from utils.exception import (
    LayerActivation,
    LayerException,
    LayerInit,
    LayerInitializer,
    LayerShape,
    UnexpectedException,
)
from utils.constant import ACTIVATION_RESTRICT_SHAPE
from utils.logger import Logger

import ml_tools.activations as Activation
import ml_tools.initialisations as Initializer

logger = Logger()

class Layer:

    shape: int

    output_activation: str
    activation: Callable

    weights_initiatilizer: str
    initializer: Callable

    def __init__(self, shape: int, activation: str | None = None, initializer: str | None = None):
        """
        Initialize a layer configuration.

        Args:
            shape (int):
                Number of units (neurons) in the layer. Must be greater than zero.

            activation (str | None, optional):
                Name of the activation function to use for the layer output.
                Must correspond to a callable defined in the `Activation` class.
                Defaults to None.

            initializer (str | None, optional):
                Name of the weights initializer to use for the layer.
                Must correspond to a callable defined in the `Initializer` class.
                Defaults to None.

        Raises:
            LayerInit:
                If shape, activation, or initializer validation fails. The original
                error message is propagated through the `context` field.

            UnexpectedException:
                If an unexpected error occurs during initialization.

        Logs:
            Error when an unexpected exception occurs.
        """
        try:
            self.shape = shape
            self.__init_shape()

            self.output_activation = activation
            self.__init__activation()

            self.weights_initiatilizer = initializer
            self.__init_initializer()
        except LayerInit as iniErr:
            raise LayerException(context=str(iniErr))
        except Exception as e:
            logger.error(f"Unexpected exception: {e}")
            raise UnexpectedException()
    

    def __init_shape(self):
        """
        Validate and initialize the layer shape.

        Raises:
            LayerShape: If `self.shape` is less than 1.

        Logs:
            Error if the provided shape is invalid.
        """
        if self.shape < 1:
            logger.error(f"Layer shape less than one: {self.shape}")
            raise LayerShape()
        

    def __init__activation(self):
        """
        Validate and initialize the activation function for the layer.

        Raises:
            LayerActivationShape: If the activation imposes a shape restriction
                                  that is not satisfied by `self.shape`.
            LayerActivation: If the specified activation name does not exist
                             in the `Activation` class.

        Logs:
            - Error if the activation shape restriction is violated.
            - Error if the activation name is unknown.

        If `self.output_activation` is None or empty, `self.activation` is set to None.
        """
        if self.output_activation:
            if restrict := ACTIVATION_RESTRICT_SHAPE.get(self.output_activation):
                if restrict < self.shape:
                    logger.error(f"Activation {self.output_activation} restrict violated with shape {self.shape}")
                    raise LayerActivation("Shape")
            try:
                self.activation = getattr(Activation, self.output_activation)
            except AttributeError:
                logger.error(f"Unknow activation: {self.output_activation}")
                raise LayerActivation("Unknow")
        else:
            self.activation = None


    def __init_initializer(self):
        """
        Initialize the weights initializer for the layer.

        Raises:
            LayerInit: If the specified initializer name does not exist in `Initializer`.
        Logs:
            - Error if the initializer name is unknown.

        If `self.weights_initiatilizer` is None or empty, `self._initializer` is set to None.
        """
        if self.weights_initiatilizer:
            try:
                self.initializer = getattr(Initializer, self.weights_initiatilizer)
            except AttributeError:
                logger.error(f"Unknow initializer: {self.weights_initiatilizer}")
                raise LayerInitializer("Unknow")
        else:
            self.initializer = None 
