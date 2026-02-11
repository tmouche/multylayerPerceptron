

UNEXPECTED = "Unexpected Exception catched"
FORMAT = "Parsing issue with format: "

NETWORK = "[NETWORK] "
NETWORK_INIT = "Initialization failed: "
NETWORK_LAYER_COUNT = "Count of layer has to be greater than 2"

LAYER = "[LAYER] "
LAYER_INIT = "Layer initialization failed: "
LAYER_SHAPE = "Shape less or egale to zero"
LAYER_ACTIVATION = "Layer activation function: "
LAYER_INITIALIZER = "Layer initializer function: "

class CriticalException(Exception):

    message: str

    def __init__(self, context:str):
        self.message = context

class FunctionalException(Exception):

    message: str

    def __init__(self, context: str):
        self.message = context

#-------------------------------------------#
#           --CRITICAL EXCEPTION--          #
#-------------------------------------------#

#           --NETWORK EXCEPTION--           #
class NetworkException(CriticalException):
    def __init__(self, context: str):
        super().__init__(NETWORK + context)

class NetworkInit(NetworkException):
    def __init__(self, context: str):
        super().__init__(NETWORK_INIT + context)

class NetworkLayerCount(NetworkInit):
    def __init__(self):
        super().__init__(NETWORK_LAYER_COUNT)

#           --LAYER EXCEPTION--             #
class LayerException(CriticalException):
    def __init__(self, context: str):
        super().__init__(LAYER + context)

class LayerInit(LayerException):
    def __init__(self, context: str):
        super().__init__(LAYER_INIT + context)

class LayerActivation(LayerInit):
    def __init__(self, context: str):
        super().__init__(LAYER_ACTIVATION + context)

class LayerInitializer(LayerInit):
    def __init__(self, context: str):
        super().__init__(LAYER_INITIALIZER + context)

class LayerShape(LayerInit):
    def __init__(self):
        super().__init__(LAYER_SHAPE)


#           --UTILS EXCEPTION--             #
class UnexpectedException(CriticalException):
    def __init__(self):
        super().__init__(UNEXPECTED)

class Format(CriticalException):
    def __init__(self, context: str):
        super().__init__(FORMAT + context)




#-------------------------------------------#
#         --FUNCTIONNAL EXCEPTION--         #
#-------------------------------------------#

