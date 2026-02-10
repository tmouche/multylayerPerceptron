

UNEXPECTED = "Unexpected Exception catched"
FORMAT = "Parsing issue with format: "

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

class UnexpectedException(CriticalException):
    def __init__(self):
        super().__init__(UNEXPECTED)

class Format(CriticalException):
    def __init__(self, context: str):
        super().__init__(FORMAT + context)

#-------------------------------------------#
#         --FUNCTIONNAL EXCEPTION--         #
#-------------------------------------------#

