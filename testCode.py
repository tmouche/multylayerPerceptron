from enum import Enum
import numpy
import sys


class test:
    
    def __init__(self, ptitMessage:str):
        self.monMessage = ptitMessage

    def testprint(self):
        print("hello", self.monMessage)

myClass = getattr(sys.modules[__name__], "test")("grrrr")

myClass.testprint()

