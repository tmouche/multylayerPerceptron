
import numpy as np

class logger:

    created = False
    _name = None
    me = None

    @staticmethod
    def singleton():
        if logger.created == False:
            logger.me = logger()
            logger.me._name = np.random.normal(loc=0,scale=1,size=(1))
            logger.created = True
        return logger.me

    async def hello(self, msg: str):
        print(f"Hello I'm {self._name} from {msg}")

    
