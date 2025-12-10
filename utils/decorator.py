from functools import wraps

from utils.logger import Logger
logger = Logger()

def call_decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            if self._is_apply == False:
                logger.error("The network have to be apply before anything else")
                raise Exception()
            return func(self, *args, **kwargs)
        return wrapper