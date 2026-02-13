from typing import List, Sequence, Tuple
from utils.types import ArrayF, FloatT


import math
import numpy as np
import numpy.typing as npt

SEED = 42

def random_normal(
        shape: Tuple[int, int] = (1,1),
        mean: FloatT = 0.0,
        stddev: FloatT = 0.05
    ) -> ArrayF | npt.ArrayLike[ArrayF]:
    if SEED: np.random.seed(SEED)
    return np.random.normal(loc=mean, scale=stddev, size=shape)
    
def random_uniform(
        shape: Tuple[int, int] = (1,1),
        minval: FloatT = 0.,
        maxval: FloatT = 1.
    ) -> ArrayF | npt.ArrayLike[ArrayF]:
    if SEED: np.random.seed(SEED)
    return np.random.uniform(low=minval, high=maxval, size=shape)
        
def zeros(shape: Tuple[int, int] = (1,1)) -> ArrayF | npt.ArrayLike[ArrayF]:
    if SEED: np.random.seed(SEED)
    return np.zeros(shape)

def ones(shape: Tuple[int, int] = (1,1)) -> ArrayF | npt.ArrayLike[ArrayF]:
    if SEED: np.random.seed(SEED)
    return np.ones(shape)
    
def xavier_normal(
        shape: Tuple[int, int] = (1,1),
        fan_in: int = 1,
        fan_out: int = 1
    ) -> ArrayF | npt.ArrayLike[ArrayF]:
    if SEED: np.random.seed(SEED)
    stddev = math.sqrt(2/(fan_in+fan_out))
    return np.random.normal(loc=0, scale=stddev, size=shape)

def xavier_uniform(
        shape: Tuple[int, int] = (1,1),
        fan_in: int = 1,
        fan_out: int = 1
    ) -> ArrayF | npt.ArrayLike[ArrayF]:
    if SEED: np.random.seed(SEED)
    limit = math.sqrt(6/(fan_in+fan_out))
    return np.random.uniform(low=-limit, high=limit, size=shape)
    
def he_normal(
        shape: Tuple[int, int] = (1,1),
        fan_in: int = 1
    ) -> ArrayF | npt.ArrayLike[ArrayF]:
    if SEED: np.random.seed(SEED)
    stddev = math.sqrt(2/fan_in)
    return np.random.normal(loc=0, scale=stddev, size=shape)

def he_uniform(
        shape: Tuple[int, int] = (1,1),
        fan_in: int = 1
    ) -> ArrayF | npt.ArrayLike[ArrayF]:
    if SEED: np.random.seed(SEED)
    limit = math.sqrt(6/fan_in)
    return np.random.uniform(low=-limit, high=limit, size=shape)