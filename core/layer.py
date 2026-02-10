from dataclasses import dataclass
from typing import Tuple

@dataclass
class Layer:

    shape: Tuple[int, int]
    activation: str

    weights_initiatilizer: str = "he_normal"

