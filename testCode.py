from enum import Enum


class act(Enum):
    SIGMOID = "sigmoid"
    HENORMAL = "he_normal"


if "sigmoid" in act:
    print("yessssss")