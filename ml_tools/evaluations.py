
import numpy as np

from core.network import Network
from typing import (
    Callable, 
    Dict,
    Sequence
)
    
from ml_tools.utils import step


from utils.logger import Logger

logger = Logger()

def classification(
    net:Network,
    loss_fct:Callable[[Sequence[float], Sequence[float]], Sequence[float] | float],
    ds_test:Sequence[float]
) -> Dict:
    output = []
    labels = []
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for d in ds_test:
        label = d["label"]
        output.append(np.clip(net.fire(d["data"]), 1e-8, 1-1e-8))
        labels.append(d["label"])

        if step(output[-1], 0.5) == d["label"]:

            accuracy += 1
    accuracy = accuracy *100 /len(ds_test)
    output = np.array(output)
    labels = np.array(labels)
    return {"accuracy": accuracy, "loss": loss_fct(output, labels)[0]} #check mais c est pas normal imo


