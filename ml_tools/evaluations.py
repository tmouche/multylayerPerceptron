
import numpy as np

from core.network import Network
from typing import Dict
from ml_tools.utils import step

from utils.logger import Logger

logger = Logger()

def classification(loss_fct, net:Network, ds_test:np.array) -> Dict:
    output = []
    labels = []
    accuracy = 0
    for d in ds_test:
        output.append(np.clip(net.fire(d["data"]), 1e-8, 1-1e-8))
        labels.append(d["label"])
        if step(output[-1], 0.5) == d["label"]:
            accuracy += 1
    accuracy = accuracy *100 /len(ds_test)
    output = np.array(output)
    labels = np.array(labels)
    return {"accuracy": accuracy, "error_mean": loss_fct(output, labels)[0]} #check mais c est pas normal imo


