
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
        output.append(net.fire(d["data"]))
        labels.append(d["label"])
        if output[-1] == d["label"]:
            accuracy += 1
    accuracy = accuracy *100 /len(ds_test)
    return {"accuracy": accuracy, "error_mean": loss_fct(output, labels)[0]}


