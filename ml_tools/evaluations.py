
import numpy as np

from core.network import Network
from typing import Dict
from ml_tools.utils import step

def classification(net:Network, ds_test:np.array) -> Dict:
    errors = []
    accuracy = 0
    for d in ds_test:
        output = net.fire(d["data"])
        errors.append(output-d["label"])
        result = step(output, 0.5)
        if result == d["label"]:
            accuracy += 1
    accuracy = accuracy *100 /len(ds_test)
    return {"accuracy": accuracy, "error_mean": np.mean(errors)}


