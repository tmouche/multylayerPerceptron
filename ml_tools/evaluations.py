
import numpy as np

from core.network import Network
from typing import (
    Callable, 
    Dict,
    List,
    Sequence
)
    
from ml_tools.utils import step

from ml_tools.utils import (
    accuracy,
    precision,
    recall,
    f1
)

from utils.logger import Logger

logger = Logger()

POSITIV = [1, 0]

def classification(
    net:Network,
    loss_fct:Callable[
        [Sequence[float] | float, Sequence[float] | float],
        Sequence[float] | float
    ],
    ds_test:Sequence[float]
) -> Dict:
    output:List = []
    labels:List = []
    losses:List = []
    tp = tn = fp = fn = 0
    for d in ds_test:
        output.append(net._fire(d["data"]))
        labels.append(d["label"])
        if step(output[-1], 0.5) == POSITIV:
            if labels[-1] == POSITIV:
                tp += 1 
            else:
                fp += 1
        else:
            if labels[-1] == POSITIV:
                fn += 1
            else:
                tn += 1
        losses.append(loss_fct(output[-1], labels[-1]))
    value_accuracy = accuracy(tp, tn, fp, fn)
    value_precision = precision(tp, fp)
    value_recall = recall(tp, fn)
    value_f1 = f1(value_precision, value_recall)
    average_loss = sum(losses) / len(losses)
    return {
        "accuracy": value_accuracy,
        "loss": average_loss,
        "precision": value_precision,
        "recall": value_recall,
        "f1": value_f1
    }


