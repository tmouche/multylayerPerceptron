from ml_tools.utils import (
    step,
    accuracy,
    precision,
    recall,
    f1
)
from typing import (
    Callable, 
    Dict,
    List,
)
from utils.logger import Logger
from utils.types import ArrayF, FloatT

logger = Logger()

def binary_classification(
    network,
    loss_fnc: Callable,
    ds_test: List[Dict[str, ArrayF]],
    positiv: List[int]
) -> Dict[str, FloatT]:
    losses:List[FloatT] = list()
    tp = tn = fp = fn = 0

    for t in ds_test:
        output: ArrayF = network.fire.full(t.get("data"), network.weights, network.biaises)
        label: List[int] = list(t.get("label"))
        if step(output, 0.5) == positiv:
            if label == positiv:
                tp += 1 
            else:
                fp += 1
        else:
            if label == positiv:
                fn += 1
            else:
                tn += 1
        losses.append(loss_fnc(output, label))
    value_precision: FloatT = precision(tp, fp)
    value_recall: FloatT = recall(tp, fn)
    value_f1: FloatT = f1(value_precision, value_recall)
    average_loss: FloatT = sum(losses) / len(losses)
    return {
        "accuracy": accuracy(tp, tn, fp, fn),
        "loss": average_loss,
        "precision": value_precision,
        "recall": value_recall,
        "f1": value_f1
    }


