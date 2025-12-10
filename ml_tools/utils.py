
def step(z, center:float):
    if isinstance(z, float):
        return 1. if z > center else 0.
    return [1. if x > center else 0. for x in z]

def accuracy(tp:int, tn:int, fp:int, fn:int) -> float:
    denominator:int = tp + tn + fp + fn
    if not denominator:
        return 0.
    return (tp + tn) / denominator

def precision(tp:int, fp:int) -> float:
    denominator:int = tp + fp
    if not denominator:
        return 0.
    return tp / denominator

def recall(tp:int, fn:int) -> float:
    denominator:int = tp + fn
    if not denominator:
        return 0.
    return tp / denominator

def f1(precision:float, recall:float):
    denominator:int = precision + recall
    if not denominator:
        return 0.
    return 2 * ((precision * recall) / denominator)

