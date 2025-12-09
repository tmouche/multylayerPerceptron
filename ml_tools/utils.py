
def step(z, center:float):
    if isinstance(z, float):
        return 1. if z > center else 0.
    return [1. if x > center else 0. for x in z]

def accuracy(tp:int, tn:int, fp:int, fn:int) -> float:
    return (tp + tn) / (tp + tn + fp + fn)

def precision(tp:int, fp:int) -> float:
    return tp / (tp + fp)

def recall(tp:int, fn:int) -> float:
    return tp / (tp + fn)

def f1(precision:float, recall:float):
    return 2 * ((precision * recall) / (precision + recall))

