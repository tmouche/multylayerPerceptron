

def step(z, center:float):
    if isinstance(z, float):
        return 1. if z > center else 0.
    return [1. if x > center else 0. for x in z]