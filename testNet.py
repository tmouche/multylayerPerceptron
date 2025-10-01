
from myMath import myMath

import time
import numpy as np
import pandas as pd
import asyncio
from myMath import myMath

from utils.logger import logger

def errorT(act, loss):
    return act*(1-act)*loss

eta = 0.5

input = [0.35, 0.7]
target = 0.5

h1 = [0.2, 0.2]
h2 = [0.3, 0.3]

o1 = [0.3, 0.9]

a1 = np.dot(input, h1)
y1 = myMath.sigmoid(a1)
print(f"a1: {a1:.2f} / y1: {y1:.2f}")

a2 = np.dot(input, h2)
y2 = myMath.sigmoid(a2)
print(f"a2: {a2:.2f} / y2: {y2:.2f}")

a3 = np.dot([y1, y2], o1)
y3 = myMath.sigmoid(a3)
print(f"a3: {a3:.2f} / y3: {y3:.2f}")

error = target - y3
print(f"error: {error:.2f}")

error_o1 = errorT(y3, error)
print(f"error o1: {error_o1:.5f}")

error_h1 = errorT(y1, o1[0]*error_o1)
print(f"error h1: {error_h1:.5f}")

error_h2 = errorT(y2, o1[1]*error_o1)
print(f"error h2: {error_h2:.5f}")


delta_o1_0 = eta * error_o1 * y1
print(f"Delta o1_0: {delta_o1_0:.5f}")
new_o1_0 = delta_o1_0 + o1[0]
delta_o1_1 = eta * error_o1 * y2
print(f"Delta o1_1: {delta_o1_1:.5f}")
new_o1_1 = delta_o1_1 + o1[1]
o1 = [new_o1_0, new_o1_1]
print(f"new o1: {o1}")

delta_h1_0 = eta * error_h1 * input[0]
print(f"Delta h1_0: {delta_h1_0:.5f}")
new_h1_0 = delta_h1_0 + h1[0]
delta_h1_1 = eta * error_h1 * input[1]
print(f"Delta h1_0: {delta_h1_1:.5f}")
new_h1_1 = delta_h1_1 + h1[1]
h1 = [new_h1_0, new_h1_1]
print(f"new h1: {h1}")

delta_h2_0 = eta * error_h2 * input[0]
print(f"Delta h1_0: {delta_h2_0:.5f}")
new_h2_0 = delta_h2_0 + h2[0]
delta_h2_1 = eta * error_h2 * input[1]
print(f"Delta h1_0: {delta_h2_1:.5f}")
new_h2_1 = delta_h2_1 + h2[1]
h2 = [new_h2_0, new_h2_1]
print(f"new h2: {h2}")

a1 = np.dot(input, h1)
y1 = myMath.sigmoid(a1)
print(f"a1: {a1:.2f} / y1: {y1:.2f}")

a2 = np.dot(input, h2)
y2 = myMath.sigmoid(a2)
print(f"a2: {a2:.2f} / y2: {y2:.2f}")

a3 = np.dot([y1, y2], o1)
y3 = myMath.sigmoid(a3)
print(f"a3: {a3:.2f} / y3: {y3:.2f}")

error = target - y3
print(f"error: {error:.2f}")