
from myMath import myMath

import time
import numpy as np
import pandas as pd

first = np.linspace(0, 1000, 1000)
second = np.linspace(0, 1000, 1000)

start = time.perf_counter()
third = first - second
end = time.perf_counter()
print(f"Execution time: {end - start:.6f} seconds")
start = time.perf_counter()
for i in range(len(first)):
    first[i] -= second[i]
end = time.perf_counter()
print(f"Execution time: {end - start:.6f} seconds")
