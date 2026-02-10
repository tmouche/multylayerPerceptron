
from  safetensors.numpy import save_file
import numpy


mylist = [[1,2,3,4,5,6], [1,2,3,4,5,6]]

tensors = {
    "a": numpy.ndarray(mylist)
}

save_file(tensors, "./test_tensors.safetensors")