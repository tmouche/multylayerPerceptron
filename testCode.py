
import timeit
import numpy

import timeit

def test1():
    c = 2
    temp = numpy.arange(345)
    return temp * c

def test2():
    c = 2
    temp = numpy.arange(345)
    new = temp * c
    return new

print(timeit.timeit(test1))
print(timeit.timeit(test2))




# all_l = config.get("layer", [])
# if len(all_l) < 3:
#     raise Exception("Error log: The network needs at least 3 layers")
# if all_l["0"].get("unit", "") != "input":
#     raise Exception(f"Error log: layer {0}: unknow name {all_l[0].get('unit')}")
# x = 1
# while x < len(all_l):
#     u = all_l[str(x)]
#     unit = u.get("unit")
#     if x < len(all_l)-1 and unit != 'hidden' or x == len(all_l)-1 and unit != 'output':
#         raise Exception(f"Error log: layer {x}: unknow name {unit}")
#     s = u.get("size",0)
#     p_s = all_l[str(x-1)].get("size",0)
#     act = u.get("activation", "sigmoid")
#     init = u.get("initializer", "default")
#     if s < 1 or p_s < 1:
#         raise Exception(f"Error log: layer {x}: too small")
#     if x == len(all_l)-1:
#         if act == "softmax" and s == 1:
#             raise Exception("Error log: softmax can not be used with less than 2 outputs neurons")
#         if __loss_name == "binaryCrossEntropy" and s != 1:
#             raise Exception("Error log: The binary cross entropy can not be used with more than 1 output neurons")
#     __layers.append(Layer(s, p_s, unit, act, init))
#     x += 1