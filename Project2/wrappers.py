import numpy as np


def return_uint8(func):
    def func_wrapper(*args):
        output = func(*args)
        return output.astype(np.uint8)

    return func_wrapper


def return_float32(func):
    def func_wrapper(*args):
        output = func(*args)
        return output.astype(np.float32)

    return func_wrapper


def normalize(func):
    def func_wrapper(*args):
        output = func(*args)
        output[output < 0] = 0
        output[output > 255] = 255
        return output

    return func_wrapper
