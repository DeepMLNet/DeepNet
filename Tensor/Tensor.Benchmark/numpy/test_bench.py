import numpy as np


def add(a, b):
    c = a + b


def get_data (dtype, shape):
    nelems = 1
    for i in range(len(shape)):
        nelems = nelems * shape[i]
    a = np.asarray(np.arange(nelems), dtype=dtype)
    a = np.reshape(a, shape)
    b = -np.copy(a)
    return (a,b)
   
def test_add_single_1000x1000(benchmark):
    a,b = get_data(np.single, (1000, 1000))
    benchmark(add, a, b)

def test_add_double_1000x1000(benchmark):
    a,b = get_data(np.double, (1000, 1000))
    benchmark(add, a, b)
    
def test_add_single_2000x2000(benchmark):
    a,b = get_data(np.single, (2000, 2000))
    benchmark(add, a, b)

def test_add_double_2000x2000(benchmark):
    a,b = get_data(np.double, (2000, 2000))
    benchmark(add, a, b)
        