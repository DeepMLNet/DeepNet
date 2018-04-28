import numpy as np
import pytest
import pytest_benchmark

shapes = ["1000x1000", "2000x2000", "8000x8000"]
types = ['int32', 'int64', 'single', 'double']

def params(func):
    return pytest.mark.parametrize('shape', shapes)(pytest.mark.parametrize('typ', types)(func))

def bool_params(func):
    return pytest.mark.parametrize('shape', shapes)(func)

def get_last_axis(a):
    return (len(a.shape) - 1)

def convert(typ, value):
    if typ == 'int32': 
        return int32(value)
    elif typ == 'int64':
        return int64(value)
    elif typ == 'single':
        return single(value)
    elif typ == 'double':
        return double(value)  

def get_shape_and_dtype(shape, typ):
    if typ == 'int32': 
        dtype = np.int32
    elif typ == 'int64':
        dtype = np.int64
    elif typ == 'single':
        dtype = np.single
    elif typ == 'double':
        dtype = np.double
    shape = shape.split('x')
    for i in range(len(shape)):
        shape[i] = int(shape[i])
    shape = tuple(shape)
    return shape, dtype

def get_data (shape, typ):
    shape, dtype = get_shape_and_dtype(shape, typ)
    np.random.seed (123)
    a = (np.random.sample (size=shape) * 100.0 - 50.0).astype(dtype)
    b = (np.random.sample (size=shape) * 100.0 - 50.0).astype(dtype)
    return (a,b)

def get_bool_data (shape):
    shape, _ = get_shape_and_dtype(shape, 'int32')
    np.random.seed (123)
    a = np.random.sample (size=shape) > 0.5
    b = np.random.sample (size=shape) > 0.5
    return (a,b)

@params
def test_nothing(benchmark, shape, typ): 
    a,b = get_data(shape, typ)
    benchmark(lambda: a)

@params
def test_zeros(benchmark, shape, typ): 
    shape, dtype = get_shape_and_dtype(shape, typ)
    benchmark(lambda: np.zeros(shape, dtype))

@params
def test_ones(benchmark, shape, typ): 
    shape, dtype = get_shape_and_dtype(shape, typ)
    benchmark(lambda: np.ones(shape, dtype))

@params
def test_arange(benchmark, shape, typ): 
    shape, dtype = get_shape_and_dtype(shape, typ)
    start = convert(typ, 0)
    stop = convert(typ, shape[0])
    step = convert(typ, 1)
    benchmark(lambda: np.arange(start, stop, step, dtype=dtype))

@params
def test_identity(benchmark, shape, typ): 
    shape, dtype = get_shape_and_dtype(shape, typ)
    benchmark(lambda: np.identity(shape[0], dtype))

@params
def test_copy(benchmark, shape, typ): 
    a,b = get_data(shape, typ)
    benchmark(lambda: np.copy(a))

@params
def test_negate(benchmark, shape, typ): 
    a,b = get_data(shape, typ)
    benchmark(lambda: -a)

@params
def test_add(benchmark, shape, typ): 
    a,b = get_data(shape, typ)
    benchmark(lambda: a + b)

@params
def test_subtract(benchmark, shape, typ): 
    a,b = get_data(shape, typ)
    benchmark(lambda: a - b)

@params
def test_multiply(benchmark, shape, typ): 
    a,b = get_data(shape, typ)
    benchmark(lambda: a * b)

@params
def test_divide(benchmark, shape, typ): 
    a,b = get_data(shape, typ)
    benchmark(lambda: a / b)

@params
def test_power(benchmark, shape, typ): 
    a,b = get_data(shape, typ)
    benchmark(lambda: a ** b)

@params
def test_modulo(benchmark, shape, typ): 
    a,b = get_data(shape, typ)
    benchmark(lambda: a % b)

@params
def test_dot(benchmark, shape, typ): 
    a,b = get_data(shape, typ)
    benchmark(lambda: a @ b)

@params
def test_sign(benchmark, shape, typ): 
    a,b = get_data(shape, typ)
    benchmark(lambda: np.sign(a))

@params
def test_log(benchmark, shape, typ): 
    a,b = get_data(shape, typ)
    benchmark(lambda: np.log(a))

@params
def test_log10(benchmark, shape, typ): 
    a,b = get_data(shape, typ)
    benchmark(lambda: np.log10(a))

@params
def test_exp(benchmark, shape, typ): 
    a,b = get_data(shape, typ)
    benchmark(lambda: np.exp(a))

@params
def test_sin(benchmark, shape, typ): 
    a,b = get_data(shape, typ)
    benchmark(lambda: np.sin(a))

@params
def test_cos(benchmark, shape, typ): 
    a,b = get_data(shape, typ)
    benchmark(lambda: np.tan(a))

@params
def test_arcsin(benchmark, shape, typ): 
    a,b = get_data(shape, typ)
    benchmark(lambda: np.arcsin(a))

@params
def test_arccos(benchmark, shape, typ): 
    a,b = get_data(shape, typ)
    benchmark(lambda: np.arccos(a))

@params
def test_arctan(benchmark, shape, typ): 
    a,b = get_data(shape, typ)
    benchmark(lambda: np.arctan(a))

@params
def test_sinh(benchmark, shape, typ): 
    a,b = get_data(shape, typ)
    benchmark(lambda: np.sinh(a))

@params
def test_cosh(benchmark, shape, typ): 
    a,b = get_data(shape, typ)
    benchmark(lambda: np.cosh(a))

@params
def test_tanh(benchmark, shape, typ): 
    a,b = get_data(shape, typ)
    benchmark(lambda: np.tanh(a))

@params
def test_sqrt(benchmark, shape, typ): 
    a,b = get_data(shape, typ)
    benchmark(lambda: np.sqrt(a))

@params
def test_maximum(benchmark, shape, typ): 
    a,b = get_data(shape, typ)
    benchmark(lambda: np.maximum(a))

@params
def test_minimum(benchmark, shape, typ): 
    a,b = get_data(shape, typ)
    benchmark(lambda: np.minimum(a))

@params
def test_equal(benchmark, shape, typ): 
    a,b = get_data(shape, typ)
    benchmark(lambda: a == b)

@params
def test_notequal(benchmark, shape, typ): 
    a,b = get_data(shape, typ)
    benchmark(lambda: a != b)

@params
def test_less(benchmark, shape, typ): 
    a,b = get_data(shape, typ)
    benchmark(lambda: a < b)

@params
def test_less_or_equal(benchmark, shape, typ): 
    a,b = get_data(shape, typ)
    benchmark(lambda: a <= b)

@params
def test_greater(benchmark, shape, typ): 
    a,b = get_data(shape, typ)
    benchmark(lambda: a > b)

@params
def test_greater_or_equal(benchmark, shape, typ): 
    a,b = get_data(shape, typ)
    benchmark(lambda: a >= b)

@params
def test_ifthenelse(benchmark, shape, typ): 
    a,b = get_data(shape, typ)
    p,q = get_bool_data(shape)
    benchmark(lambda: np.where(p, a, b))

@params
def test_sumaxis(benchmark, shape, typ): 
    a,b = get_data(shape, typ)
    lax = get_last_axis(a)
    benchmark(lambda: np.sum(a, axis=lax))




@bool_params
def test_not(benchmark, shape): 
    a,b = get_bool_data(shape)
    benchmark(lambda: np.logical_not(a))

@bool_params
def test_and(benchmark, shape): 
    a,b = get_bool_data(shape)
    benchmark(lambda: np.logical_and(a, b))

@bool_params
def test_or(benchmark, shape): 
    a,b = get_bool_data(shape)
    benchmark(lambda: np.logical_or(a, b))

@bool_params
def test_xor(benchmark, shape): 
    a,b = get_bool_data(shape)
    benchmark(lambda: np.logical_xor(a, b))

@bool_params
def test_count_true(benchmark, shape): 
    a,b = get_bool_data(shape)
    lax = get_last_axis(a)
    benchmark(lambda: np.count_nonzero(a, axis=lax))

