import numpy as np
import time

#np.__config__.show()

shape = (10000, 1000)
print ("Shape=", shape)

a = np.zeros(shape, dtype=np.float32)
b = np.ones(shape, dtype=np.float32)

a[0, 0] = 1.0
b[0, 0] = 2.0

iters = 30

start_time = time.time()
for i in range(iters):
    c = a + b
duration = time.time() - start_time
time_per_iter = duration * 1000.0 / float(iters)
print ("Plus time per iteration: %.3f ms" % time_per_iter)

start_time = time.time()
for i in range(iters):
    c = np.abs(a)
duration = time.time() - start_time
time_per_iter = duration * 1000.0 / float(iters)
print ("Abs time per iteration: %.3f ms" % time_per_iter)

start_time = time.time()
for i in range(iters):
    c = np.dot(a.T, b)
duration = time.time() - start_time
time_per_iter = duration * 1000.0 / float(iters)
print ("Dot time per iteration: %.3f ms" % time_per_iter)

start_time = time.time()
for i in range(iters):
    c = np.sin(a)
duration = time.time() - start_time
time_per_iter = duration * 1000.0 / float(iters)
print ("Sin  time per iteration: %.3f ms" % time_per_iter)

start_time = time.time()
for i in range(iters):
    c = np.sqrt(a)
duration = time.time() - start_time
time_per_iter = duration * 1000.0 / float(iters)
print ("Sqrt time per iteration: %.3f ms" % time_per_iter)

start_time = time.time()
for i in range(iters):
    c = np.sign(a)
duration = time.time() - start_time
time_per_iter = duration * 1000.0 / float(iters)
print ("Sign time per iteration: %.3f ms" % time_per_iter)

start_time = time.time()
for i in range(iters):
    c = a ** b
duration = time.time() - start_time
time_per_iter = duration * 1000.0 / float(iters)
print ("Power time per iteration: %.3f ms" % time_per_iter)

start_time = time.time()
for i in range(iters):
    c = a < b
duration = time.time() - start_time
time_per_iter = duration * 1000.0 / float(iters)
print ("Less time per iteration: %.3f ms" % time_per_iter)
print (c)
