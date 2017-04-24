import numpy as np
import time

np.__config__.show()

shape = (10000, 1000)

a = np.zeros(shape, dtype=np.float32)
b = np.zeros(shape, dtype=np.float32)
a[0, 0] = 1.0
b[0, 0] = 1.0

iters = 1000

start_time = time.time()
for i in range(iters):
    c = a + b
duration = time.time() - start_time
time_per_iter = duration * 1000.0 / float(iters)

print ("Shape=", shape)
print ("Time per iteration: %.3f ms" % time_per_iter)
