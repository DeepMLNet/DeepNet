import numpy as np
import theano
import theano.tensor as T
import h5py
import mlutils
import climin
from theano import function


datafile = h5py.File("TrnData.h5", 'r')
tst_datafile = h5py.File("TstData.h5", 'r')

ds = mlutils.Dataset({'inpt': np.asarray(datafile['Biotac']).T,
                      'trgt': np.asarray(datafile['OptimalVel']).T},
                     fractions=[1.0, 0.0, 0.0], minibatch_size=10000)

tst_inpt_data = mlutils.post(np.asarray(tst_datafile['Biotac']))[0:10000, ...].T
tst_trgt_data = mlutils.post(np.asarray(tst_datafile['OptimalVel']))[0:10000, ...].T


inpt = T.fmatrix("inpt")
trgt = T.fmatrix("trgt")

n_biotac = tst_inpt_data.shape[0]
n_velocity = tst_trgt_data.shape[0]
n_hidden = 100

ps = mlutils.ParameterSet(weights1=(n_hidden, n_biotac), bias1=(n_hidden,),
                          weights2=(n_velocity, n_hidden), bias2=(n_velocity,))

lyr1 = T.tanh(T.dot(ps.sym('weights1'), inpt) +  T.shape_padright(ps.sym('bias1')))
lyr2 = T.dot(ps.sym('weights2'), lyr1) + T.shape_padright(ps.sym('bias2'))

loss = T.mean((lyr2 - trgt)**2)
dloss = T.grad(loss, ps.sym_data)

loss_fun = mlutils.function([ps.sym_data, inpt, trgt], loss)
dloss_fun = mlutils.function([ps.sym_data, inpt, trgt], dloss)

ps.num_data[:] = np.random.uniform(-0.1, 0.1, size=ps.num_data.shape)


current_batch = 0
def loss_for_climin(p):
    global current_batch
    mb = ds.trn.minibatch(current_batch)
    return dloss_fun(p, mb['inpt'], mb['trgt'])

i=0
for opt in climin.GradientDescent(ps.num_data, loss_for_climin, step_rate=1e-4):
    current_batch += 1
    if current_batch == ds.trn.n_minibatches:
        current_batch = 0
        i += 1
        print "loss after %d iterations: %f" % (i, mlutils.gather(loss_fun(ps.num_data, tst_inpt_data, tst_trgt_data)))
        if i == 1000: break
