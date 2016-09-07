import numpy as np
import matplotlib.pyplot as plt
import h5py

from numpy import dot, log, pi, exp
from numpy.linalg import det, pinv



def gp_prior(x, mean_fn, cov_fn):
    """
    Samples from a Gaussian process prior.
    :param x: x[smpl] - points to sample at
    :param mean_fn: mean_fn(x) - mean function
    :param cov_fn: cov_fn(x, x') - covariance function
    :return: (mean[smpl], cov[smpl, smpl]) - mean and covariance at x
    """
    x = np.asarray(x)
    n_samples = x.shape[0]

    # build mean vector
    mu = np.zeros((n_samples,))
    for smpl in range(n_samples):
        mu[smpl] = mean_fn(x[smpl])

    # build covariance matrix
    sigma = np.zeros((n_samples, n_samples))
    for m in range(n_samples):
        for n in range(n_samples):
            sigma[m, n] = cov_fn(x[m], x[n])

    return mu, sigma


def gp_regression(tst_x, trn_x, trn_y, trn_sigma, mean_fn, cov_fn):
    """
    Performs Gaussian process regression.
    :param tst_x: tst_x[smpl] - points to predict at
    :param trn_x: trn_x[trn_smpl] - training inputs
    :param trn_y: trn_y[trn_smpl] - training targets
    :param trn_sigma: trn_sigma[trn_smpl] - standard deviation of training targets
    :param mean_fn: mean_fn(x) - mean function
    :param cov_fn: cov_fn(x, x') - covariance function
    :return: (mean[smpl], cov[smpl, smpl]) - predicted mean and covariance at x
    """
    tst_x = np.asarray(tst_x)
    trn_x = np.asarray(trn_x)
    trn_y = np.asarray(trn_y)
    n_tst_samples = tst_x.shape[0]
    n_trn_samples = trn_x.shape[0]

    # build training mean vector
    m_trn = np.zeros((n_trn_samples,))
    for smpl in range(n_trn_samples):
        m_trn[smpl] = mean_fn(trn_x[smpl])

    # build tst_x mean vector
    m_tst = np.zeros((n_tst_samples,))
    for smpl in range(n_tst_samples):
        m_tst[smpl] = mean_fn(tst_x[smpl])

    # build training covariance matrix
    K_trn_trn = np.zeros((n_trn_samples, n_trn_samples))
    for m in range(n_trn_samples):
        for n in range(n_trn_samples):
            K_trn_trn[m, n] = cov_fn(trn_x[m], trn_x[n])
            if n == m:
                K_trn_trn[m, n] += trn_sigma[m]**2.0

    # build test covariance matrix
    K_tst_tst = np.zeros((n_tst_samples, n_tst_samples))
    for m in range(n_tst_samples):
        for n in range(n_tst_samples):
            K_tst_tst[m, n] = cov_fn(tst_x[m], tst_x[n])

    # build training/test covariance matrix
    K_trn_tst = np.zeros((n_trn_samples, n_tst_samples))
    for m in range(n_trn_samples):
        for n in range(n_tst_samples):
            K_trn_tst[m, n] = cov_fn(trn_x[m], tst_x[n])

    # predict
    K_trn_trn_inv = np.linalg.pinv(K_trn_trn)
    tst_mean = m_tst + np.dot(K_trn_tst.T, np.dot(K_trn_trn_inv, trn_y - m_trn))
    tst_cov = K_tst_tst - np.dot(K_trn_tst.T, np.dot(K_trn_trn_inv, K_trn_tst))

    # fix numerical instabilities
    epsilon = 1e-8
    tst_cov[tst_cov < epsilon] = epsilon
    return tst_mean, tst_cov


def gp_uncertain_regression(tst_mu, tst_var, trn_x, trn_y, trn_sigma, cov_fn):
    """
    Performs one-dimensional Gaussian process regression with uncertain test inputs.
    Mean function is zero.
    :param tst_mu: tst_mu - mean of point to predict at
    :param tst_var: tst_var - variance of point to predict at
    :param trn_x: trn_x[trn_smpl] - training inputs
    :param trn_y: trn_y[trn_smpl] - training targets
    :param trn_sigma: trn_sigma[trn_smpl] - standard deviation of training targets
    :param cov_fn: dictionary of covaraince function information {'type': 'cov func name', ...}
    :return: (mean, var) - predicted mean and variance
    """
    tst_mu = float(tst_mu)
    tst_var = float(tst_var)
    trn_x = np.asarray(trn_x)
    trn_y = np.asarray(trn_y)
    trn_sigma = np.asarray(trn_sigma)
    n_trn_samples = trn_x.shape[0]

    # build training covariance matrix
    if cov_fn['type'] == 'SE':
        cov_f = make_cov_se(1.0, cov_fn['l'])
    elif cov_fn['type'] == 'ID':
        cov_f = cov_id
    K_trn_trn = np.zeros((n_trn_samples, n_trn_samples))
    for m in range(n_trn_samples):
        for n in range(n_trn_samples):
            K_trn_trn[m, n] = cov_f(trn_x[m], trn_x[n])
            if n == m:
                K_trn_trn[m, n] += trn_sigma[m] ** 2.0
    K_trn_trn_inv = np.linalg.pinv(K_trn_trn)

    # build E[K(tst_x, trn_x_k)]
    E_K_tst_trn = np.zeros(n_trn_samples)
    for j in range(n_trn_samples):
        if cov_fn['type'] == 'SE':
            l = cov_fn['l']
            E_K_tst_trn[j] = np.sqrt(l ** 2. / (l ** 2. + tst_var)) * \
                             np.exp(-(tst_mu - trn_x[j])**2. / (2. * (l ** 2. + tst_var)))
        elif cov_fn['type'] == 'ID':
            E_K_tst_trn[j] = tst_mu * trn_x[j]

    # calculate mean
    mean = np.dot(E_K_tst_trn.T, np.dot(K_trn_trn_inv, trn_y))

    # calculate variance
    if cov_fn['type'] == 'SE':
        l = cov_fn['l']

        # calculate matrix L
        L = np.zeros((n_trn_samples, n_trn_samples))
        for i in range(n_trn_samples):
            for j in range(n_trn_samples):
                L[i,j] = \
                    np.sqrt(l ** 2. / (l ** 2. + 2. * tst_var)) * \
                    np.exp(-(tst_mu - (trn_x[i] + trn_x[j])/2.)**2. / (l ** 2. + 2. * tst_var) -
                           (trn_x[i] - trn_x[j]) ** 2. / (4. * l**2.))

        # define beta and lv
        beta = np.dot(K_trn_trn_inv, trn_y)[:, np.newaxis]
        lv = E_K_tst_trn[:, np.newaxis]

        # calculate variance
        var = \
            1. - np.trace(np.dot(K_trn_trn_inv - np.dot(beta, beta.T), L)) - \
            np.trace(np.dot(np.dot(lv, lv.T), np.dot(beta, beta.T)))
    elif cov_fn['type'] == 'ID':
        var = \
            (tst_mu + tst_var) * \
            (1. - np.dot(trn_x.T, np.dot(K_trn_trn_inv, trn_x)) + np.dot(trn_x.T, np.dot(K_trn_trn_inv, trn_y))**2.) - \
            tst_mu**2. * np.dot(trn_x.T, np.dot(K_trn_trn_inv, trn_y))**2.

    return mean, var


def multi_gp_uncertain_regression(tst_mu, tst_Sigma, trn_x, trn_y, trn_sigma, cov_fn):
    """
    Performs one-dimensional Gaussian process regression with uncertain test inputs.
    Mean function is zero.
    :param tst_mu: tst_mu[gp] - mean of input for each GP
    :param tst_Sigma: tst_sigma[gp, gp'] - covariance between inputs of each GP
    :param trn_x: trn_x[gp, trn_smpl] - training inputs for each GP
    :param trn_y: trn_y[gp, trn_smpl] - training targets for each GP
    :param trn_sigma: trn_sigma[gp, trn_smpl] - standard deviation of training targets for each GP
    :param cov_fn: dictionary of covaraince function information {'type': 'cov func name', ...}
    :return: (mean[gp], cov[gp, gp']) - predicted mean and covariance matrix
    """
    tst_mu = np.asarray(tst_mu)
    tst_Sigma = np.asarray(tst_Sigma)
    trn_x = np.asarray(trn_x)
    trn_y = np.asarray(trn_y)
    trn_sigma = np.asarray(trn_sigma)
    n_gps = trn_x.shape[0]
    n_trn_samples = trn_x.shape[1]

    # build training covariance matrix
    K_trn_trn = np.zeros((n_gps, n_trn_samples, n_trn_samples))
    K_trn_trn_inv = np.zeros_like(K_trn_trn)
    for g in range(n_gps):
        if cov_fn['type'] == 'SE':
            cov_f = make_cov_se(1.0, cov_fn['l'][g])
        elif cov_fn['type'] == 'ID':
            cov_f = cov_id
        for m in range(n_trn_samples):
            for n in range(n_trn_samples):
                K_trn_trn[g, m, n] = cov_f(trn_x[g, m], trn_x[g, n])
                if n == m:
                    K_trn_trn[g, m, n] += trn_sigma[g, m] ** 2.0
        K_trn_trn_inv[g, :, :] = np.linalg.pinv(K_trn_trn[g, :, :])
    # build E[K(tst_x, trn_x_k)]
    E_K_tst_trn = np.zeros((n_gps, n_trn_samples))
    for g in range(n_gps):
        for j in range(n_trn_samples):
            if cov_fn['type'] == 'SE':
                l = cov_fn['l'][g]
                E_K_tst_trn[g, j] = np.sqrt(l**2. / (l**2. + tst_Sigma[g, g])) * \
                                    np.exp(-(tst_mu[g] - trn_x[g, j])**2. / (2. * (l ** 2. + tst_Sigma[g, g])))
            elif cov_fn['type'] == 'ID':
                E_K_tst_trn[g, j] = tst_mu * trn_x[j]

    # calcualte "targets"
    tgt = np.zeros((n_gps, n_trn_samples))
    for g in range(n_gps):
        tgt[g, :] = np.dot(K_trn_trn_inv[g, :, :], trn_y[g, :])

    # calculate means
    mean = np.zeros(n_gps)
    for g in range(n_gps):
        mean[g] = np.dot(E_K_tst_trn[g,:], tgt[g, :])

    print "Kk=\n", K_trn_trn
    print "Kk_inv=\n", K_trn_trn_inv
    print "lk=\n", E_K_tst_trn
    print "beta=\n", tgt


    # calculate covariance
    cov = np.zeros((n_gps, n_gps))
    if cov_fn['type'] == 'SE':
        for g1 in range(n_gps):
            for g2 in range(n_gps):
                l1 = cov_fn['l'][g1]
                l2 = cov_fn['l'][g2]

                if g1 == g2:  # variance
                    # calculate matrix L
                    L = np.zeros((n_trn_samples, n_trn_samples))
                    for i in range(n_trn_samples):
                        for j in range(n_trn_samples):
                            L[i,j] = \
                                np.sqrt(l1 ** 2. / (l1 ** 2. + 2. * tst_Sigma[g1, g1])) * \
                                np.exp(-(tst_mu[g1] - (trn_x[g1, i] + trn_x[g1, j])/2.)**2. / (l1 ** 2. + 2. * tst_Sigma[g1, g1]) -
                                       (trn_x[g1, i] - trn_x[g1, j]) ** 2. / (4. * l1**2.))

                    # define beta and lv
                    beta = np.dot(K_trn_trn_inv[g1, :, :], trn_y[g1, :])[:, np.newaxis]
                    lv = E_K_tst_trn[g1, :, np.newaxis]

                    # calculate variance
                    cov[g1, g1] = \
                        1. - np.trace(np.dot(K_trn_trn_inv[g1, :, :] - np.dot(beta, beta.T), L)) - \
                        np.trace(np.dot(np.dot(lv, lv.T), np.dot(beta, beta.T)))

                else:      # covariance
                    T = np.zeros((n_trn_samples, n_trn_samples))
                    for i in range(n_trn_samples):
                        for j in range(n_trn_samples):
                            mu_s = np.asarray([trn_x[g1, i], trn_x[g2, j]])
                            Sigma_s = np.asarray([[l1**2., 0.],
                                                  [0,      l2**2.]])

                            mu_y = np.asarray([tst_mu[g1], tst_mu[g2]])
                            Sigma_y = np.asarray([[tst_Sigma[g1, g1], tst_Sigma[g1, g2]],
                                                  [tst_Sigma[g2, g1], tst_Sigma[g2, g2]]])

                            # Sigma_sum_inv = pinv(Sigma_s + Sigma_y)

                            S1 = -0.5*dot(mu_s, dot(pinv(Sigma_s), mu_s))
                            S2 = -log(2.*pi) - 0.5*log(det(Sigma_y)) - 0.5*dot(mu_y, dot(pinv(Sigma_y), mu_y))
                            Sn = -log(2.*pi) + 0.5*log(det(pinv(Sigma_s) + pinv(Sigma_y))) - \
                                0.5 * np.dot((np.dot(pinv(Sigma_s), mu_s) + np.dot(pinv(Sigma_y), mu_y)).T,
                                             np.dot(pinv(pinv(Sigma_s) + pinv(Sigma_y)),
                                                    np.dot(pinv(Sigma_s), mu_s) + np.dot(pinv(Sigma_y), mu_y)))
                            # Sn = -log(2.*pi) + 0.5*log(det(pinv(Sigma_s) + pinv(Sigma_y))) - \
                            #     np.dot(mu_s, np.dot(Sigma_sum_inv, mu_y)) - \
                            #     0.5*dot(mu_s, dot(Sigma_sum_inv, dot(Sigma_y, dot(pinv(Sigma_s), mu_s)))) - \
                            #     0.5*dot(mu_y, dot(Sigma_sum_inv, dot(Sigma_s, dot(pinv(Sigma_y), mu_y))))

                            T[i, j] = exp(S1 + S2 - Sn)

                            if g1 == 0 and g2 == 1 and i == 0 and j == 0:
                                print "mu_k=%f    mu_l=%f" % (tst_mu[g1], tst_mu[g2])
                                print "Sigma_kk=%f    Sigma_kl=%f    Sigma_ll=%f" % \
                                      (tst_Sigma[g1, g1], tst_Sigma[g1, g2], tst_Sigma[g2, g2])
                                print "x_ki=%f     x_lj=%f" % (trn_x[g1, i], trn_x[g2, j])
                                print "l_k=%f      l_l=%f" % (l1, l2)
                                print "T[i,j]=%f" % T[i,j]

                    if g1 == 0 and g2 == 1:
                        print "T=\n", T

                    cov[g1, g2] = np.dot(trn_y[g1, :],
                                         np.dot(K_trn_trn_inv[g1, :, :],
                                                np.dot(T,
                                                       np.dot(K_trn_trn_inv[g2, :, :],
                                                              trn_y[g2, :])))) - mean[g1] * mean[g2]
    elif cov_fn['type'] == 'ID':
        raise NotImplementedError()

    return mean, cov


def gp_uncertain_regression_by_sampling(tst_mu, tst_var, trn_x, trn_y, trn_sigma, cov_fn, n_samples):
    """
    Performs one-dimensional Gaussian process regression with uncertain test inputs by sampling.
    Mean function is zero.
    :param tst_mu: tst_mu - mean of point to predict at
    :param tst_var: tst_var - variance of point to predict at
    :param trn_x: trn_x[trn_smpl] - training inputs
    :param trn_y: trn_y[trn_smpl] - training targets
    :param trn_sigma: trn_sigma[trn_smpl] - standard deviation of training targets
    :param cov_fn: dictionary of covaraince function information {'type': 'cov func name', ...}
    :param n_samples: number of samples to use for sampling
    :return: (mean, var) - estimated mean and variance
    """
    # draw samples from tst
    tst_x = np.random.normal(tst_mu, np.sqrt(tst_var), size=n_samples)

    # apply standard GP on sampled inputs
    if cov_fn['type'] == 'SE':
        cov_f = make_cov_se(1.0, cov_fn['l'])
    elif cov_fn['type'] == 'ID':
        cov_f = cov_id
    y_mean, y_cov = gp_regression(tst_x, trn_x, trn_y, trn_sigma, zero_fn, cov_f)
    y_std = np.sqrt(np.diagonal(y_cov))

    # sample from predicted distribution given tst_x
    tst_y = np.zeros(n_samples)
    for smpl in range(n_samples):
        tst_y[smpl] = np.random.normal(y_mean[smpl], y_std[smpl])

    # calculate mean and variance
    mean = np.mean(tst_y)
    var = np.var(tst_y)
    return mean, var


def multi_gp_uncertain_regression_by_sampling(tst_mu, tst_Sigma, trn_x, trn_y, trn_sigma, cov_fn, n_samples):
    """
    Performs one-dimensional Gaussian process regression with uncertain test inputs by sampling.
    Mean function is zero.
    :param tst_mu: tst_mu[gp] - mean of input for each GP
    :param tst_Sigma: tst_sigma[gp, gp'] - covariance between inputs of each GP
    :param trn_x: trn_x[gp, trn_smpl] - training inputs for each GP
    :param trn_y: trn_y[gp, trn_smpl] - training targets for each GP
    :param trn_sigma: trn_sigma[gp, trn_smpl] - standard deviation of training targets for each GP
    :param cov_fn: dictionary of covaraince function information {'type': 'cov func name', ...}
    :param n_samples: number of samples to use for sampling
    :return: (mean[gp], cov[gp, gp']) - predicted mean and covariance matrix
    """
    n_gps = trn_x.shape[0]

    # draw samples from tst
    # tst_x[smpl, gp]
    tst_x = np.zeros((n_samples, n_gps))

    for smpl in range(n_samples):
        tst_x[smpl, :] = np.random.multivariate_normal(tst_mu, tst_Sigma)

    # apply standard GPs on sampled inputs
    # tst_y[smpl, gpl]
    tst_y = np.zeros((n_samples, n_gps))
    for g in range(n_gps):
        if cov_fn['type'] == 'SE':
            cov_f = make_cov_se(1.0, cov_fn['l'][g])
        elif cov_fn['type'] == 'ID':
            cov_f = cov_id
        y_mean, y_cov = gp_regression(tst_x[:, g], trn_x[g, :], trn_y[g, :], trn_sigma[g, :], zero_fn, cov_f)
        y_std = np.sqrt(np.diagonal(y_cov))

        # sample from predicted distribution given tst_x
        for smpl in range(n_samples):
            tst_y[smpl, g] = np.random.normal(y_mean[smpl], y_std[smpl])

    # calculate mean and variance
    mean = np.mean(tst_y, axis=0)
    cov = np.cov(tst_y.T)
    return mean, cov


def test_gp_uncertain_regression():
    np.random.seed(1)

    n_sampling = 2000
    n_training = 30
    n_test = 10
    trn_l = 2.0
    cov_fn = dict(type='SE', l=1.0)

    # sample training points from a GP prior
    trn_x = np.random.uniform(-10., 10., n_training)
    trn_x = np.sort(trn_x)
    trn_mean, trn_cov = gp_prior(trn_x, zero_fn, make_cov_se(1.0, trn_l))
    assert np.min(np.linalg.eigvals(trn_cov)) > -1e-5
    trn_y = np.random.multivariate_normal(trn_mean, trn_cov)
    trn_sigma = np.zeros(n_training)
    # plt.plot(trn_x, trn_y, '-xk')
    # plt.show()

    # sample some test points
    for n in range(n_test):
        tst_mu = np.random.uniform(-10., 10.)
        tst_var = np.random.uniform(0., 2.0)

        pred_mean, pred_var = gp_uncertain_regression(tst_mu, tst_var, trn_x, trn_y, trn_sigma, cov_fn)
        sampling_mean, sampling_var = \
            gp_uncertain_regression_by_sampling(tst_mu, tst_var, trn_x, trn_y, trn_sigma, cov_fn, n_sampling)

        print "predicted: mean=%7.3f  var=%7.3f" % (pred_mean, pred_var)
        print "sampled:   mean=%7.3f  var=%7.3f" % (sampling_mean, sampling_var)
        print ""


def test_multi_gp_uncertain_regression():
    print "\n"
    print "test_multi_gp_uncertain_regression ============================================"
    print "\n"

    np.random.seed(1)

    n_sampling = 2000
    n_training = 30
    n_test = 2
    n_gps = 3
    trn_l = 2.0
    l = [1.0, 1.5, 2.0]
    cov_fn = dict(type='SE', l=l)

    trn_x = np.zeros((n_gps, n_training))
    trn_y = np.zeros((n_gps, n_training))
    trn_sigma = np.zeros((n_gps, n_training))
    #hf = h5py.File('MulitGpTest2.h5')
    #hf2 = h5py.File('trnx.h5')
    for gp in range(n_gps):
        # sample training points from a GP prior
        trn_x[gp, :] = np.random.uniform(-5., 5., n_training)
        trn_x[gp, :] = np.sort(trn_x[gp, :])
        trn_mean, trn_cov = gp_prior(trn_x[gp, :], zero_fn, make_cov_se(1.0, trn_l))
        assert np.min(np.linalg.eigvals(trn_cov)) > -1e-5
        trn_y[gp, :] = np.random.multivariate_normal(trn_mean, trn_cov)
        trn_sigma[gp, :] = np.zeros(n_training)
        # plt.plot(trn_x, trn_y, '-xk')
        # plt.show()
    print "trn_x = "
    print trn_x
    print "\n"
    print "trn_x = "
    print trn_x
    print "\n"
    #hf.create_dataset("trn_x1",data = trn_x)
    #hf2.create_dataset("trn_x",data = trn_x,dtype=float )
    #hf.create_dataset("trn_y",data = trn_y)
    #hf.create_dataset("trn_sigma",data = trn_sigma)
    #hf.create_dataset("l",data = l)

    # sample some test points
    for n in range(n_test):
        tst_mu = np.random.uniform(-10., 10., size=n_gps)
        tst_cov = np.random.uniform(0., 2.0, size=(n_gps, n_gps))
        # make PSD
        tst_cov = np.dot(tst_cov.T, tst_cov)
        #hf.create_dataset("tst_mu%i"%(n),data = tst_mu)
        #hf.create_dataset("tst_cov%i"%(n),data = tst_cov)

        pred_mean, pred_cov = multi_gp_uncertain_regression(tst_mu, tst_cov, trn_x, trn_y, trn_sigma, cov_fn)
        sampling_mean, sampling_cov = \
            multi_gp_uncertain_regression_by_sampling(tst_mu, tst_cov, trn_x, trn_y, trn_sigma, cov_fn, n_sampling)
                
        #hf.create_dataset("pred_mean%i"%(n),data = pred_mean)
        #hf.create_dataset("pred_cov%i"%(n),data = pred_cov)
        #hf.create_dataset("sampling_mean%i"%(n),data = sampling_mean)
        #hf.create_dataset("sampling_cov%i"%(n),data = sampling_cov)

        print "predicted:"
        print "means=", pred_mean
        print "cov=\n", pred_cov
        print
        print "sampled:"
        print "means=", sampling_mean
        print "cov=\n", sampling_cov
        print
        print

def file_test_multi_gp_uncertain_regression():
    
    print "\n"
    print "file_test_multi_gp_uncertain_regression ============================================"
    print "\n"
    h5train = h5py.File("bin/Debug/TrainData.h5",'r')
    h5test = h5py.File("bin/Debug/TestData.h5",'r')
    np.random.seed(1)

    n_sampling = 2000

    l = h5train.get("Lengthscale")
    l = l[0]
    print "l = "
    print l
    print "\n"
    cov_fn = dict(type='SE', l=l)

    trn_x = np.asarray(h5train.get("Trn_X"),dtype = float)
    trn_x = trn_x[0]
    print "trn_x = "
    print trn_x
    print "\n"
    trn_y = np.asarray(h5train.get("Trn_T") ,dtype = float)
    trn_y = trn_y[0]
    print "trn_y = "
    print trn_y
    print "\n"
    trn_sigma = np.asarray(h5train.get("Trn_Sigma"),dtype = float)
    trn_sigma = trn_sigma[0]

    tst_mus = np.asarray(h5test.get("In_Mean"),dtype = float)
    tst_covs = np.asarray(h5test.get("In_Cov"),dtype = float)

    file_pred_mus = np.asarray(h5test.get("Pred_Mean"),dtype = float)
    file_pred_covs = np.asarray(h5test.get("Pred_Cov"),dtype = float)

    n_training = trn_x.shape[1]
    n_test = tst_mus.shape[0]
    n_gps = l.shape[0]

    absMeanErrors = np.zeros(shape=(n_test,n_gps))
    absCovErrors = np.zeros(shape=(n_test,n_gps,n_gps))
    print "n_training = %d\n"%(n_training)
    print "n_test = %d\n"%(n_test)
    print "n_gps = %d\n"%(n_gps)
    # sample some test points
    for n in range(n_test):
        tst_mu = tst_mus[n][0]
        tst_cov = tst_covs[n][0]

        pred_mean, pred_cov = multi_gp_uncertain_regression(tst_mu, tst_cov, trn_x, trn_y, trn_sigma, cov_fn)
        #sampling_mean, sampling_cov = \
        #    multi_gp_uncertain_regression_by_sampling(tst_mu, tst_cov, trn_x, trn_y, trn_sigma, cov_fn, n_sampling)
        file_pred_mean = file_pred_mus[n][0]        
        file_pred_cov = file_pred_covs[n][0]
        absMeanErrors[n] = np.abs(file_pred_mean - pred_mean)  
        absCovErrors[n] = np.abs(file_pred_cov - pred_cov)
        print "predicted:"
        print "means=", pred_mean
        print "cov=\n", pred_cov
        print
        #print "sampled:"
        #print "means=", sampling_mean
        #print "cov=\n", sampling_cov
        #print
        print "file predicted:"
        print "means=", file_pred_mean
        print "cov=\n", file_pred_cov
        print
        print
    print    
    print "All prediction mean errors:"
    print absMeanErrors
    print
    print "Highest prediction mean error:"
    print np.max(absMeanErrors, axis=0)
    print
    print "Mean prediction mean error:"
    print np.mean(absMeanErrors,axis=0)
    print
    """
    print "All prediction cov errors:"
    print absCovErrors
    print
    """
    print "Highest prediction cov error:"
    print np.max(absCovErrors, axis=0)
    print
    print "Mean prediction cov error:"
    print np.mean(absCovErrors,axis=0)
    print
def cov_id(x, y):
    return x * y

def make_cov_se(sigma, l):
    def cov_se(x, y):
        return sigma * np.exp(-(x-y)**2. / (2. * l **2.))
    return cov_se


def make_cov_se_tanh(sigma, sigma2, l):
    def cov_se(x, y):
        return sigma * np.exp(-(x-y)**2. / (2. * l**2.)) + sigma2 * np.sum(np.tanh(x) * np.tanh(y))
    return cov_se

def make_cov_se_id(sigma, sigma2, l):
    def cov_se(x, y):
        return sigma * np.exp(-(x-y)**2. / (2. * l**2.)) + sigma2 * np.sum(x * y)
    return cov_se


def zero_fn(x):
    return 0.0

def identity_fn(x):
    return x

def tanh_fn(x):
    return np.tanh(x)

def plot_mean_cov(x, mean, cov):
    """
    Plots a function with normal distributed points.
    :param x: x[smpl] - x coordinates
    :param mean: mean[smpl] - mean vector
    :param cov: cov[smpl, smpl] - covariance matrix
    """
    var = np.diagonal(cov)
    std = np.sqrt(var)
    plt.plot(x, mean, 'k')
    plt.hold(True)
    plt.plot(x, mean + std, 'b')
    plt.plot(x, mean - std, 'b')



def demo_prior():
    x = np.linspace(-5, 5, 60)

    mu, sigma = gp_prior(x, zero_fn, make_cov_se(0.01, 1.0))

    # cov_fn = make_cov_se(0.01, 1.0)
    # mu, sigma = gp_prior(x, tanh_fn, cov_fn)

    # cov_fn = make_cov_se_tanh(0.1, 1, 1.0)
    # mu, sigma = gp_prior(x, zero_fn, cov_fn)

    # cov_fn = make_cov_se_id(0.5, 1, 1.0)
    # mu, sigma = gp_prior(x, identity_fn, cov_fn)

    # print "x=",x
    # print "mu=",mu
    # print "sigma=",sigma

    plot_mean_cov(x, mu, sigma)

    # sample
    for s in range(5):
        ys = np.random.multivariate_normal(mu, sigma)
        plt.plot(x, ys, 'r')

    plt.ylim(-1, 1)
    plt.show()


def demo_regression():
    tst_x = np.linspace(-8, 8, 80)

    trn_x = np.asarray([0.0, 2.0, -4.0])
    trn_y = np.asarray([2.0, -.3,  1.0])
    trn_s = np.asarray([0.0, 0.0,  0.2])

    mean_fn = zero_fn
    # mean_fn = identity_fn
    cov_fn = make_cov_se(0.1, 1.0)
    # cov_fn = make_cov_se_id(0.01, 0.01, 1.0)
    mu, sigma = gp_regression(tst_x, trn_x, trn_y, trn_s, mean_fn, cov_fn)

    plot_mean_cov(tst_x, mu, sigma)
    plt.plot(trn_x, trn_y, 'x')

    # sample
    for s in range(5):
        ys = np.random.multivariate_normal(mu, sigma)
        plt.plot(tst_x, ys, 'r')

    plt.xlim(-8, 8)
    plt.ylim(-6, 6)
    plt.show()


def debug_cuda():
    path = r"C:\Local\surban\GPTransfer\GPTransfer\test_input.h5"
    tst_smpl = 1
    f = h5py.File(path, 'r')

    tst_mu = np.asarray(f['tst_mu'])[tst_smpl, :]
    tst_Sigma = np.asarray(f['tst_Sigma'])[tst_smpl, :, :]
    trn_x = np.asarray(f['trn_x'])
    trn_y = np.asarray(f['trn_y'])
    trn_sigma = np.asarray(f['trn_sigma'])
    cov_l = np.asarray(f['cov_l'])
    cov_fn = dict(type='SE', l=cov_l)

    print "Using sample %d" % tst_smpl
    print "tst_mu=\n", tst_mu
    print "tst_Sigma=\n", tst_Sigma
    print "trn_x=\n", trn_x
    print "trn_y=\n", trn_y
    print "trn_sigma=\n", trn_sigma
    print "cov_l=\n", cov_l
    print

    mean, cov = multi_gp_uncertain_regression(tst_mu, tst_Sigma, trn_x, trn_y, trn_sigma, cov_fn)

    print "mean=\n", mean
    print "cov=\n", cov


if __name__ == '__main__':
    # demo_prior()
    # demo_regression()

    #test_gp_uncertain_regression()
    #test_multi_gp_uncertain_regression()
    file_test_multi_gp_uncertain_regression()

    #debug_cuda()



