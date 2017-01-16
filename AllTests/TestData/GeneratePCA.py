from __future__ import division
import numpy as np
import os, sys, glob

def pca_white(data, n_components=None, return_axes=False, variances=None, axes=None, means=None):
    """
    Performs PCA whitening of the data.
    For every component the mean over the samples is zero and the variance is one.
    The covariance matrix of the components is diagonal.

    :param data: data[feature, smpl] - input data
    :param n_components: If specified, limits the number of principal components to keep.
    :param return_axes: If True, this function additionaly returns the PCA variances, corresponding axes and means.
    :param variances: If specified, use these variances instead of computing them from the data.
    :param axes: If specified, use theses axes instead of computing them from the data.
    :param means: If specified, use these means instead of computing them from the data.
    :return: if return_axes=False: whitened[comp, smpl] - PCA whitened data.
             if return_axes=True: (whitened[comp, smpl], variances[comp], axes[dim, comp], means[dim]).
    """
    data = np.asarray(data)
    n_samples = data.shape[1]

    # center data on mean
    if means is None:
        means = np.mean(data, axis=1)
    centered = data - means[:, np.newaxis]

    # calculate principal axes and their variances
    if axes is None or variances is None:
        cov = np.dot(centered, centered.T) / n_samples
        variances, axes = np.linalg.eigh(cov)
        sort_idx = np.argsort(variances)[::-1]
        variances = variances[sort_idx]
        axes = axes[:, sort_idx]
        if n_components is not None:
            variances = variances[0:n_components]
            axes = axes[:, 0:n_components]

    # transform data into that coordinate system
    pcaed = np.dot(axes.T, centered)

    # scale axes so that each has unit variance
    whitened = np.dot(np.diag(np.sqrt(1.0 / variances)), pcaed)

    if return_axes:
        return whitened, variances, axes, means
    else:
        return whitened


def pca_white_inverse(whitened, variances, axes, means):
    """
    Restores original data from PCA whitened data.
    :param whitened: whitened[comp, smpl] - PCA whitened data.
    :param variances: variances[comp] - variances of PCA components
    :param axes: axes[dim, comp] - PCA axes
    :param means: means[dim] - data means
    :return: data[feature, smpl] - reconstructed data
    """
    whitened = np.asarray(whitened)
    variances = np.asarray(variances)
    axes = np.asarray(axes)
    means = np.asarray(means)

    # reverse unit variance
    pcaed = np.dot(np.diag(np.sqrt(variances)), whitened)

    # restore original coordinate system
    centered = np.dot(axes, pcaed)

    # restore mean
    data = centered + means[:, np.newaxis]

    return data


def zca(data):
    """
    Performs zero phase component analysis (ZCA) of the data.
    :param data: data[feature, smpl] - input data
    :return: zcaed[feature, smpl] - ZCA whitened data.
             For every feature the mean over the samples is zero and the variance is one.
             The covariance matrix is diagonal.
             Furhtermore zcaed is most similar to data in the least square sense, i.e. (zcaed - data)**2 is minimized
             with the constraint that above properties are satisfied.
    """
    # get PCA whitened data
    whitened, variances, axes, _ = pca_white(data, return_axes=True)

    # rotate back into original coordinate system
    zcaed = np.dot(axes, whitened)

    return zcaed


if __name__ == '__main__':
    # load data
    f = np.load("curve/smpl00.npz")
    data = np.asarray(f['biotac'], dtype=np.float32)   # [feature, smpl]
    print data.shape

    pca_whitened_full = pca_white(data, n_components=None)
    pca_whitened_10 = pca_white(data, n_components=10)
    zca_whitened = zca(data)

    np.savez_compressed("PCA.npz",
                        data=np.asarray(data.T, dtype=np.float32),
                        pca_whitened_full=np.asarray(pca_whitened_full.T, dtype=np.float32),
                        pca_whitened_10=np.asarray(pca_whitened_10.T, dtype=np.float32),
                        zca_whitened=np.asarray(zca_whitened.T, dtype=np.float32))


