import numpy as np
import tensorflow as tf
import math
import platform
import os, sys
from mathutils import *


# histogram of 1D distances with soft asssignment and normalization
def differentialHistogram(points, histogramSetup, normalize=True):
    with tf.name_scope('differentialHistogram'):

        pointCount = points.shape[1]
        dimCount = tf.cast(points.shape[2], tf.float32)
        hBinCount = histogramSetup.binCount
        hScale = histogramSetup.scale
        hStdDev = histogramSetup.stdDev
        hNorm = histogramSetup.normalization

        # get mutual distances
        dst = mutualToroidalDistances(points, namespace='lossDst')

        # "delete" zeros on the diagonal
        dst += tf.expand_dims(tf.diag(np.float32(np.full(pointCount, LARGE_NUMBER))), 0)

        # "delete" distances not in histogram range
        dst = tf.where(
                tf.greater(dst, tf.sqrt(dimCount) / hScale),
                LARGE_NUMBER * tf.ones_like(dst),
                dst)

        # reformat to 4D
        dstExtended = tf.expand_dims(dst, -1)

        # define bin centers
        halfSize = 0.5 / hBinCount
        dstBins = tf.linspace(halfSize, 1 - halfSize, hBinCount)
        dstBins *= tf.sqrt(dimCount) / hScale
        dstBinsExtended = tf.reshape(dstBins, [1, 1, 1, hBinCount])

        # perform soft binning
        softBinning = gauss(dstExtended, dstBinsExtended, hStdDev)

        # normalize Gaussian
        gaussWeightSum = tf.reduce_sum(softBinning, [3])
        gaussWeightSum = tf.where(
                tf.less(gaussWeightSum, EPS),
                tf.ones_like(gaussWeightSum),
                gaussWeightSum)
        softBinning /= tf.expand_dims(gaussWeightSum, -1)

        # collapse to bin dimension
        histogram = tf.reduce_sum(softBinning, [1, 2])

        # average over batches
        histogram = tf.reduce_mean(histogram, [0])

        # normalize
        if normalize:
                assert hNorm is not None, "No normalization data available."
                histogram /= hNorm

        return histogram
