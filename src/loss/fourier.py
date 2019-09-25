import numpy as np
import tensorflow as tf
import math
from mathutils import *

#======================================================

def powerSpectrum1D(points, fourierSetup):

    with tf.name_scope('spectrum1D'):
        batchCount = points.shape[0]
        pointCount = tf.cast(points.shape[1], tf.float32)
        dimCount = points.shape[2]

        assert dimCount == 1, "powerSpectrum1D() can only handle 1D inputs, {0}D provided.".format(dimCount)

        # Compute the fourier power spectrum
        xlow = 0.0
        xhigh = fourierSetup.resolution * fourierSetup.freqStep
        halfres = int(fourierSetup.resolution)
        ylow = xlow
        yhigh = xhigh

        u = tf.range(xlow, xhigh, fourierSetup.freqStep)
        uu = tf.meshgrid(u)
        grid = tf.to_float(uu)
        batchGrid = tf.tile(tf.expand_dims(grid,0),[batchCount,1,1])

        dotXU = tf.tensordot(points, batchGrid, [[2],[1]])
        angle = tf.scalar_mul(2.0*tf.constant(math.pi), dotXU)

        angleout = tf.reduce_mean(angle, 2)
        realCoeff = tf.reduce_sum(tf.cos(angleout), 1)
        imagCoeff = tf.reduce_sum(tf.sin(angleout), 1)
        power = (realCoeff**2 + imagCoeff**2) / pointCount

        # Average across all mini batches
        power = tf.reduce_mean(power, 0)

        if fourierSetup.cancelDC:
            dcPos = 0
            dcComp = power[dcPos]
            power -= tf.scatter_nd([[dcPos]], [dcComp], power.shape)

        return power

#======================================================

def powerSpectrum2D(points, fourierSetup):

    with tf.name_scope('spectrum2D'):
        batchCount = points.shape[0]
        pointCount = tf.cast(points.shape[1], tf.float32)
        dimCount = points.shape[2]

        assert dimCount == 2, "powerSpectrum2D() can only handle 2D inputs, {0}D provided.".format(dimCount)

        #Compute the fourier power spectrum
        resFactor = 1 if fourierSetup.target.ndim == 2 else 2
        spectrumRes = fourierSetup.resolution * resFactor

        xlow = -spectrumRes * fourierSetup.freqStep * 0.5
        xhigh = spectrumRes * fourierSetup.freqStep * 0.5
        halfres = int(spectrumRes * 0.5)
        ylow = xlow
        yhigh = xhigh

        u = tf.range(xlow, xhigh, fourierSetup.freqStep)
        v = tf.range(xlow, xhigh, fourierSetup.freqStep)
        uu, vv = tf.meshgrid(u,v)
        grid = tf.to_float([uu,vv])
        batchGrid = tf.tile(tf.expand_dims(grid,0),[batchCount,1,1,1])

        dotXU = tf.tensordot(points, batchGrid, [[2],[1]])
        angle = tf.scalar_mul(2.0*tf.constant(math.pi), dotXU)

        angleout = tf.reduce_mean(angle, 2)
        realCoeff = tf.reduce_sum(tf.cos(angleout), 1)
        imagCoeff = tf.reduce_sum(tf.sin(angleout), 1)
        power = (realCoeff**2 + imagCoeff**2) / pointCount

        # Average across all mini batches
        power = tf.reduce_mean(power, 0)

        if fourierSetup.cancelDC:
            dcPos = int(spectrumRes / 2.)
            dcComp = power[dcPos, dcPos]
            power -= tf.scatter_nd([[dcPos, dcPos]], [dcComp], power.shape)

        return power

#======================================================

# Monte Carlo estimate of radially averaged power spectrum
def radialSpectrumMC(points, fourierSetup):

    with tf.name_scope('radialSpectrumMC'):

        #-------------------------------------------

        def sampleSpectrum(input):
            freqSamples = input[0]
            points = input[1]
            pointCount = tf.cast(tf.shape(points)[0], tf.float32)
            dotProduct = tf.tensordot(freqSamples, points, [[2], [1]])
            twoPi = 2.0 * math.pi
            real = tf.cos(twoPi * dotProduct)
            imag = tf.sin(twoPi * dotProduct)
            sumReal = tf.reduce_sum(real, -1)
            sumImag = tf.reduce_sum(imag, -1)
            power = (sumReal * sumReal + sumImag * sumImag) / pointCount
            return power

        #-------------------------------------------

        def ceilAwayFromZero(input):
            return tf.sign(input) * tf.ceil(tf.abs(input))

        #-------------------------------------------
         
        batchSize, _, dimCount = points.shape
        freqRes = fourierSetup.resolution
        freqStep = fourierSetup.freqStep
        mcSampleCount = fourierSetup.mcSamplesPerShell

        # generate normal samples
        normDst = tf.distributions.Normal(
            loc=np.full((dimCount,), 0.),
            scale=np.full((dimCount,), 1.))
        mcSamples = tf.cast(normDst.sample([batchSize, freqRes, mcSampleCount]), tf.float32)

        # project samples to unit hypersphere
        # https://dl.acm.org/citation.cfm?id=377946
        shellSamples = tf.nn.l2_normalize(mcSamples, axis=-1, epsilon=SQRT_EPS)
        
        # scale shells by frequencies
        frequencies = tf.range(
            start = 0,
            limit = freqRes * freqStep, 
            delta = freqStep, 
            dtype = tf.float32)
        
        shellSamples *= tf.reshape(frequencies, [1, freqRes, 1, 1])
        #shellSamples = tf.round(shellSamples)
        shellSamples = ceilAwayFromZero(shellSamples)
        
        # power spectrum for each frequency sample
        spectrum = tf.map_fn(
            lambda b: sampleSpectrum(b), 
            (shellSamples, points),
            dtype=tf.float32)
        
        # radial and batch average
        avg = tf.reduce_mean(spectrum, [0, -1])

        if fourierSetup.cancelDC:
            dcComp = avg[0]
            avg -= tf.scatter_nd([[0]], [dcComp], avg.shape)

        return avg
