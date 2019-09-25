import numpy as np
import tensorflow as tf
import mathutils

#======================================================

def radialAverage2D(data):

    with tf.name_scope('radialProjection'):

        assert len(data.shape) == 2, "Can only radially project 2D spectra."

        resolution = int(data.shape[0])
        xlow = -resolution*0.5
        xhigh = resolution*0.5
        ylow = xlow
        yhigh = xhigh
        u = tf.range(xlow, xhigh, 1.0)
        v = tf.range(ylow, yhigh, 1.0)

        uu, vv = tf.meshgrid(u,v)
    
        r = tf.sqrt(uu**2 + vv**2)
        rflat = tf.reshape(r, [-1])
        rflat = tf.cast(tf.round(rflat), tf.int32)
    
        tbin = tf.bincount(rflat, tf.reshape(data, [-1]))
        nr = tf.bincount(rflat)
        radialprofile = tf.cast(tbin, tf.float32) / tf.cast(nr, tf.float32)
        return radialprofile[0:int(xhigh)]

#======================================================

# convert string-based projection interface to indices
def parseProjections(projectionsStrings):
    projections = []
    for pstring in projectionsStrings:
        projections.append([int(ch) for ch in pstring])
    return projections

#======================================================

# axis-aligned projection, use in conjunction with parseProjections()
def axisAlignedPointProjection(points, dimensions, namespace="aaProjection"):
    with tf.name_scope(namespace):
        return tf.gather(points, dimensions, axis=-1)

#======================================================

# undoes axisAlignedPointProjection by filling in zeros
def axisAlignedPointReProjection(points, dimensions, fullShape, namespace="aaReProjection"):
    with tf.name_scope(namespace):
        reProjPoints = tf.transpose(points, [2, 1, 0])
        expDims = np.expand_dims(dimensions, -1)
        reProjPoints = tf.scatter_nd(expDims, reProjPoints, fullShape[::-1])
        return tf.transpose(reProjPoints, [2, 1, 0])

#======================================================

def radialFalloff2D(res):
    falloff = np.linspace(0, res-1, res)
    falloff = np.tile(np.expand_dims(falloff, -1), [1, res])
    falloffT = np.transpose(falloff)
    falloff = np.stack([falloff, falloffT], -1)
    falloff = falloff - 0.5 * res
    falloff = np.linalg.norm(falloff, 2, -1)
    dcPos = int(res / 2.)
    falloff[dcPos, dcPos] = mathutils.EPS
    falloff = 1. / falloff
    falloff[dcPos, dcPos] = 0
    return falloff