import numpy as np
import tensorflow as tf
import math

EPS = 10e-5
SQRT_EPS = 10e-12
DST_EPS = 10e-7
LARGE_NUMBER = 10e8

#------------------------------------------------

def lerp(a, b, w):
    return ((1.0 - w) * a) + (w * b)

#------------------------------------------------

def linearRemap(value, inputMin, inputMax, outputMin, outputMax):
    return (value - inputMin) * (outputMax - outputMin) / (inputMax - inputMin) + outputMin

#------------------------------------------------

# decaying function from b to a on [0, l]
def rampDown(x, l, a, b):
    return (b - a) * math.exp(-5.0 * (x/l)**2) + a

#------------------------------------------------

def l1Loss(x, target):
    with tf.name_scope('l1loss'):
        return tf.reduce_sum(tf.abs(x - target))

#------------------------------------------------

def l2Loss(x, target):
    with tf.name_scope('l2loss'):
        return tf.reduce_sum(tf.square(x - target))

#------------------------------------------------

# element-wise Gaussian
def gauss(x, mean, stdDev):
    return tf.exp(-tf.square(x - mean) / (2 * tf.square(stdDev)))

#------------------------------------------------

# mutual toroidal distances between a set of nD points
def mutualToroidalDistances(points, identityLarge=False, namespace="toroidDst"):
    with tf.name_scope(namespace):
        pointsExtended = tf.expand_dims(points, -1)
        transposed = tf.transpose(pointsExtended, perm=[0, 3, 2, 1])
        absDiff = tf.abs(pointsExtended - transposed)
        torDiff = tf.where(tf.greater(absDiff, 0.5), 1 - absDiff, absDiff)
        squaredDst = tf.reduce_sum(tf.square(torDiff), [2])

        # gradient of sqrt(0) is infinite => clamp
        clipped = tf.maximum(squaredDst, SQRT_EPS)
        dst = tf.sqrt(clipped)

        # replace diagonal with large number
        if identityLarge:
            dst += tf.expand_dims(
                tf.diag(np.float32(np.full(points.shape[1], LARGE_NUMBER))),
                0)
        return dst 