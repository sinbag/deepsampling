import os, sys
import numpy as np
import tensorflow as tf
import platform
from mathutils import *
import projection
import sampler

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '../ops/src/filter'))
from dist_filter_gpu import *

#================================================================

# create and initialize filter kernel variables
def createKernelValues(trainingSetup, convIter, projIter):
    sampleCount = trainingSetup.kernelSampleCount
    projStrings = trainingSetup.projectionsStrings
    kernelId = str(convIter) + projStrings[projIter]
    projCount = trainingSetup.projectionCount

    with tf.variable_scope("kernelVars" + kernelId, reuse=tf.AUTO_REUSE) as scope:
        if trainingSetup.kernelWeights:
            init = trainingSetup.kernelWeights[convIter * projCount + projIter]
        else:
            init = np.zeros(sampleCount, np.float32)
        varName = "kernelValues" + kernelId
        kernelValues = tf.get_variable(varName, initializer=init)
        return kernelValues

#================================================================

# map distance to convolution weight
# using a simple trainable lookup table with linear interpolation
# [used by TF filter]
def evalKernel(dst, kernelValues, trainingSetup):

    sampleCount = trainingSetup.kernelSampleCount
    receptiveField = trainingSetup.receptiveField

    # map distance to kernel bins
    sampleDst = tf.minimum(sampleCount-1.0, dst * (sampleCount-1) / receptiveField)
    lowerIndex = tf.floor(sampleDst)
    lerpWeight = sampleDst - lowerIndex
    lowerIndex = tf.cast(lowerIndex, tf.int32)
    upperIndex = tf.minimum(lowerIndex+1, sampleCount-1)
    values = tf.gather(kernelValues, [lowerIndex, upperIndex])
    interp = lerp(values[0], values[1], lerpWeight)

    # zero out weights if distance is zero or larger than receptiveField
    return tf.where(
        tf.logical_or(
            tf.greater(dst, receptiveField),
            tf.less(dst, DST_EPS)),
        tf.zeros_like(interp),
        interp)

#================================================================

# toroidal filter
# [used by TF filter]
def toroidalFilter(points, kernelMatrix, namespace="filter"):
    with tf.name_scope(namespace):
        batchCount = points.shape[0]
        pointCount = points.shape[1]
        pointsBroadcast = tf.tile(
            tf.expand_dims(points, -1),
            [1, 1, 1, pointCount])
        pointsBroadcastTransposed = tf.transpose(pointsBroadcast, perm=[0, 3, 2, 1])
        diff = tf.abs(pointsBroadcast - pointsBroadcastTransposed)
        torPoints = tf.where(tf.greater(diff, 0.5),
                             tf.where(tf.less(pointsBroadcastTransposed, 0.5),
                                      pointsBroadcast - 1, pointsBroadcast + 1),
                             pointsBroadcast)
        torDiff = tf.nn.l2_normalize(   torPoints - pointsBroadcastTransposed,
                                        axis=2,
                                        epsilon=SQRT_EPS)
        kernelMatrixExtended = tf.expand_dims(kernelMatrix, 2)

        convT = kernelMatrixExtended * torDiff
        return tf.transpose(tf.reduce_sum(convT, [1]), [0, 2, 1])

#================================================================

# linear combination of projections with trainable weights
def combineProjections(projections, trainingSetup, name=""):
    with tf.name_scope("combProj"):
        dimCount = trainingSetup.dimCount
        projections /= np.reshape(trainingSetup.projectionDimensionCount, [1, 1, dimCount, 1])
        return tf.reduce_sum(projections, axis=-1)

#================================================================

# points leaving unit hypercube re-appear on the other side
def toroidalWrapAround(points):
    with tf.name_scope("toroidWrap"):
        points = tf.where(tf.greater(points, 1), points - tf.floor(points), points)
        return tf.where(tf.less(points, 0), points + tf.ceil(tf.abs(points)), points)

#================================================================

def enforceGridding(points, origPoints, trainingSetup):
    with tf.name_scope("gridding"):
        return tf.concat(
            [origPoints[..., 0:trainingSetup.griddingDims],
            points[..., trainingSetup.griddingDims:trainingSetup.dimCount]],
            axis=-1)

#================================================================

# build filter cascade
def createNetwork(trainingSetup, inputPoints):

    print("==== Creating network...")

    points = inputPoints

    kernelWeights = []

    for i in range(trainingSetup.kernelCount):
        for j in range(trainingSetup.projectionCount):
            kernelWeights.append(createKernelValues(trainingSetup, i, j))

    # iterate over filters
    for i in range(trainingSetup.convCount):
        with tf.name_scope("convBlock_" + str(i)):

            projIteration = 0
            projOutputsList = []

            # iterate over projections
            for projString, projAxes in zip(trainingSetup.projectionsStrings, trainingSetup.projections):

                projPoints = points

                # project points
                needsProj = len(projAxes) < trainingSetup.dimCount
                if needsProj:
                    projPoints = projection.axisAlignedPointProjection(points, projAxes, "aaProj" + projString)

                if trainingSetup.convCount == trainingSetup.kernelCount:
                    weights = kernelWeights[i * trainingSetup.projectionCount + projIteration]
                else:
                    # less kernels than filters: interpolate trainable kernels
                    sweep = i / (trainingSetup.convCount-1) * (trainingSetup.kernelCount-1)
                    lowIndex = int(sweep)
                    upIndex = min(lowIndex + 1, trainingSetup.kernelCount-1)
                    frac = sweep - lowIndex
                    weights1 = kernelWeights[lowIndex * trainingSetup.projectionCount + projIteration]
                    weights2 = kernelWeights[upIndex * trainingSetup.projectionCount + projIteration]
                    weights = lerp(weights1, weights2, frac)

                # custom op
                if trainingSetup.customOp:
                    projPoints = distFilterGPU(
                        input=projPoints,
                        weights=weights,
                        receptive_field=trainingSetup.receptiveField,
                        dst_eps=DST_EPS,
                        name="UnstructuredDstFilter")

                # TF model (requires m^2 memory)
                else:
                    # compute distances in projected space
                    distances = mutualToroidalDistances(projPoints, namespace="toroidDst" + projString)

                    # evaluate kernel
                    with tf.name_scope("kernelEval" + projString):
                        convWeights = evalKernel(distances, weights, trainingSetup)

                    # perform filtering
                    projPoints += toroidalFilter(projPoints, convWeights, namespace="conv" + projString)

                # reproject to full space (filling in zeros)
                if needsProj:
                    projPoints = projection.axisAlignedPointReProjection(
                        projPoints,
                        projAxes,
                        points.shape,
                        "aaReProj" + projString)

                projOutputsList.append(projPoints)
                projIteration += 1

            # combine re-projected outputs
            if len(trainingSetup.projections) == 1:
               points = projOutputsList[0]
            else:
                projOutputs = tf.stack(projOutputsList, axis=-1)
                points = combineProjections(projOutputs, trainingSetup)

            # snap back to original grid
            if trainingSetup.griddingDims > 0:
                points = enforceGridding(points, inputPoints, trainingSetup)

            # toroidal wrap
            points = toroidalWrapAround(points)

    return points