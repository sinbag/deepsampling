import os, sys
import numpy as np
import tensorflow as tf
import setup
import sampler
import model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, 'utils'))

import ioutils

#============================================================================

def produceExpectedOutput(env, session, outputShape, inputNode, outputNode):
    acc = np.zeros(outputShape, np.float32)
    itCount = int(env.trainingSetup.evalRealizations / env.trainingSetup.batchSize)
    for i in range(itCount):
        points = sampler.griddedRandom(
            env.trainingSetup.batchSize,
            env.trainingSetup.pointCount,
            env.trainingSetup.dimCount,
            env.trainingSetup.griddingDims)
        acc += session.run(outputNode, {inputNode:points})
    return acc / itCount

#============================================================================

def saveRealizations(env, session, inputNode, outputNode, realizationCount, outFolder, header=''):
    print("==== Creating and saving realizations...")
    executionCount = int(realizationCount / env.trainingSetup.batchSize)
    assert executionCount == realizationCount / env.trainingSetup.batchSize, "realizationCount must be multiple of batchSize"
    fileIndex = 0
    for i in range(executionCount):
        testInput = sampler.griddedRandom(
            env.trainingSetup.batchSize,
            env.trainingSetup.pointCount,
            env.trainingSetup.dimCount,
            env.trainingSetup.griddingDims)
        testOutput = session.run(outputNode, {inputNode:testInput})
        for b in range(env.trainingSetup.batchSize):
            sys.stdout.write("producing realization: %d  %d\r" % (i, b))
            sys.stdout.flush()
            fileName = outFolder + "rez"+'-d'+str(env.trainingSetup.dimCount)+'-n'+str(env.trainingSetup.pointCount)+'-'+str(fileIndex).zfill(5) + ".txt"
            ioutils.saveSamplePattern(fileName, testOutput[b],header=header)
            fileIndex += 1

#============================================================================

# fetch trainable model weights
def extractModelWeights(session, trainingSetup):
    variableNames = [v.name for v in tf.trainable_variables()]
    variables = session.run(variableNames)
    return variables

#============================================================================

def simpleCheckpoint(env, session, path, iteration=None):

    if not os.path.exists(path):
        os.makedirs(path)

    # store environment
    networkFile = ioutils.joinPath(path, 'env.txt')
    if not os.path.isfile(networkFile):
        f = open(networkFile, "w")
        f.write("TrainingSetup\n")
        f.write("pointCount " + str(env.trainingSetup.pointCount) + "\n")
        f.write("dimCount " + str(env.trainingSetup.dimCount) + "\n")
        f.write("batchSize " + str(env.trainingSetup.batchSize) + "\n")
        f.write("griddingDims " + str(env.trainingSetup.griddingDims) + "\n")
        f.write("convCount " + str(env.trainingSetup.convCount) + "\n")
        f.write("kernelCount " + str(env.trainingSetup.kernelCount) + "\n")
        f.write("kernelSampleCount " + str(env.trainingSetup.kernelSampleCount) + "\n")
        f.write("receptiveField " + str(env.trainingSetup.receptiveField) + "\n")
        f.write("projections " + str(env.trainingSetup.projectionsStrings) + "\n")
        f.write("trainIterations " + str(env.trainingSetup.trainIterations) + "\n")
        f.write("learningRate " + str(env.trainingSetup.learningRate) + "\n")

        if env.fourierSetupList:
            for fs in env.fourierSetupList:
                f.write("\nFourierSetup\n")
                f.write("resolution " + str(fs.resolution) + "\n")
                f.write("cancelDC " + str(fs.cancelDC) + "\n")
                f.write("freqStep " + str(fs.freqStep) + "\n")

        if env.histogramSetupList:
            for dh in env.histogramSetupList:
                f.write("\nHistogramSetup\n")
                f.write("binCount " + str(dh.binCount) + "\n")
                f.write("stdDev " + str(dh.stdDev) + "\n")
                f.write("scale " + str(dh.scale) + "\n")

        f.close()

    # store network weights
    kernelWeights = extractModelWeights(session, env.trainingSetup)

    def writeVars(vars, name):
        for i, var in enumerate(vars):
            pathname = path
            if iteration is not None:
                pathname = ioutils.joinPath(path, "i" + str(iteration) + "/")
                if not os.path.exists(pathname):
                    os.makedirs(pathname)
            filename = name + str(i) + ".txt"
            np.savetxt(ioutils.joinPath(pathname, filename), var)

    writeVars(kernelWeights, "kernel")