import platform
import os, sys

if platform.system() == 'Linux':
    # To find available GPU on a multi-gpu machine cluster
    import utils.selectgpu as setgpu
    os.environ["CUDA_VISIBLE_DEVICES"] = str(setgpu.pick_gpu_lowest_memory())

import argparse
import importlib

import numpy as np
import math
import time

import tensorflow as tf
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, 'loss'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
sys.path.append(os.path.join(BASE_DIR, 'experiments'))

import model
import fourier
import pcf
import plotutils as plot
import ioutils as io
from mathutils import *
from telemetryutils import *
import setup
import sampler
import ioutils
import projection
import evaluation

#============================================================================

def train(env, experiment):

    experimentID = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    logPath = io.joinPath(experiment.LOGS_DIR, experimentID)

    tf.reset_default_graph()
    sess = tf.Session()

    # create input placeholder
    inputNode = tf.placeholder(
        tf.float32,
        shape=(
            env.trainingSetup.batchSize,
            env.trainingSetup.pointCount,
            env.trainingSetup.dimCount),
        name="inputNode")

    # create network
    outputNode = model.createNetwork(env.trainingSetup, inputNode)

    # create loss(es)
    lossNode, spectrumNode, histogramNode = experiment.lossSetup(env, outputNode)

    #-------------------------------------------------

    # create optimizer
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(env.trainingSetup.learningRate, global_step, 200, 0.99, staircase=True)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    trainStep = optimizer.minimize(lossNode, global_step=global_step)

    # initialize variables
    sess.run(tf.global_variables_initializer())

    # create telemetry
    writer = tf.summary.FileWriter(logdir=logPath)
    writer.add_graph(sess.graph)
    writer.flush()

    #--------------------------
    def trainTimed(feedDict):
        t0 = time.time()
        _, loss = sess.run((trainStep, lossNode), feedDict)
        t = round(time.time() - t0, 3)
        return loss, t
    #--------------------------

    # training loop
    if env.trainingSetup.trainIterations > 0:
        print("==== Start training...")
        for i in range(env.trainingSetup.trainIterations):

            trainPoints = sampler.griddedRandom(
                env.trainingSetup.batchSize,
                env.trainingSetup.pointCount,
                env.trainingSetup.dimCount,
                env.trainingSetup.griddingDims)

            loss, t = trainTimed({inputNode:trainPoints})

            # monitor
            outputStats(writer, i, loss)
            if i % 10 == 0:
                writer.flush()

                sys.stdout.write('iter ' + str(i) + ' | loss ' + str(round(loss, 3)) + ' | time ' + str(t) + '\r')
                sys.stdout.flush()
                        
            if i != 0 and env.trainingSetup.storeNetwork and i % env.trainingSetup.backupInterval == 0:
                evaluation.simpleCheckpoint(env, sess, logPath, i)

    print("")

    writer.flush()
    writer.close()

    #-------------------------------------------------

    # store trained network
    if env.trainingSetup.storeNetwork:
        evaluation.simpleCheckpoint(env, sess, logPath)

    #-------------------------------------------------

    # evaluation
    print("==== Evaluating...")

    testPoints = sampler.griddedRandom(
        env.trainingSetup.batchSize,
        env.trainingSetup.pointCount,
        env.trainingSetup.dimCount,
        env.trainingSetup.griddingDims)

    #-------------------------------------------------

    # output points visualization
    outPoints = sess.run(outputNode, {inputNode:testPoints})

    # scatter plots of points
    if env.trainingSetup.griddingDims == 0:
        grid = 1/math.sqrt(env.trainingSetup.pointCount) if env.trainingSetup.displayGrid else None
        plot.multiPointPlot(
            np.stack((testPoints[0], outPoints[0]), 0),
            ("input", "output"),
            grid=grid,
            filename = ioutils.joinPath(logPath, "points" + experiment.FILE_EXT))
    
    # dither masks (when using gridding)
    else:
        if env.trainingSetup.dimCount - env.trainingSetup.griddingDims <= 3:
            textures = plot.convertGriddedToArray(outPoints, env.trainingSetup.griddingDims)
            # 2D textures
            if env.trainingSetup.griddingDims == 2:
                for b in range(env.trainingSetup.batchSize):
                    filename = ioutils.joinPath(logPath, "dithermask_" + str(b) + ".exr")
                    ioutils.saveExr(textures[b], filename)
            # 3D textures (as 2D slices)
            elif env.trainingSetup.griddingDims == 3:
                for b in range(env.trainingSetup.batchSize):
                    for s in range(textures.shape[1]):
                        filename = ioutils.joinPath(logPath, "dithermask_b" + str(b) + "_s" + str(s) + ".exr")
                        ioutils.saveExr(textures[b, s, ...], filename)
            else:
                print("Could not save dither masks: gridding dimension > 3")
        else:
            print("Could not save dither masks: value dimensions > 3")

    #-------------------------------------------------

    # spectrum figures
    if spectrumNode is not None:

        #--------------------------
        def spectrumOutput(spectrumNode, spectrumTarget, path):
            expectedSpectrum = evaluation.produceExpectedOutput(
                env,
                sess,
                spectrumTarget.shape,
                inputNode,
                spectrumNode)
            if len(expectedSpectrum.shape) == 1:
                plot.multiLinePlot((spectrumTarget, expectedSpectrum),
                    title="1d spectra", legend=("target", "result"), filename=path)
            else:
                io.saveExr(expectedSpectrum, filename=path)
        #--------------------------

        spectrumNode = [spectrumNode] if not isinstance(spectrumNode, list) else spectrumNode
        for i, s in enumerate(spectrumNode):
            spectrumPath = io.joinPath(logPath, "spectra_" + str(i) + experiment.FILE_EXT)
            spectrumOutput(s, env.fourierSetupList[i].target, spectrumPath)

    #-------------------------------------------------

    # histogram figures
    if histogramNode is not None:

        #--------------------------
        def histogramOutput(histogramNode, histogramTarget, path):
            expectedHistogram = evaluation.produceExpectedOutput(
                env,
                sess,
                histogramTarget.shape,
                inputNode,
                histogramNode)
            plot.multiLinePlot((histogramTarget, expectedHistogram),
                title="histograms", legend=("target", "result"), filename=path)
        #--------------------------

        histogramNode = [histogramNode] if not isinstance(histogramNode, list) else histogramNode
        for i, h in enumerate(histogramNode):
            histogramPath = io.joinPath(logPath, "histogram" + str(i) + experiment.FILE_EXT)
            histogramOutput(h, env.histogramSetupList[i].target, histogramPath)

    #-------------------------------------------------

    # visualize trained variables
    if env.trainingSetup.storeNetwork:
        print("==== Extracting trained variables...")
        kernelWeights = evaluation.extractModelWeights(sess, env.trainingSetup)

        # plot kernels for each projection in different figure
        for i in range(env.trainingSetup.projectionCount):
            
            # line plots
            plot.multiLinePlot(
                kernelWeights[i:len(kernelWeights)+1:env.trainingSetup.projectionCount],
                title="kernelWeights" + env.trainingSetup.projectionsStrings[i],
                legend=None,
                filename=ioutils.joinPath(logPath, "kernelVars_" + str(i) + experiment.FILE_EXT))

            # surface plots
            if env.trainingSetup.kernelCount > 1:
                x = np.arange(env.trainingSetup.kernelSampleCount)
                y = np.arange(env.trainingSetup.kernelCount)
                x, y = np.meshgrid(x, y)
                z = np.stack(kernelWeights[i:len(kernelWeights)+1:env.trainingSetup.projectionCount])
                plot.surfacePlot(
                    [x, y, z], 
                    title="kernelWeights" + env.trainingSetup.projectionsStrings[i],
                    filename=ioutils.joinPath(logPath, "kernelVars3D_" + str(i) + experiment.FILE_EXT))
    
    #-------------------------------------------------

    # save realizations
    if env.trainingSetup.saveEvalRealizations:

        realizationPath = ioutils.joinPath(logPath, "realizations/")
        io.makeDir(realizationPath)
        evaluation.saveRealizations(
            env, 
            sess, 
            inputNode, 
            outputNode,
            env.trainingSetup.evalRealizations,
            realizationPath)

    sess.close()


##============================================================================

def main():

    # import experiment
    parser = argparse.ArgumentParser()
    parser.add_argument('-e','--experiment', help='experiment to run from /experiments', required=True)
    args = parser.parse_args()
    print("==== Import", args.experiment, "...")
    experiment = importlib.import_module(args.experiment)

    # setup
    env = experiment.buildEnvironment()

    # train
    train(env, experiment)

#============================================================================

if __name__ == '__main__':
    main()
    print("========= TERMINATED =========")
    
