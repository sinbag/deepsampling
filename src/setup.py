import sys, os
import re
import numpy as np
import tensorflow as tf
import sampler

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
sys.path.append(os.path.join(BASE_DIR, 'loss'))

import fourier
import pcf
from mathutils import *
import ioutils
import projection as proj

#=======================================================

class Environment:

    def __init__(
        self,
        trainingSetup,
        fourierSetupList = None,
        histogramSetupList = None):

        self.trainingSetup = trainingSetup
        self.histogramSetupList = histogramSetupList
        self.fourierSetupList = fourierSetupList

#=======================================================

class TrainingSetup:

    def __init__(
        self,
        pointCount,
        dimCount,
        batchSize,
        griddingDims,
        convCount,
        kernelSampleCount,
        receptiveField,
        projectionsStrings,
        customOp,
        trainIterations,
        learningRate,
        displayGrid,
        evalRealizations,
        saveEvalRealizations,
        storeNetwork,
        backupInterval,
        weightDir = None,
        kernelCount=None):

        # input
        self.pointCount = pointCount
        self.dimCount = dimCount
        self.batchSize = batchSize
        self.griddingDims = griddingDims

        # architecture
        self.convCount = convCount
        self.kernelCount = kernelCount if kernelCount is not None else convCount
        assert self.kernelCount <= convCount, "kernelCount must not be larger than convCount."
        self.kernelSampleCount = kernelSampleCount
        self.receptiveField = receptiveField

        assert len(projectionsStrings) > 0, "projectionsStrings must contain at least one element."
        assert len(projectionsStrings) == len(set(projectionsStrings)), "projectionStrings must not contain duplicate elements."
        self.projectionsStrings = ['--' + ps +  '--' for ps in projectionsStrings]
        self.projections = proj.parseProjections(projectionsStrings)
        self.projectionCount = len(self.projections)
        self.projectionDimensionCount = self.countProjectionDimensions()
        self.customOp = customOp

        # training
        self.trainIterations = trainIterations
        self.learningRate = learningRate

        # evaluation
        self.displayGrid = displayGrid
        self.evalRealizations = evalRealizations
        self.saveEvalRealizations = saveEvalRealizations

        assert self.evalRealizations >= self.batchSize, "evalRealizations must be at least as large as batchSize."

        # IO
        self.storeNetwork = storeNetwork
        self.backupInterval = backupInterval
        self.weightDir = weightDir
        self.kernelWeights = []

        if weightDir is not None:
            self.loadWeights()

      #---------------------------------------------------

    def loadWeights(self):

        #https://stackoverflow.com/questions/12093940/reading-files-in-a-particular-order-in-python
        FILENUMBERS = re.compile(r'(\d+)')
        def numericalSort(value):
            parts = FILENUMBERS.split(value)
            parts[1::2] = map(int, parts[1::2])
            return parts

        def load(target, targetCount, label):
            varCounter = 0
            for file in sorted(os.listdir(self.weightDir),key=numericalSort):
                if file.startswith(label) and file.endswith(".txt"):
                    varCounter += 1
                    w = np.loadtxt(os.path.join(self.weightDir, file)).astype(np.float32)
                    w = np.reshape(w, (1,)) if w.shape is () else w
                    target.append(w)
            assert varCounter != 0, "Could not load weights."
            assert targetCount == varCounter, "Loaded weights do not match network."

        load(self.kernelWeights, self.kernelCount * self.projectionCount, "kernel")
        print("====Kernel weights successfully loaded")

    #---------------------------------------------------

    # count how often dimensions are used in projections
    def countProjectionDimensions(self):
        p = np.asarray([item for sublist in self.projections for item in sublist])
        dimProjCount = [(p == i).sum() for i in range(self.dimCount)]
        assert dimProjCount.count(0) == 0, "Not all dimensions are used in projections."
        return dimProjCount

#=======================================================

# super class for all losses
class LossSetup:

    def __init__(self):
        self.target = None
        self.batchCounter = 0

    #---------------------------------------------------

    def resetBatchCounter(self):
        self.batchCounter = 0

#=======================================================

class HistogramSetup(LossSetup):

    def __init__(
        self,
        trainingSetup,
        binCount,
        stdDev,
        scale,
        mcSamples = 512):
        self.binCount = binCount
        self.scale = scale
        self.stdDev = stdDev
        self.normalization = None
        super().__init__()

        self.createNormalization(trainingSetup, mcSamples, trainingSetup.batchSize)

    #---------------------------------------------------

    # estimate a differential histogram using an arbitrary generator
    def estimateHistogram(self, generator, trainingSetup, mcSamples, batchSize, normalize):

        inputPoints = tf.placeholder(
            tf.float32,
            shape=(
                batchSize,
                trainingSetup.pointCount,
                trainingSetup.dimCount))

        hNode = pcf.differentialHistogram(inputPoints, self, normalize)
        iterations = int(mcSamples / batchSize)
        assert iterations > 0, "Number of mcSamples must be a positive multiple of batchSize."
        sess = tf.Session()
        hAcc = np.zeros((self.binCount,), np.float32)
        for i in range(iterations):
            batch = generator(
                batchSize,
                trainingSetup.pointCount,
                trainingSetup.dimCount)
            hAcc += sess.run(hNode, {inputPoints:batch})

        sess.close()
        self.resetBatchCounter()
        return hAcc / iterations

    #---------------------------------------------------

    # MC estimate of histogram normalization
    def createNormalization(self, trainingSetup, mcSamples, batchSize):
        print("====Producing histogram normalization...")
        self.normalization = self.estimateHistogram(sampler.random,
                                                    trainingSetup,
                                                    mcSamples,
                                                    batchSize,
                                                    normalize=False)
        # be sure to have no zeros
        self.normalization += EPS
        return self.normalization

    #---------------------------------------------------

    # create target histogram with an explicit generator function
    def createTarget(self, trainingSetup, generator, mcSamples, batchSize):
        print("====Producing target histogram...")
        self.target = self.estimateHistogram(generator,
                                            trainingSetup,
                                            mcSamples,
                                            batchSize,
                                            normalize=True)
        return self.target

    #---------------------------------------------------

    # create target histogram using precomputed realizations
    def createTargetFromData(self, trainingSetup, path, mcSamples, batchSize):
        print("====Producing target histogram from saved realizations...")
        p = ioutils.loadSamplePatternsMultiFile(path, trainingSetup.pointCount)
        dRealizationCount, dPointCount, dDimCount = p.shape
        assert dPointCount == trainingSetup.pointCount, "Point counts don't match."
        assert mcSamples <= dRealizationCount, "Cannot produce requested number of MC samples, number of realizations too small."

        def dataSlicer(batchSize, pointCount, dimCount):
            slice = p[self.batchCounter:self.batchCounter+batchSize, ...]
            self.batchCounter += batchSize
            return slice

        self.target = self.estimateHistogram(dataSlicer,
                                            trainingSetup,
                                            mcSamples,
                                            batchSize,
                                            normalize=True,
                                            validDimCount=dDimCount)
        return self.target

#=======================================================

class FourierSetup(LossSetup):

    def __init__(
        self,
        resolution,
        cancelDC,
        freqStep=1,
        mcSamplesPerShell=32):

        self.resolution = resolution
        self.freqStep = freqStep
        self.cancelDC = cancelDC
        self.mcSamplesPerShell = mcSamplesPerShell
        super().__init__()

    #---------------------------------------------------

    # load 2D spectrum from EXR file
    def loadTarget2D(self, filename, factor=1):
        print('======== target 2D: ', filename)
        def cropSpectrum2D(spectrum):
            centerPx = int(spectrum.shape[0] / 2.)
            halfSize = int(self.resolution / 2.)
            box = (centerPx - halfSize, centerPx + halfSize)
            return spectrum[box[0]:box[1], box[0]:box[1]]

        target = ioutils.loadExr(filename, numChannels=1)
        dataRes = target.shape[0]
        assert dataRes >= self.resolution, "Target spectrum is smaller than requested resolution"
        if dataRes > self.resolution:
            print("Target spectrum too large ({0}x{0}), will be cropped to {1}x{1}.".format(dataRes, self.resolution))
            target = cropSpectrum2D(target)
        if self.cancelDC:
            dcPos = int(self.resolution / 2.)
            target[dcPos, dcPos] = 0
        self.target = target * factor
        return self.target

    #---------------------------------------------------

    # radial projection of 2D spectra
    def radiallyProjectTarget(self):
        assert len(self.target.shape) == 2, "Can only radially project 2D spectra."
        inputSpectrumNode = tf.placeholder(tf.float32, shape=(self.resolution, self.resolution))
        radialAvgNode = proj.radialAverage2D(inputSpectrumNode)
        sess = tf.Session()
        self.target = sess.run(radialAvgNode, {inputSpectrumNode: self.target})
        sess.close()
        self.resolution = int(self.resolution / 2)
        if self.cancelDC:
            self.target[0] = 0
        return self.target

    #---------------------------------------------------

    # load 1D spectrum from TXT file
    def loadTarget1D(self, filename, factor=1):
        print('======== target 1D: ', filename)
        def cropSpectrum1D(spectrum):
            return spectrum[0:self.resolution]

        target = np.loadtxt(filename).astype(np.float32)[::, 1]
        dataRes = target.shape[0]
        assert dataRes >= self.resolution, "Target spectrum is smaller than requested resolution"
        if dataRes > self.resolution:
            print("Target spectrum too large ({0}), will be cropped to {1}.".format(dataRes, self.resolution))
            target = cropSpectrum1D(target)
        if self.cancelDC:
            target[0] = 0
        self.target = target * factor
        return self.target

    #---------------------------------------------------

    # MC estimate the radial average of a spectrum using an arbitrary generator
    def estimateSpectrum(self, generator, trainingSetup, realizationCount):

        inputNode = tf.placeholder(
            tf.float32,
            shape=(
                trainingSetup.batchSize,
                trainingSetup.pointCount,
                trainingSetup.dimCount))

        spectrumNode = fourier.radialSpectrumMC(inputNode, self)
        iterations = int(realizationCount / trainingSetup.batchSize)
        assert iterations > 0, "Number of realizationCount must be a positive multiple of batchSize."
        sess = tf.Session()
        spectrumAcc = np.zeros((self.resolution, ), np.float32)

        for i in range(iterations):
            batch = generator(
                trainingSetup.batchSize,
                trainingSetup.pointCount,
                trainingSetup.dimCount)
            spectrumAcc += sess.run(spectrumNode, {inputNode:batch})

        sess.close()
        self.resetBatchCounter()
        return spectrumAcc / iterations

    #---------------------------------------------------

    # create target spectrum with an explicit generator function
    def createTargetFromGenerator(self, trainingSetup, generator, realizationCount):
        print("====Producing target spectrum...")
        trainingSetup.needMCSamples = True
        self.target = self.estimateSpectrum(generator, trainingSetup, realizationCount)
        return self.target

    #---------------------------------------------------

    # create target spectrum using precomputed realizations
    def createTargetFromData(self, trainingSetup, path, realizationCount):
        print("====Producing target spectrum from saved realizations...")
        trainingSetup.needMCSamples = True
        p = ioutils.loadSamplePatternsMultiFile(path, trainingSetup.pointCount)
        dRealizationCount, dPointCount, dDimCount = p.shape
        assert dPointCount == trainingSetup.pointCount, "Point counts don't match."

        def dataSlicer(batchSize, pointCount, dimCount):
            slice = p[self.batchCounter:self.batchCounter+batchSize, ...]
            self.batchCounter += batchSize
            return slice

        self.target = self.estimateSpectrum(dataSlicer, trainingSetup, realizationCount)
        return self.target
