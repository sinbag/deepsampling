import fourier
import pcf
import ioutils as io
from mathutils import *
import setup

#============================================================================

LOGS_DIR = "/HPS/Sinbag/archive00/2019-NeuralSampling/webpage-content/fig8b-jitter/"
TARGET_DIR = "../targets/"
FILE_EXT = ".pdf"

#============================================================================

def buildEnvironment():
    print("Building environment...")

    trainingSetup = setup.TrainingSetup(

        # input
        pointCount = 1024,
        dimCount = 2,
        batchSize = 2,
        griddingDims = 0,

        # architecture
        convCount = 60,
        kernelCount = 20,
        kernelSampleCount = 64,
        receptiveField = 0.5,
        projectionsStrings = [ '01' ],
        customOp = True,

        # training
        trainIterations = 100000,
        learningRate = 10e-7,

        # evaluation
        displayGrid = False,
        evalRealizations = 1000,
        saveEvalRealizations = True,

        # IO
        storeNetwork = True,
        backupInterval = 10000,
        weightDir = None
    )

    histogramSetupList = []
    fourierSetupList = []

    fourierSetup0 = setup.FourierSetup(
        resolution=64,
        cancelDC=True,
        mcSamplesPerShell=48)
    fourierSetup0.loadTarget1D(io.joinPath(TARGET_DIR, "spectra/jitter-powspec-radialmean-d2-n1024.txt"))
    fourierSetupList.append(fourierSetup0)

    return setup.Environment(trainingSetup, fourierSetupList, histogramSetupList)

#============================================================================

def lossSetup(env, outputNode):

    histogramNode = None

    spectrumNode = fourier.radialSpectrumMC(outputNode, env.fourierSetupList[0])
    lossNode = l1Loss(spectrumNode, env.fourierSetupList[0].target)

    return lossNode, spectrumNode, histogramNode
