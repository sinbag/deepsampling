import fourier
import pcf
import ioutils as io
from mathutils import *
import setup

#============================================================================

LOGS_DIR = "/HPS/Sinbag/archive00/2019-NeuralSampling/webpage-content/fig8c-step/"
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
        backupInterval = 5000,
        weightDir = None
    )

    histogramSetupList = []
    fourierSetupList = []

    fourierSetup0 = setup.FourierSetup(
<<<<<<< HEAD:src/experiments/experiment_singleSpectrum_fig8c.py
        resolution=48,
=======
        resolution=64,
>>>>>>> de1c056b2d8aca3940e48acac012b9c70af093ed:src/experiments/experiment_fig8c.py
        cancelDC=True,
        mcSamplesPerShell=48)
    fourierSetup0.loadTarget1D(io.joinPath(TARGET_DIR, "spectra/step-powspec-radialmean-d2-n1024.txt"))
    fourierSetupList.append(fourierSetup0)

    return setup.Environment(trainingSetup, fourierSetupList, histogramSetupList)

#============================================================================

def lossSetup(env, outputNode):

    histogramNode = None

    spectrumNode = fourier.radialSpectrumMC(outputNode, env.fourierSetupList[0])
    lossNode = l1Loss(spectrumNode, env.fourierSetupList[0].target)

    return lossNode, spectrumNode, histogramNode