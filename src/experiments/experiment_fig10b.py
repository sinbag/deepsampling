import fourier
import pcf
import ioutils as io
from mathutils import *
import setup

#============================================================================

LOGS_DIR = "../fig10b-bnot-step-jitter/"
TARGET_DIR = "../targets/"
FILE_EXT = ".pdf"

#============================================================================

def buildEnvironment():
    print("Building environment...")

    trainingSetup = setup.TrainingSetup(

        # input
        pointCount = 1024,
        dimCount = 3,
        batchSize = 2,
        griddingDims = 0,

        # architecture
        convCount = 60,
        kernelCount = 30,
        kernelSampleCount = 128,
        receptiveField = 0.5,
        projectionsStrings = [ '01', '12', '02' ],
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
        resolution=64,
        cancelDC=True,
        mcSamplesPerShell=48)
    fourierSetup0.loadTarget1D(io.joinPath(TARGET_DIR, "spectra/bnot-powspec-radialmean-d2-n1024.txt"))
    fourierSetupList.append(fourierSetup0)

    fourierSetup1 = setup.FourierSetup(
        resolution=48,
        cancelDC=True,
        mcSamplesPerShell=48)
    fourierSetup1.loadTarget1D(io.joinPath(TARGET_DIR, "spectra/step-powspec-radialmean-d2-n1024.txt"))
    fourierSetupList.append(fourierSetup1)

    fourierSetup2 = setup.FourierSetup(
        resolution=64,
        cancelDC=True,
        mcSamplesPerShell=48)
    fourierSetup2.loadTarget1D(io.joinPath(TARGET_DIR, "spectra/jitter-powspec-radialmean-d2-n1024.txt"))
    fourierSetupList.append(fourierSetup2)

    return setup.Environment(trainingSetup, fourierSetupList, histogramSetupList)

#============================================================================

def lossSetup(env, outputNode):

    histogramNode = None
    outputSpectrumNodes = []
    lossNode=None
    projs = env.trainingSetup.projections
    print(projs, len(env.fourierSetupList))

    if len(env.fourierSetupList) > 0:
        print("======== setting the Fourier lossSetup for projections")

    assert len(env.fourierSetupList) == len(projs), "lossSetup() Not enough fourierSetups provided, {0} required.".format(len(projs))

    outputSpectrumNodes = []
    for k in range(len(projs)):
        if len(projs[k]) == 2: #2D Projections
            print('Fourier lossStep 2D radialSpectrumMC: ', projs[k])
            ptNode = tf.transpose([outputNode[:,:,projs[k][0]],outputNode[:,:,projs[k][1]]],perm=[1,2,0])
            outputSpectrumNodes.append(fourier.radialSpectrumMC(ptNode, env.fourierSetupList[k]))
        else:
            print('Fourier lossStep {0}D radialSpectrumMC'.format(env.trainingSetup.dimCount))
            outputSpectrumNodes.append(fourier.radialSpectrumMC(outputNode, env.fourierSetupList[k]))

        loss = l1Loss(outputSpectrumNodes[k], env.fourierSetupList[k].target)
        lossNode = lossNode + loss if lossNode is not None else loss

    return lossNode, outputSpectrumNodes, histogramNode
