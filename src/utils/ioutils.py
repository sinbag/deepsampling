from os import listdir, mkdir
from os.path import isfile, isdir, join
import numpy as np
import Imath, OpenEXR
from imageio import imread, imwrite

#======================================================

def loadSamplePatternsSingleFile(filename, pointCount, skiprows=0):
    p = np.loadtxt(fname=filename, skiprows=skiprows)
    assert p is not None, "Could not load sample pattern from file"
    batchSize = int(p.shape[0] / pointCount)
    assert batchSize == p.shape[0] / pointCount, "Sample pattern does not support given pointCount"
    if len(p.shape) ==1:
        dimCount = 1
    else:
        dimCount = p.shape[1]
    p = p.reshape(batchSize, pointCount, dimCount)
    return p.astype(np.float32)

#======================================================

def loadSamplePatternsMultiFile(foldername, pointCount, skiprows=0):
    files = [join(foldername, f) for f in listdir(foldername) if not f.startswith('.DS_Store') if isfile(join(foldername, f))]
    p = None
    for f in files:
        pf = loadSamplePatternsSingleFile(f, pointCount, skiprows)
        p = pf if p is None else np.append(p, pf, axis=0)
    return p

#======================================================

def loadSamplePatterns(path, pointCount, skiprows=0):
    if isfile(path):
        p = loadSamplePatternsSingleFile(path, pointCount, skiprows)
    elif isdir(path):
        p = loadSamplePatternsMultiFile(path, pointCount, skiprows)
    assert p is not None, "Could not load point sets."
    return p

#======================================================

def saveSamplePattern(filename, pattern, header=None):
    np.savetxt(filename, pattern, header=header,comments="")

#======================================================

def loadExr(filename, numChannels=3):
    pxType = Imath.PixelType(Imath.PixelType.FLOAT)
    file = OpenEXR.InputFile(filename)
    dw = file.header()['dataWindow']
    sz = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
    chnls = ("R", "G", "B")[0:numChannels]
    rgb = [np.fromstring(file.channel(c, pxType), dtype=np.float32) for c in chnls]
    rgb = [c.reshape(sz[1], sz[0]) for c in rgb]
    rgb = np.asarray(rgb).transpose(1, 2, 0)
    if numChannels == 1:
        rgb = np.squeeze(rgb, axis=2)
    return rgb

#======================================================

def saveExr(data, filename):
    res = data.shape[0:2]
    assert res[0] == res[1], "Only square images supported so far."
    if len(data.shape) == 2:
        data = np.reshape(data, (res[0], res[1], 1))
    channels = data.shape[-1]
    assert (channels <= 3), "More than 3 channels not supported for EXR output"
    pixelsG = np.zeros_like(data[...,0], dtype=np.float32).tostring()
    pixelsB = np.zeros_like(data[...,0], dtype=np.float32).tostring()
    if channels >= 1:
        pixelsR = data[...,0].astype(np.float32).tostring()
    if channels == 1:
        pixelsG = pixelsR
        pixelsB = pixelsR
    if channels >= 2:
        pixelsG = data[...,1:2].astype(np.float32).tostring()
    if channels == 3:
        pixelsB = data[...,2:3].astype(np.float32).tostring()
    header = OpenEXR.Header(res[0], res[1])
    half_chan = Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))
    header['channels'] = dict([(c, half_chan) for c in "RGB"])
    exr = OpenEXR.OutputFile(filename, header)
    exr.writePixels({'R': pixelsR, 'G': pixelsG, 'B': pixelsB})
    exr.close()

#======================================================

def joinPath(p1, p2):
    return join(p1, p2)

#======================================================

def makeDir(path):
    mkdir(path)

#======================================================

def loadPNG(filename, numChannels=3):
    img = imread(filename).astype("float32")
    channels = img.shape[2]
    assert channels >= numChannels, "Image channel count is smaller than requested channel count."
    if channels > numChannels:
        img = img[..., 0:numChannels]
        img = np.squeeze(img)
    return img / 255.

#======================================================

def savePNG(image, filename):
    imwrite(filename, image)

#======================================================

def loadMultiPNG(path, numChannels=3, enforceSameSize=False):
    files = [join(path, f) for f in listdir(path) if f.endswith('.png')]
    imgList = []
    size = None
    for f in files:
        img = loadPNG(f, numChannels)
        if enforceSameSize:
            if size is not None and size != img.shape:
                assert False, "Images must have same size."    
            size = img.shape
        imgList.append(img)
    return imgList