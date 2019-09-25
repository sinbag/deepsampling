import numpy as np

#======================================================

def random(batchSize, pointCount, dimCount):
    return np.random.rand(batchSize, pointCount, dimCount).astype(np.float32)

#======================================================

def regular(pointCount, dimCount):
    pointsPerDim = round(pointCount**(1./dimCount))
    assert (pointCount == pointsPerDim**dimCount), "Regular %dD grid with %d points not possible" % (dimCount, pointCount)
    maxValue = 1 - 1/pointsPerDim
    p = np.meshgrid(*[np.linspace(i, j, pointsPerDim) for i,j in zip([0]*dimCount, [maxValue]*dimCount)])
    p = np.transpose(np.asarray(p), list(range(dimCount+1))[::-1])
    p = np.reshape(p, (pointCount, dimCount))
    return p.astype(np.float32)

#======================================================

def jittered(batchSize, pointCount, dimCount):
    reg = regular(pointCount, dimCount)
    cellSize = 1. / round(pointCount**(1./dimCount))
    noise = np.random.uniform(0, cellSize, (batchSize, pointCount, dimCount)).astype(np.float32)
    return np.expand_dims(reg, 0) + noise

#======================================================

# samples that are regular in the first gridDimCount dimensions
# and random in the remaining dimensions
def griddedRandom(batchSize, pointCount, dimCount, griddingDims=0):
    if griddingDims == 0:
        return random(batchSize, pointCount, dimCount)
    assert (dimCount >= griddingDims), "Number of gridded dimensions is larger than base dimensionality"
    reg = regular(pointCount, griddingDims)
    reg = np.tile(np.expand_dims(reg, 0), [batchSize, 1, 1])
    ran = random(batchSize, pointCount, dimCount - griddingDims)
    return np.append(reg, ran, -1)