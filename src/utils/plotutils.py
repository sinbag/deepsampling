import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
import mathutils
import math

#=================================================

def show():
    plt.show()

#=================================================

def close(figure):
    plt.close(figure)

#=================================================

def multiPointPlot(
        points,
        title=None, 
        grid=None, 
        filename=None, 
        unit=True, 
        figRes=None,
        pointSize=4,
        connect=False):
    plotCount = len(points)
    fig = plt.figure(figsize=figRes)
    for n, p in enumerate(points):
        ax = fig.add_subplot(1, plotCount, n+1)
        if unit:
            ax.set_aspect(1)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
        if grid is not None:
            ax.set_xticks(np.arange(0, 1, grid), minor=True)
            ax.set_yticks(np.arange(0, 1, grid), minor=True)
            ax.grid(which='minor')
        if (p.shape[1] < 2):
            p = np.tile(p, (1, 2))
            p[:,1] = 0.5
        if title is not None:
            plt.title(title[n])             
        ax.scatter(p[:,0], p[:,1], c='black', s=pointSize);
        if connect:
            ax.plot(p[:,0], p[:,1])
    if filename is not None:
        plt.savefig(filename)
        plt.close('all')

#=================================================

def multiLinePlotSeparate(
        points, 
        title=None, 
        filename=None, 
        figRes=None):
    plotCount = len(points)
    fig = plt.figure(figsize=figRes)
    for n, p in enumerate(points):
        ax = fig.add_subplot(1, plotCount, n+1)
        if (p.shape[1] < 2):
            p = np.tile(p, (1, 2))
            p[:,1] = 0.5
        if title is not None:
            plt.title(title[n])
        ax.plot(p[:,0], p[:,1])
    if filename is not None:
        plt.savefig(filename)
        plt.close('all')

#=================================================

def linePlot(values, title=None, filename='onecurve.png'):
    safeScale = 0.1
    min = np.min(values)
    max = np.max(values)
    min -= abs(min) * safeScale
    max += abs(max) * safeScale
    fig, ax = plt.subplots()
    ax.set_xlim(0, values.shape[0])
    ax.set_ylim(min - mathutils.EPS, max + mathutils.EPS)
    if title is not None:
        plt.title(title)
    plt.plot(values)
    fig.tight_layout()
    plt.savefig(filename)
    return fig

#=================================================

# potentially irregular grid for surface plot
def customGrid2D(x, y):
    xDims = len(x.shape)
    yDims = len(y.shape)
    assert xDims <= 2 and yDims <= 2, "customGrid2D does only accept up to 2D inputs"
    if xDims == 1 and yDims == 1:
        return np.meshgrid(y, x)
    if xDims == 2 and yDims == 1:
        xSizeY = x.shape[0]
        yy = np.tile(np.expand_dims(y, 0), [xSizeY, 1])
        return x, yy
    if xDims == 1 and yDims == 2:
        ySizeX = y.shape[0]
        xx = np.tile(np.expand_dims(x, 0), [ySizeX, 1])
        return xx, y
    if xDims == 2 and yDims == 2:
        return x, y

#=================================================

def surfacePlot(
    values, 
    angles=(35, -97), 
    size=(25, 15), 
    wireframe=False,
    showAxes=True,
    title=None, 
    filename=None):

    fig = plt.figure(figsize=size)
    ax = plt.axes(projection="3d")
    ax.view_init(angles[0], angles[1])
    ax._axis3don = showAxes
    if wireframe:  
        ax.plot_wireframe(
            values[0], values[1], values[2], 
            color="black")
    else:
        ax.plot_surface(
            values[0], values[1], values[2], 
            edgecolor='black')
    fig.tight_layout()
    if title is not None:
        plt.title(title)
    if filename is not None:
        plt.savefig(filename)
        close(fig)

#=================================================

def multiLinePlot(valueList, title=None, legend=None,filename='lineplot.png',min=None,max=None):
    safeScale = 0.1
    if min is None:
        min = np.min(valueList)
        min -= abs(min) * safeScale
    if max is None:
        max = np.max(valueList)
        max += abs(max) * safeScale
    fig, ax = plt.subplots()
    ax.set_xlim(0, valueList[0].shape[0])
    ax.set_ylim(min - mathutils.EPS, max + mathutils.EPS)
    for n, v in enumerate(valueList):
        plt.plot(v, label=legend[n] if legend is not None else None)
    if legend is not None:
        ax.legend()
    if title is not None:
        plt.title(title)
    fig.tight_layout()
    plt.savefig(filename)
    return fig

#=================================================

def convertGriddedToArray(points, griddingDims):
    batchSize, pointCount, dimCount = points.shape
    pointsPerDim = round(pointCount**(1./griddingDims))
    valueDims = dimCount - griddingDims
    values = points[..., griddingDims:dimCount]
    outShape = np.full((griddingDims,), pointsPerDim)
    outShape = np.append([batchSize], outShape)
    outShape = np.append(outShape, valueDims)
    return np.reshape(values, outShape)