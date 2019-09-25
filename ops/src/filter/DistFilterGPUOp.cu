#include <cuda_runtime.h> 
#include <stdio.h>
#include "DistFilterGPUOpData.h"
#include "../utils/FilterMathCuda.h"

//=============================================================

__device__ float sampleWeights(const DistFilterGPUOpData& data, const float dst)
{
    float sampleCountM1 = data.weightSize - 1.0f;
    float sampleDst = min(sampleCountM1, dst * sampleCountM1 / data.receptiveField);
    int lowerIndex = int(floor(sampleDst));
    int upperIndex = min(lowerIndex+1, int(sampleCountM1));
    float lerpWeight = sampleDst - lowerIndex;
    return lerpGPU(data.weights[lowerIndex], data.weights[upperIndex], lerpWeight);
}

//=============================================================

template <int dim>
__device__ void forwardPass(DistFilterGPUOpData data)
{
    int batchId = blockIdx.x * blockDim.x + threadIdx.x;
    int pointId = blockIdx.y * blockDim.y + threadIdx.y;

    if (batchId > data.batchSize - 1 || pointId > data.pointCount - 1)
        return;
    
    Vector<dim> pI;
    pI.slice(data.input, batchId, pointId, data.pointCount);
    Vector<dim> offset;

    for (int j=0; j<data.pointCount; j++)
    {
        if (j == pointId) continue;
        Vector<dim> pJ;
        pJ.slice(data.input, batchId, j, data.pointCount);
        pJ = toroidalMinShift(pJ, pI);
        Vector<dim> diff = pJ - pI;
        float dst = diff.norm();
        if (dst > data.receptiveField || dst < data.dstEps) continue;
        offset = offset + diff * (sampleWeights(data, dst) / dst);
    }
    Vector<dim> newPI = pI + offset;
    newPI.store(data.output, batchId, pointId, data.pointCount);
}

//=============================================================

__global__ void cudaKernel(DistFilterGPUOpData data)
{   
    if      (data.dimCount ==  1) forwardPass< 1>(data);
    else if (data.dimCount ==  2) forwardPass< 2>(data);
    else if (data.dimCount ==  3) forwardPass< 3>(data);
    else if (data.dimCount ==  4) forwardPass< 4>(data);
    else if (data.dimCount ==  5) forwardPass< 5>(data);
    else if (data.dimCount ==  6) forwardPass< 6>(data);
    else if (data.dimCount ==  7) forwardPass< 7>(data);
    else if (data.dimCount ==  8) forwardPass< 8>(data);
    else if (data.dimCount ==  9) forwardPass< 9>(data);
    else if (data.dimCount == 10) forwardPass<10>(data);
    else if (data.dimCount == 20) forwardPass<20>(data);
}

//=============================================================

extern "C" void launchFilter(DistFilterGPUOpData& data)
{
    const int batchBlocks = int(ceil(float(data.batchSize) / BATCH_BLOCK_SIZE));
    const int pointBlocks = int(ceil(float(data.pointCount) / POINT_BLOCK_SIZE));

    dim3 blockCount(batchBlocks, pointBlocks);
    dim3 blockSize(BATCH_BLOCK_SIZE, POINT_BLOCK_SIZE);
    
	cudaKernel <<< blockCount, blockSize >>> (data);
}

//=============================================================

// // template magic for auto selection of dimCount
// template <int dim>
// __device__ void forwardPassT2(DistFilterGPUOpData data, const int batchId, const int pointId)
// {
//     if (dim < 1) return;
//     if (data.dimCount == dim) forwardPassT<dim>(data, batchId, pointId);
//     else forwardPassT2<dim-1>(data, batchId, pointId);
// }