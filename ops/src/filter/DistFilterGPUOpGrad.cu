#include <cuda_runtime.h> 
#include <stdio.h>
#include "DistFilterGPUOpGradData.h"
#include "../utils/FilterMathCuda.h"

//=============================================================

template <int dim>
__device__ void interactionGradUpdate(
    DistFilterGPUOpGradData& data,
    Matrix<dim>& sum1,
    Vector<dim>& sum2,
    const Vector<dim>& p,
    const Vector<dim>& pR,
    const Vector<dim>& gradP,
    const Vector<dim>& gradPR)
{
    Vector<dim> p_ = toroidalMinShift(p, pR);
    Vector<dim> diff = p_ - pR;
    float dst = diff.norm();
    if (dst > data.receptiveField || dst < data.dstEps) return;
    float dstSqr = dst * dst;
    float sampleCountM1 = data.weightSize - 1.0f;
    float dConvFactor = sampleCountM1 / data.receptiveField;
    float sampleDst = min(sampleCountM1, dst * sampleCountM1 / data.receptiveField);
    int lowerIndex = int(floor(sampleDst));
    int upperIndex = min(lowerIndex+1, int(sampleCountM1));
    float lowerWeight = data.weights[lowerIndex];
    float upperWeight = data.weights[upperIndex];
    float weightDiff = lowerWeight - upperWeight;
    float lerpWeight = sampleDst - lowerIndex;
    float weightInterp = lerpGPU(lowerWeight, upperWeight, lerpWeight);
    float gradWeightsUpdate = diff.dot(gradPR) / dst;
    
    // scatter weight gradient
    atomicAdd(&data.grad_weights[lowerIndex], (1.0f - lerpWeight) * gradWeightsUpdate);
    atomicAdd(&data.grad_weights[upperIndex], lerpWeight * gradWeightsUpdate);

    for (int k=0; k<dim; k++)
    {
        Vector<dim> diffSqr = diff * diff(k);
        Vector<dim> dstVec = oneHot<dim>(k) * dst;
        Vector<dim> gradInputUpdate = (diffSqr * (dConvFactor * weightDiff) + ((diffSqr / dst) - dstVec) * weightInterp) / dstSqr;

        //accumulate input gradient
        sum1.col(k) = sum1.col(k) + gradInputUpdate;
        sum2(k) = sum2(k) - gradInputUpdate.dot(gradP);
    } 
}

//=============================================================

template <int dim>
__device__ void backwardPass(DistFilterGPUOpGradData data)
{
    int batchId = blockIdx.x * blockDim.x + threadIdx.x;
    int pointId = blockIdx.y * blockDim.y + threadIdx.y;

    if (batchId > data.batchSize - 1 || pointId > data.pointCount - 1)
        return;
    
    Vector<dim> pI;
    pI.slice(data.input, batchId, pointId, data.pointCount);
    Vector<dim> gradI;
    gradI.slice(data.grad, batchId, pointId, data.pointCount);
    Vector<dim> gradInputI(gradI);
    Matrix<dim> sum1;
    Vector<dim> sum2;
    
    for (int j=0; j<data.pointCount; j++)
    {
        if (j == pointId) continue;
        Vector<dim> pJ;
        pJ.slice(data.input, batchId, j, data.pointCount);
        Vector<dim> gradJ;
        gradJ.slice(data.grad, batchId, j, data.pointCount);
        interactionGradUpdate(data, sum1, sum2, pJ, pI, gradJ, gradI);
    }    
    gradInputI = gradInputI + (sum1 * gradI) + sum2;
    gradInputI.store(data.grad_input, batchId, pointId, data.pointCount);
}

//=============================================================

__global__ void cudaKernel(DistFilterGPUOpGradData data)
{
    if      (data.dimCount ==  1) backwardPass< 1>(data);
    else if (data.dimCount ==  2) backwardPass< 2>(data);
    else if (data.dimCount ==  3) backwardPass< 3>(data);
    else if (data.dimCount ==  4) backwardPass< 4>(data);
    else if (data.dimCount ==  5) backwardPass< 5>(data);
    else if (data.dimCount ==  6) backwardPass< 6>(data);
    else if (data.dimCount ==  7) backwardPass< 7>(data);
    else if (data.dimCount ==  8) backwardPass< 8>(data);
    else if (data.dimCount ==  9) backwardPass< 9>(data);
    else if (data.dimCount == 10) backwardPass<10>(data);
    else if (data.dimCount == 20) backwardPass<20>(data);
}

//=============================================================

extern "C" void launchGradientFilter(DistFilterGPUOpGradData& data)
{
    // zero out weight grad memory for proper atomic add behaviour
    cudaMemset(data.grad_weights, 0, data.weightSize * sizeof(float));
    
    // run gradient filter
    const int batchBlocks = int(ceil(float(data.batchSize) / BATCH_BLOCK_SIZE));
    const int pointBlocks = int(ceil(float(data.pointCount) / POINT_BLOCK_SIZE));
    dim3 blockCount(batchBlocks, pointBlocks, 1);
    dim3 blockSize(BATCH_BLOCK_SIZE, POINT_BLOCK_SIZE, 1);
	cudaKernel <<< blockCount, blockSize >>> (data);
}

//=============================================================
