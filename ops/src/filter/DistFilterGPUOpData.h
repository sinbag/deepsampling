#pragma once
#include <cuda_runtime.h> 

//=============================================================

struct DistFilterGPUOpData
{
	// input
	const float* input;
	const float* weights;

	// output
	float* output;

	// aux
	int batchSize;
	int pointCount;
	int dimCount;
	int weightSize;

	float receptiveField;
	float dstEps;
};