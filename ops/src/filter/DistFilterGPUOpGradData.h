#pragma once
#include <cuda_runtime.h> 

//=============================================================

struct DistFilterGPUOpGradData
{
	// input
	const float* grad;
	const float* input;
	const float* weights;

	// output
	float* grad_input;
	float* grad_weights;

	// aux
	int batchSize;
	int pointCount;
	int dimCount;
	int weightSize;

	float receptiveField;
	float dstEps;
};

