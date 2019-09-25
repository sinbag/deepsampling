#pragma once
#pragma GCC diagnostic ignored "-Wignored-attributes"

//==============================================================================================//

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "DistFilterGPUOpGradData.h"
#include "../utils/print_utils.h"

//==============================================================================================//

using namespace tensorflow;

//==============================================================================================//

class DistFilterGPUOpGrad : public OpKernel 
{
	public:

		explicit DistFilterGPUOpGrad(OpKernelConstruction* context);
		void Compute(OpKernelContext* context);
	
	private:
		
		void initializeData(OpKernelContext* context);

		float receptiveField;
		float dstEps;

		DistFilterGPUOpGradData data;
};

//==============================================================================================//

