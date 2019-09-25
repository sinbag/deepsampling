#pragma once
#pragma GCC diagnostic ignored "-Wignored-attributes"

//==============================================================================================//

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "DistFilterGPUOpData.h"
#include "../utils/print_utils.h"

//==============================================================================================//

using namespace tensorflow;

//==============================================================================================//

class DistFilterGPUOp : public OpKernel 
{
	public:

		explicit DistFilterGPUOp(OpKernelConstruction* context);
		void Compute(OpKernelContext* context);
	
	private:
		
		void initializeData(OpKernelContext* context);

		float receptiveField;
		float dstEps;

		DistFilterGPUOpData data;
};

//==============================================================================================//

