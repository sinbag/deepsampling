#include "DistFilterGPUOp.h"

//=============================================================

REGISTER_OP("DistFilterGpu")
	.Attr("receptive_field: float")
	.Attr("dst_eps: float")
	.Input("input: float")
	.Input("weights: float")
	.Output("output: float")
	.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
		c->set_output(0, c->input(0));
		return Status::OK();
	});

//=============================================================

extern "C" void launchFilter(DistFilterGPUOpData& data);

//=============================================================

DistFilterGPUOp::DistFilterGPUOp(OpKernelConstruction* context) : OpKernel(context) 
{
	// get attributes
	OP_REQUIRES_OK(context, context->GetAttr("receptive_field", &receptiveField));
	OP_REQUIRES_OK(context, context->GetAttr("dst_eps", &dstEps));

	// basic validity checks
	OP_REQUIRES(context, receptiveField > 0,
		errors::InvalidArgument("Need receptive_field > 0, got ", receptiveField));
	OP_REQUIRES(context, dstEps > 0,
		errors::InvalidArgument("Need dst_eps > 0, got ", dstEps));
}

//=============================================================

void DistFilterGPUOp::initializeData(OpKernelContext* context)
{	
	// get inputs
	const Tensor& input = context->input(0);
	const Tensor& weights = context->input(1);
	
	OP_REQUIRES(context, input.shape().dims() == 3,
		errors::InvalidArgument("UnstructuredDistFilter expects a 3D tensor as input."));
	OP_REQUIRES(context, TensorShapeUtils::IsVector(weights.shape()),
		errors::InvalidArgument("UnstructuredDistFilter expects a 1D tensor as weight input."));

	auto inputData = input.flat<float>();
	auto weightData = weights.flat<float>();

	// create output
	Tensor* output = NULL;
	OP_REQUIRES_OK(context, context->allocate_output(0, input.shape(), &output));
	auto outputData = output->flat<float>();

	// pipe into data structure
	data.weightSize = weightData.size();
	data.batchSize = input.shape().dim_size(0);
	data.pointCount = input.shape().dim_size(1);
	data.dimCount = input.shape().dim_size(2);
	
	data.receptiveField = receptiveField;
	data.dstEps = dstEps;

	data.input = inputData.data();
	data.weights = weightData.data();
	data.output = outputData.data();
}

//=============================================================

void DistFilterGPUOp::Compute(OpKernelContext* context)
{	
	initializeData(context);
	launchFilter(data);
}

//=============================================================

REGISTER_KERNEL_BUILDER(Name("DistFilterGpu").Device(DEVICE_GPU), DistFilterGPUOp);
