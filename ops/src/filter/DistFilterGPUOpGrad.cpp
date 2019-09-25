#include "DistFilterGPUOpGrad.h"

//==============================================================================================//

REGISTER_OP("DistFilterGpuGrad")
	.Attr("receptive_field: float")
    .Attr("dst_eps: float")
    .Input("grad: float32")
    .Input("input: float32")
    .Input("weights: float32")
    .Output("grad_input: float32")
    .Output("grad_weights: float32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    	c->set_output(0, c->input(1));
        c->set_output(1, c->input(2));
    	return Status::OK();
	});

//==============================================================================================//

extern "C" void launchGradientFilter(DistFilterGPUOpGradData& data);

//==============================================================================================//

DistFilterGPUOpGrad::DistFilterGPUOpGrad(OpKernelConstruction* context) : OpKernel(context) 
{
	// get attributes
	OP_REQUIRES_OK(context, context->GetAttr("receptive_field", &receptiveField));
	OP_REQUIRES_OK(context, context->GetAttr("dst_eps", &dstEps));
}

//==============================================================================================//

void DistFilterGPUOpGrad::initializeData(OpKernelContext* context)
{
	// get inputs
	const Tensor& grad = context->input(0);
	const Tensor& input = context->input(1);
	const Tensor& weights = context->input(2);

	auto inputData = input.flat<float>();
	auto weightData = weights.flat<float>();
	auto gradData = grad.flat<float>();

	// create outputs
	Tensor* gradInput = NULL;
	Tensor* gradWeights = NULL;
	OP_REQUIRES_OK(context, context->allocate_output(0, input.shape(), &gradInput));
	OP_REQUIRES_OK(context, context->allocate_output(1, weights.shape(), &gradWeights));

	auto gradInputData = gradInput->flat<float>();
	auto gradWeightsData = gradWeights->flat<float>();

	// pipe into data structure
	data.weightSize = weightData.size();
	data.batchSize = input.shape().dim_size(0);
	data.pointCount = input.shape().dim_size(1);
	data.dimCount = input.shape().dim_size(2);
	
	data.receptiveField = receptiveField;
	data.dstEps = dstEps;

	data.grad = gradData.data();
	data.input = inputData.data();
	data.weights = weightData.data();
	
	data.grad_input = gradInputData.data();
	data.grad_weights = gradWeightsData.data();
}

//==============================================================================================//

void DistFilterGPUOpGrad::Compute(OpKernelContext* context)
{
	initializeData(context);
	launchGradientFilter(data);
}

//==============================================================================================//

REGISTER_KERNEL_BUILDER(Name("DistFilterGpuGrad").Device(DEVICE_GPU), DistFilterGPUOpGrad);
