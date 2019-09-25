import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import sparse_ops

CUSTOM_OP_PATH = "/path/to/libCustomOp.so"

customOpModule = tf.load_op_library(CUSTOM_OP_PATH)

distFilterGPU = customOpModule.dist_filter_gpu

@ops.RegisterGradient("DistFilterGpu")
def dist_filter_gpu_grad(op, grad):
    return customOpModule.dist_filter_gpu_grad(
        grad, op.inputs[0], op.inputs[1],
        op.get_attr("receptive_field"),
        op.get_attr("dst_eps"))
