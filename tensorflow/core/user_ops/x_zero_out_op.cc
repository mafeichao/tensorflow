// op
#include "tensorflow/core/framework/op.h"
REGISTER_OP("XZeroOut").Input("in: int32").Output("out: int32");
// kernel
#include "tensorflow/core/framework/op_kernel.h"
using namespace tensorflow;
class XZeroOutOp : public OpKernel {
 public:
  explicit XZeroOutOp(OpKernelConstruction* context) : OpKernel(context) {}
  void Compute(OpKernelContext* context) override {
    const Tensor& input_tensor = context->input(0);
    auto flat_input_tensor = input_tensor.flat<int32>();
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),
                                                     &output_tensor));
    auto flat_output_tensor = output_tensor->template flat<int32>();
    for (int i = 0; i < flat_input_tensor.size(); ++i) {
      flat_output_tensor(i) = 0;
    }
  }
};
REGISTER_KERNEL_BUILDER(Name("XZeroOut").Device(DEVICE_GPU), XZeroOutOp)