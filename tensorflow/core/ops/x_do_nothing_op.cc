#include "tensorflow/core/framework/op.h"

REGISTER_OP("XDoNothing").Input("in:float32").Output("out:float32");