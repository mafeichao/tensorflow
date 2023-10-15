from tensorflow.python.ops import gen_x_do_nothing_op
from tensorflow.python.util.tf_export import tf_export
@tf_export('xx_do_nothing')
def xxx_zero_out(a):
    return gen_x_do_nothing_op.x_do_nothing()