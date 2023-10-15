import tensorflow as tf
from  tensorflow.python.platform import test as test_lib
class ZeroOutTest(test_lib.TestCase):
  def testZeroOut(self):
    with self.test_session():
      result = tf.user_ops.zero_out([5, 4, 3, 2, 1])
      self.assertAllEqual(result.eval(), [0, 0, 0, 0, 0])