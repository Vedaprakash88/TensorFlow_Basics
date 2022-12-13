# TensorFlow's HW1

import tensorflow as tf
import numpy as np

# The TensorFlow 2.0 has enabled eager execution by default. At the starting of algorithm,
# you need to use tf.compat.v1.disable_eager_execution() to disable eager execution.
tf.compat.v1.disable_eager_execution()

nm = np.mat('3 5')
a = tf.constant(nm, name="input_matrix_a")
b = tf.reduce_prod(a, name="prod_reduce_b")
c = tf.reduce_mean(a, name="mean_reduce_c")
d = tf.reduce_sum(a, name="sum_reduce_d")

e = tf.add(b, c, name="add_e")
f = tf.add(e, d, name="add_f")

sess = tf.compat.v1.Session()
result = sess.run(f)
print(result)
sess.graph.as_graph_def()
file_writer = tf.compat.v1.summary.FileWriter('./', sess.graph)
file_writer.close()
sess.close()
