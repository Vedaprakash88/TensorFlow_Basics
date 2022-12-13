# TensorFlow's HW2

import tensorflow as tf
import numpy as np

# The TensorFlow 2.0 has enabled eager execution by default. At the starting of algorithm,
# you need to use tf.compat.v1.disable_eager_execution() to disable eager execution.

tf.compat.v1.disable_eager_execution()
graph = tf.Graph()
with graph.as_default():
    with tf.name_scope("Inputs"):
        nm = np.array([(1, 3, 6, 2, 2), (4, 6, 1, 5, 5), (8, 3, 1, 4, 4)])
        a = tf.constant(nm)
        a = tf.add(a, 0, name="input_matrix_a")
    with tf.name_scope("Hidden_Layer_1"):
        b = tf.reduce_prod(a, name="prod_reduce_b")
        c = tf.reduce_mean(a, name="mean_reduce_c")
        d = tf.reduce_sum(a, name="sum_reduce_d")
    with tf.name_scope("Hidden_Layer_2"):
        e = tf.add(b, c, name="add_e")
    with tf.name_scope("Hidden_Layer_3"):
        f = tf.add(e, d, name="add_f")

    with tf.compat.v1.Session() as sess:
        result = sess.run(f)
        print(result)
        sess.graph.as_graph_def()
        file_writer = tf.compat.v1.summary.FileWriter('./', sess.graph)
        file_writer.close()
        sess.close()
