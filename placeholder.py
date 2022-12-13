# The code has been created by Veda Chintha (3108224; VedaPrakash.Chintha@stud.srh-campus-berlin.de)

# importing required packages

import tensorflow as tf
import numpy as np
from numpy import random
import matplotlib.pyplot as plt

# The TensorFlow 2.0 has enabled eager execution by default. So, disabling the eager execution
# such that a session can be run.

tf.compat.v1.disable_eager_execution()

# creating a graph object
graph = tf.Graph()

# creating name_scopes
with graph.as_default():
    with tf.name_scope("Input_placeholder"):  # name_scope for input placeholder
        a = tf.compat.v1.placeholder(tf.float32, shape=None, name="input_a")  # placeholder for numpy array of float values, with shape none
    with tf.name_scope("Middle_section"):  # name_scope for Middle Section
        b = tf.reduce_prod(a, name="prod_b")  # product function
        c = tf.reduce_mean(a, name="mean_c")  # mean function
        d = tf.reduce_sum(a, name="sum_d")  # sum function
        e = tf.add(b, c, name="add_e")  # add function
    with tf.name_scope("Final_node"):  # name_scope for final node
        f = tf.multiply(d, e, name="mul_f")  # multiply function

    # creating a session to run the final node and passing the numpy array as input
    with tf.compat.v1.Session() as sess:
        mu, sigma = 1, 2  # mean and standard deviation
        inp_array = np.random.normal(mu, sigma, 100)  # creating a numpy array
        # inp2_array = [np.random.normal(mu, sigma, 100), np.random.normal(mu, sigma, 100)]
        input_dict = {a: inp_array}  # passing the array to placeholder as dictionary
        result = sess.run(f, feed_dict=input_dict)  # storing the output of session in a variable
        print(result)  # printing result

        # saving the graph
        sess.graph.as_graph_def()
        file_writer = tf.compat.v1.summary.FileWriter('./placeholder', sess.graph)
        file_writer.close()
        sess.close()
# plotting the input array (however this is line plot as scatter plot requires 2D array for x- and y-axis)
plt.plot(inp_array)
plt.show()

# # Redundant code for demonstration
# plt.scatter(inp2_array[0], inp2_array[1])
# plt.show()
