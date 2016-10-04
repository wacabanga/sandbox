"""Proof-of-concept tf.Graph inversion. Drafted in 2016-10-3 meeting."""

import tensorflow as tf
from queue import Queue

__author__ = "Lawrence Wu and Edgar Minasyan"
__status__ = "Prototype"


def invert(output):
    """Inverts a Tensorflow graph.

    Args:
        output (tf.Tensor): The output of the original graph.
    Returns:
        [tf.Tensor]. A list containing the outputs of the inverse graph.
    """
    # TODO accept a list of outputs
    # TODO handle Tensor reuse
    # TODO more ops
    # TODO handle not in domain
    # TODO make a useful API
    # naming: "new" means of inverse graph, "old" means of original graph
    first_new_input = tf.placeholder(output.dtype)
    new_inputs = [first_new_input]
    new_outputs = []
    to_invert = Queue()
    to_invert.put((output, first_new_input))

    while not to_invert.empty():
        (old_tensor, new_tensor) = to_invert.get()
        old_op = old_tensor.op
        if old_op.type == 'Placeholder':
            new_outputs.append(new_tensor)
        elif old_op.type == 'Add':
            theta = tf.placeholder(old_tensor.dtype)
            new_inputs.append(theta)
            to_invert.put((old_op._inputs[0], theta))
            to_invert.put((old_op._inputs[1], new_tensor - theta))
        elif old_op.type == 'Mul':
            theta = tf.placeholder(old_tensor.dtype)
            new_inputs.append(theta)
            to_invert.put((old_op._inputs[0], theta))
            to_invert.put((old_op._inputs[1], new_tensor / theta))

    return new_outputs


def make_tests():
    """Makes (input, output) pairs. TODO: Write code to verify."""
    tests = []

    x = tf.placeholder(tf.float32)
    y = tf.placeholder(tf.float32)
    z = x + y
    w = invert(z)
    tests.append((z, w))

    x = tf.placeholder(tf.float32)
    y = tf.placeholder(tf.float32)
    z = tf.placeholder(tf.float32)
    w = x * y + z
    v = invert(w)
    tests.append((w, v))

    return tests


def print_usage():
    """Prints usage information."""
    print("Currently, tests can only be checked manually.")
    print("So, in the python interpreter:")
    print(">>> import graph_inversion")
    print(">>> tests = graph_inversion.make_tests()")
    print("will give a list of (input, output) pairs which can be examined.")

if __name__ == "__main__":
    print_usage()
